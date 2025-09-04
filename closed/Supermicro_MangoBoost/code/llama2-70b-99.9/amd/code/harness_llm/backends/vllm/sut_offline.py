import multiprocessing as mp
from multiprocessing import connection as conn
import os
import logging
import time
import numpy as np
import array
import mlperf_loadgen as lg

from harness_llm.backends.vllm.vllm_engine import (
    initialize_engine_and_generate,
    LLM_MODEL_LOAD_DONE,
    LLM_GENERATION_DONE,
)
from harness_llm.loadgen.sut import SUT, SUTConfig
from harness_llm.backends.vllm.utils import check_parallelism_configuration
from harness_llm.common.rpd_trace_utils import rpd_trace_range_non_timed
from threading import Thread, Event
import harness_llm.backends.vllm.constants as constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

class LLMProc:
    def __init__(
        self,
        device_ids: tuple[int, ...],
        qdata_in: conn.Connection,
        qdata_out: conn.Connection,
        llm_config: dict,
        sampling_params: dict,
    ):
        self.llm_config = llm_config
        self.sampling_params = sampling_params
        self.qdata_in = qdata_in
        self.qdata_out = qdata_out
        self.qstatus_out = mp.Queue()
        self.device_ids = device_ids

        log.info(f"llm_config={self.llm_config}")
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join((str(i) for i in self.device_ids))

        self.llm_proc = mp.Process(
            target=initialize_engine_and_generate,
            args=(
                self.device_ids,
                self.qdata_in,
                self.qdata_out,
                self.qstatus_out,
                self.llm_config,
                self.sampling_params,
            ),
        )

        self.llm_proc.start()

    def check_llm_loaded(self) -> bool:
        while True:
            status = self.qstatus_out.get()
            if status == LLM_MODEL_LOAD_DONE:
                log.info(f"LLM is loaded")
                return True


class SUTvLLMOffline(SUT):
    def __init__(self, config: dict):
        log.info(f"Init SUTvLLM")

        super().__init__(
            SUTConfig(
                model=config["llm_config"]["model"],
                dataset_path=config["harness_config"]["dataset_path"],
                total_sample_count=(
                    config["harness_config"]["total_sample_count"]
                    if "total_sample_count" in config["harness_config"]
                    else 24576
                ),
                model_max_length=(
                    config["harness_config"]["model_max_length"]
                    if "model_max_length" in config["harness_config"]
                    else None
                ),
                debug=False,
            )
        )

        self.llm_config: dict = config["llm_config"]
        self.harness_config: dict = config["harness_config"]
        self.sampling_params: dict = config["sampling_params"]

        self.tp = (
            self.llm_config["tensor_parallel_size"]
            if "tensor_parallel_size" in self.llm_config
            else 1
        )
        self.dp = (
            self.harness_config["data_parallel_size"]
            if "data_parallel_size" in self.harness_config
            else 8  # TODO: get based on the number of devices available & tp config.
        )
        self.effective_dp = self.dp


        self.warm_up_sample_count_per_server = self.harness_config[
            "warm_up_sample_count_per_server"
        ]
        self.warm_up_sample = constants.WarmUp.ENCODED_SAMPLES.get(config["benchmark_name"], None)
        self.enable_warm_up = self.harness_config["enable_warm_up"] and (self.warm_up_sample is not None)
        self.sorting = self.harness_config.get("sort_samples", "ignore")

        self.qdata_in_senders = []
        self.qdata_out_receivers = []
        self.qstatus_out = mp.Queue()

        self.llm_procs = []
        self.llm_objs = []
        self.warm_up_done = []

        self.sample_ids = []
        self.completion_threads = []
        self.start_t = time.time()
        self.infer_start_t = time.time()

    @rpd_trace_range_non_timed("SUT:Main")
    def init_llms(self):
        check_parallelism_configuration(self.dp, self.tp)

        qdata_out = mp.Queue()
        self.qdata_out_receivers.append(qdata_out)

        for device in range(0, self.dp, self.tp):
            device_ids = tuple(range(device, device + self.tp))
            qdata_in_receiver, qdata_in_sender = mp.Pipe(False)
            self.qdata_in_senders.append(qdata_in_sender)

            self.warm_up_done.append(Event())
            llm_obj = LLMProc(
                device_ids,
                qdata_in_receiver,
                qdata_out,
                self.llm_config,
                self.sampling_params,
            )

            self.llm_objs.append(llm_obj)
        self.effective_dp = self.dp // self.tp

        for obj in self.llm_objs:
            obj.check_llm_loaded()

    @rpd_trace_range_non_timed("SUT:Main")
    def start_completion_threads(self):
        for i in range(self.effective_dp):
            self.completion_threads.append(Thread(target=self.completion_pipe, args=(i,), daemon=True))
            self.completion_threads[-1].start()

    def start_completion_loop(self):
        self.completion_threads.append(Thread(target=self.completion_queue, args=(self.effective_dp,), daemon=True))
        self.completion_threads[-1].start()

    @rpd_trace_range_non_timed("SUT:Main")
    def warm_up(self):
        log.info("Running warm-up")
        for i in range(self.effective_dp):
            prompt_token_ids = [self.warm_up_sample] * self.warm_up_sample_count_per_server
            query_types = ["WARMUP_QUERY_TYPE"] * self.warm_up_sample_count_per_server
            self.qdata_in_senders[i].send((0, None, prompt_token_ids, query_types))
        for i in range(self.effective_dp):
            log.info(f"Waiting for server[{i}] warm-up to complete...")
            self.warm_up_done[i].wait()
        log.info("Running warm-up finished")

    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        for t in self.completion_threads:
            t.join()
        log.info(f"Total time spent with run: {time.time() - self.start_t}")

    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):
        log.info(f"SUT start")
        self.init_llms()
        self.start_completion_loop()
        if self.enable_warm_up:
            self.warm_up()
        self.infer_start_t = time.time()
        log.info(
            f"Time spent from start to inference start: {self.infer_start_t - self.start_t}"
        )

    @rpd_trace_range_non_timed("SUT:Main")
    def make_ranges(self, query_samples):
        query_chunk_size = (
            len(query_samples) + self.effective_dp - 1
        ) // self.effective_dp
        ranges = []
        for i in range(self.effective_dp):
            start = i * query_chunk_size
            end = start + query_chunk_size
            if end > len(query_samples):
                end = None
            ranges.append((start, end))
        return ranges

    @rpd_trace_range_non_timed("SUT:Main")
    def sort_by_length(self, query_samples, weight=1):
        reord_start = time.time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(
            query_samples, ranges[0][1] - ranges[0][0]
        )
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(
                key=lambda sample: weight
                * len(self.data_object.input_ids[sample.index])
            )
            reordered_samples.extend(chunk)
        reord_dur = (time.time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples

    @rpd_trace_range_non_timed("SUT:Main")
    def sort_lexicog(self, query_samples):
        reord_start = time.time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(
            query_samples, ranges[0][1] - ranges[0][0]
        )
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(key=lambda sample: self.data_object.input_ids[sample.index])
            reordered_samples.extend(chunk)
        reord_dur = (time.time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples

    @rpd_trace_range_non_timed("SUT:Main")
    def sort_modulo(self, query_samples, weight=1):
        reord_start = time.time_ns()

        query_samples.sort(
            key=lambda sample: weight
                * len(self.data_object.input_ids[sample.index])
        )

        parts = [[] for _ in range(self.effective_dp)]
        for index, value in enumerate(query_samples):
            part_index = index % self.effective_dp
            parts[part_index].append(value)

        reordered_samples = []
        for i in range(len(parts)):
            reordered_samples.extend(parts[i])

        ranges = self.make_ranges(query_samples)
       
        reord_dur = (time.time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples

    @rpd_trace_range_non_timed("SUT:Main")
    def even_out_token_count(self, query_samples, query_chunk_size):
        full_buckets = []
        buckets = [[] for _ in range(self.effective_dp)]
        bucket_sizes = [0 for _ in range(self.effective_dp)]
        for sample in query_samples:
            smallest_bucket = bucket_sizes.index(min(bucket_sizes))
            buckets[smallest_bucket].append(sample)
            bucket_sizes[smallest_bucket] += len(
                self.data_object.input_ids[sample.index]
            )
            if len(buckets[smallest_bucket]) == query_chunk_size and len(buckets) > 1:
                full_buckets.append(buckets[smallest_bucket])
                del buckets[smallest_bucket]
                del bucket_sizes[smallest_bucket]
        reordered_samples = []
        for bucket in full_buckets + buckets:
            reordered_samples.extend(bucket)
        return reordered_samples

    @rpd_trace_range_non_timed("SUT:Main")
    def sort_samples(self, query_samples):
        sorting_strategies = ("ascending", "descending", "lexicographic", "modulo_desc", "ignore")
        assert self.sorting in sorting_strategies, f"No such sorting strategy '{self.sorting}'"
        if self.sorting != "ignore":
            log.info(f"Sorting samples in {self.sorting} order")
        if self.sorting == "ascending":
            return self.sort_by_length(query_samples, weight=1)
        elif self.sorting == "descending":
            return self.sort_by_length(query_samples, weight=-1)
        elif self.sorting == "lexicographic":
            return self.sort_lexicog(query_samples)
        elif self.sorting == "modulo_desc":
            return self.sort_modulo(query_samples, weight=-1)
        else:
            return (self.make_ranges(query_samples), query_samples)

    @rpd_trace_range_non_timed("SUT:Main")
    def post_proc(self, response):
        start, end, output_token_ids = response
        log.info(
            f"Got item  |  start, end = {start}, {end}  |  n outputs = {len(output_token_ids)}"
        )

        output_sample_ids = self.sample_ids[start:end]
        assert len(output_sample_ids) == len(output_token_ids)

        log.info(f"Signaling LoadGen output")

        try:
            for i in range(len(output_token_ids)):
                response_array = array.array(
                    "B", np.array(output_token_ids[i], np.int32).tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        output_sample_ids[i], bi[0], bi[1], len(output_token_ids[i])
                    )
                ]
                lg.QuerySamplesComplete(response)
        except:
            log.info(f"Error sending completed response to LoadGen")

    @rpd_trace_range_non_timed("SUT:Main")
    def completion_pipe(self, device):
        while True:
            try:
                response = self.qdata_out_receivers[device].recv()
                if response == LLM_GENERATION_DONE:
                    log.info(f"Query chunk done for GPU {device}")
                    break
                if self.enable_warm_up and not self.warm_up_done[device].is_set():
                    self.warm_up_done[device].set()
                else:
                    self.post_proc(response)
            except:
                logging.exception("Exception during completion")
                break

    @rpd_trace_range_non_timed("SUT:Main")
    def completion_queue(self, devices):
        warm_up_in_progress_count = devices
        while True:
            try:
                response = self.qdata_out_receivers[-1].get()
                if response == LLM_GENERATION_DONE:
                    log.info(f"Query chunk done. Remaining GPUs: {devices}")
                    devices -= 1
                    if devices <= 0:
                        break
                else:
                    if self.enable_warm_up and warm_up_in_progress_count > 0:
                        warm_up_in_progress_count -= 1
                        self.warm_up_done[warm_up_in_progress_count].set()
                    else:
                        self.post_proc(response)
            except:
                logging.exception("Exception during completion")
                break

    @rpd_trace_range_non_timed("SUT:Main")
    def issue_queries(self, query_samples):
        log.info(f"Issue queries  |  number of queries = {len(query_samples)}")
        ranges, query_samples = self.sort_samples(query_samples)
        self.sample_ids = [query_samples[i].id for i in range(len(query_samples))]
        prompt_token_ids = [
            self.data_object.input_ids[query_samples[i].index]
            for i in range(len(query_samples))
        ]

        query_types = [
            self.data_object.query_types[query_samples[i].index]
            for i in range(len(query_samples))
        ]

        log.info(
            f"Converted queries to prompt tokens  |  number of queries = {len(prompt_token_ids)}"
        )

        for i, (start, end) in enumerate(ranges):
            self.qdata_in_senders[i].send((start, end, prompt_token_ids[start:end], query_types[start:end]))
            log.info(f"Put prompt tokens in pipe #{i}")

        for i in range(self.effective_dp):
            self.qdata_in_senders[i].send(None)
            self.qdata_in_senders[i].close()
