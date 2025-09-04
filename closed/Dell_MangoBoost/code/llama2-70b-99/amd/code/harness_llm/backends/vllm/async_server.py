import logging
import multiprocessing as mp
import os
import asyncio
import logging
import harness_llm.common.numa_helpers as nh
import threading
from harness_llm.common.rpd_trace_utils import rpd_trace_range_async, rpd_trace_range, rpd_trace_range_non_timed
import queue
import gc
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

class AsyncServer:

    SIG_RUN = 1
    SIG_STOP = 2

    def __init__(
        self,
        devices,
        qdata_in,
        qdata_out,
        qstatus_out: mp.Queue,
        llm_config: dict,
        sampling_params: dict,
    ):
        self.qdata_in = qdata_in
        self.qdata_out = qdata_out
        self.qstatus_out = qstatus_out
        self.devices = devices
        self.engine = None
        self.process = None
        self.llm_config = llm_config
        self.sampling_params = sampling_params

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        self.process = mp.Process(target=self.launch)
        self.process.start()

    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        nh.set_affinity_by_device(self.devices[0])

        self.log(f"llm_config={self.llm_config}")
        self.log(f"sampling_params={self.sampling_params}")

        self.sampling_params = SamplingParams(**self.sampling_params)

        engine_args = AsyncEngineArgs(
            **self.llm_config
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args, start_engine_loop=True)

        self.signal_running()
        self.run()

    def signal_running(self):
        self.qstatus_out.put_nowait(AsyncServer.SIG_RUN)

    @rpd_trace_range("SUT:Worker")
    def run(self):
        async_event_loop = asyncio.new_event_loop()
        async_thread = threading.Thread(target=run_async_event_loop, args=([async_event_loop]), daemon=True)
        async_thread.start()
        self.log("Processing started...")
        while True:
            try:
                sample = self.qdata_in.get()
                if sample is None:
                    self.error("qdata_in got end signal...")
                    break
                asyncio.run_coroutine_threadsafe(self.generate_v2(sample), async_event_loop)
            except queue.Empty:
                break


    def is_running(self):
        try:
            return self.qstatus_out.get_nowait() == AsyncServer.SIG_RUN
        except:
            return False


    async def generate_v2(self, sample):
        prompt_token_ids = sample[0]
        sample_ids = sample[1]
        is_warm_up = sample[2]
        await asyncio.wait([asyncio.create_task(self.generate((prompt_token_ids[i], sample_ids[i], is_warm_up))) for i in range(len(sample_ids))])


    async def generate(self, sample):
        prompt_token_ids = sample[0]
        request_id = str(sample[1])
        is_warm_up = sample[2]
        results_generator = self.engine.generate({"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id)
        output_token_ids = []
        is_first_token = True
        async for request_output in results_generator:
            output_token_ids = request_output.outputs[0].token_ids
            if is_first_token:
                is_first_token = False
                self.qdata_out.send([output_token_ids, request_id, True, is_warm_up])
        self.qdata_out.send([output_token_ids, request_id, False, is_warm_up])


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")


def run_async_event_loop(async_event_loop):
    asyncio.set_event_loop(async_event_loop)
    async_event_loop.run_forever()
