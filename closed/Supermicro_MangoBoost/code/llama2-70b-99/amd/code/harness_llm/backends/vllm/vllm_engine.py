from vllm import LLM, SamplingParams
import logging
import multiprocessing as mp
from multiprocessing import connection as conn
import os, gc

import harness_llm.common.numa_helpers as nh
from harness_llm.common.rpd_trace_utils import rpd_trace_range

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# flags to notify the processes.
LLM_MODEL_LOAD_DONE = "LLM_MODEL_LOAD_DONE"
LLM_GENERATION_DONE = "LLM_GENERATION_DONE"

HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))


@rpd_trace_range("SUT:Worker")
def _run_vllm(llm, prompt_token_ids, sampling_params):
    return llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True,
    )


@rpd_trace_range("SUT:Worker")
def initialize_engine_and_generate(
    device_ids: tuple[int, ...],
    qdata_in: conn.Connection,
    qdata_out: conn.Connection,
    qstatus_out: mp.Queue,
    llm_config: dict,
    sampling_params_config: dict = {"temperature": 0.0, "max_tokens": 1024},
):
    """
    Initialize the llm engine and generate the responses.
    """
    for id in device_ids:
        nh.set_affinity_by_device(int(id))

    # process sampling params config
    stop_seq_ids_config: dict = None
    if "stop_seq_ids_config" in sampling_params_config.keys():
        stop_seq_ids_config = sampling_params_config['stop_seq_ids_config']
        del sampling_params_config['stop_seq_ids_config']

    # Initialize the vllm engine.
    llm = LLM(**llm_config)
    sampling_params = SamplingParams(**sampling_params_config)
    qstatus_out.put(LLM_MODEL_LOAD_DONE)

    # The GC is going to be called after certain number of steps
    sample_count = 0
    is_gc_limit_specified = HARNESS_GC_LIMIT > 0
    if is_gc_limit_specified:
        gc.collect()
        gc.disable()

    # Generates the completions for the input prompt tokens.
    while True:
        try:
            item = qdata_in.recv()
            if item is None:
                log.info(f"LLM is stopping")
                qdata_out.put(LLM_GENERATION_DONE)
                break

            start, end, prompt_token_ids, query_types = item
            sample_count += len(prompt_token_ids)
            if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                gc.collect()
                sample_count = 0

            sampling_params_list = []
            pred_output_tokens = None

            if stop_seq_ids_config:
                for query_type in query_types:
                    sampling_param = SamplingParams(**sampling_params_config)
                    if query_type in stop_seq_ids_config.keys():
                        stop_seq_ids = stop_seq_ids_config[query_type]['stop_seq_ids']
                        sampling_param.stop_seq_ids = tuple(stop_seq_ids)
                    sampling_params_list.append(sampling_param)
                pred_output_tokens = _run_vllm(llm, prompt_token_ids, sampling_params_list)
            else:
                sampling_params = SamplingParams(**sampling_params_config)
                pred_output_tokens = _run_vllm(llm, prompt_token_ids, sampling_params)

            log.info(f"VLLM finished")

            processed_output = [
                output.outputs[0].token_ids for output in pred_output_tokens
            ]
            log.info(f"output tokens collected")

            qdata_out.put((start, end, processed_output))
            log.info(f"Processed output | start, end = {start}, {end}")
        except:
            logging.exception("Exception running vLLM")
            break
    log.info(f"vLLM engine thread finished for {device_ids=}")
