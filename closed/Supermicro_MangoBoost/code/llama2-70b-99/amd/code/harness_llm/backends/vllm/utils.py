import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

def check_parallelism_configuration(dp, tp):
    if dp % tp != 0:
        error_message = f"TP={tp} and DP={dp} are not compatible"
        log.error(error_message)
        raise ValueError(error_message)