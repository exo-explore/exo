import argparse
import multiprocessing as mp

from loguru import logger

from exo.master.main import main as master_main
from exo.shared.constants import EXO_LOG
from exo.shared.logging import logger_cleanup, logger_setup
from exo.worker.main import main as worker_main


def main():
    parser = argparse.ArgumentParser(prog="exo")
    parser.add_argument(
        "-v", "--verbose", action="store_const", const=1, dest="verbosity", default=0
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        action="store_const",
        const=2,
        dest="verbosity",
        default=0,
    )
    args = parser.parse_args()
    if type(args.verbosity) is not int:  # type: ignore
        raise TypeError("Verbosity was parsed incorrectly")
    logger_setup(EXO_LOG, args.verbosity)
    logger.info("starting exo")

    # This is for future PyInstaller compatibility
    mp.set_start_method("spawn", force=True)

    worker = mp.Process(target=worker_main, args=(EXO_LOG, args.verbosity))
    master = mp.Process(target=master_main, args=(EXO_LOG, args.verbosity))
    worker.start()
    master.start()
    worker.join()
    master.join()

    logger_cleanup()
