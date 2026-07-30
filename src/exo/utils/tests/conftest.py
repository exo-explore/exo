import multiprocessing as mp

import pytest


@pytest.fixture(scope="session", autouse=True)
def mp_force_spawn():
    mp.set_start_method("spawn", force=True)
