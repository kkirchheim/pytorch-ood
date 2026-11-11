import logging
import os
import socket

import pytest
import torch

from tests.helpers.hardware import missing_hardware

DOWNLOAD_TESTS = "PYTORCH_OOD_DOWNLOAD_TESTS"

# PIL logs every image plugin it loads at DEBUG level
logging.getLogger("PIL").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption(
        "--require-hardware",
        action="store_true",
        help="fail tests marked with @requires(...) instead of skipping them when the hardware "
        "is missing",
    )


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Tests must not reach the network unless downloads are explicitly enabled."""
    if os.environ.get(DOWNLOAD_TESTS):
        return
    connect = socket.socket.connect

    def guarded(sock, address):
        if sock.family == socket.AF_UNIX:
            return connect(sock, address)
        raise RuntimeError(
            f"test tried to connect to {address!r}; network tests must be gated on {DOWNLOAD_TESTS}"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded)


def _missing(config, what: str):
    if config.getoption("--require-hardware"):
        pytest.fail(f"hardware: {what} (--require-hardware)", pytrace=False)
    pytest.skip(f"hardware: {what}")


def pytest_runtest_setup(item):
    if item.get_closest_marker("download") and not os.environ.get(DOWNLOAD_TESTS):
        pytest.skip(f"downloads data; set {DOWNLOAD_TESTS}=1 to run")
    for marker in item.iter_markers(name="hardware"):
        missing = missing_hardware(*marker.args, **marker.kwargs)
        if missing is not None:
            _missing(item.config, missing)


@pytest.fixture
def accelerator(request) -> torch.device:
    """The first available non-CPU device. Without one, the test is skipped, or fails under
    ``--require-hardware``."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    _missing(request.config, "needs an accelerator")


@pytest.fixture
def seed() -> torch.Generator:
    """A freshly seeded generator, so tests do not depend on the global RNG or on test order."""
    return torch.Generator().manual_seed(0)
