import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _no_network_no_sleep(monkeypatch):
    """Tests never touch the network or wait on retry back-off.

    A test that needs a fake HTTP response patches requests.get itself, which
    overrides this.
    """
    import requests

    import data_lib

    def blocked(*a, **k):
        raise requests.ConnectionError("network disabled in tests")

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(data_lib.time, "sleep", lambda s: None)
    monkeypatch.setattr(data_lib, "_gnews_down", False)   # the circuit breaker is process-wide state
