from types import SimpleNamespace

from max_live_v1 import MaxLiveController


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode


def test_controller_starts_and_stops_only_owned_receiver(tmp_path):
    (tmp_path / "generator_receiver.py").write_text("# test")
    process = FakeProcess()
    calls = []

    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    controller = MaxLiveController(tmp_path, popen=popen, port_probe=lambda *_: True)
    assert controller.start()["owned_by_app"] is True
    assert calls[0][0][0][-1].endswith("generator_receiver.py")
    controller.stop()
    assert process.terminated is True


def test_existing_receiver_is_reported_without_starting_duplicate(tmp_path):
    controller = MaxLiveController(
        tmp_path,
        popen=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
        port_probe=lambda *_: False,
    )
    result = controller.start()
    assert result["receiver_active"] is True
    assert result["owned_by_app"] is False
    assert "already active" in result["message"]


def test_connection_test_uses_dedicated_non_rendering_message(tmp_path):
    sent = []
    client = SimpleNamespace(send_message=lambda address, value: sent.append((address, value)))
    controller = MaxLiveController(tmp_path, client_factory=lambda *_: client)
    result = controller.send_test()
    assert result["sent"] is True
    assert sent == [("/hyponoia/test", 1)]
