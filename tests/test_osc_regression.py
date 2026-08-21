import generator_receiver
from types import SimpleNamespace


class _FakeThread:
    def __init__(self, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


def test_render_handler_dispatches_background_work_without_blocking(monkeypatch):
    threads = []

    def make_thread(*, target, args, daemon):
        thread = _FakeThread(target, args, daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(generator_receiver.threading, "Thread", make_thread)
    generator_receiver.render_handler("/generator/render", "D3")
    assert len(threads) == 1
    assert threads[0].started is True
    assert threads[0].daemon is True
    assert threads[0].args[0] == "D3"


def test_harmony_handlers_preserve_physio_max_control_state():
    generator_receiver.root_handler("/harmony/root", 14)
    generator_receiver.scale_handler("/harmony/scale", "dorian")
    generator_receiver.confidence_handler("/harmony/confidence", 1.4)
    assert generator_receiver.HARMONY_STATE == {
        "root": 2,
        "scale": "dorian",
        "confidence": 1.0,
    }


def test_render_sends_path_before_ready_to_prevent_max_preload_race(monkeypatch):
    messages = []

    class FakeClient:
        def send_message(self, address, value):
            messages.append((address, value))

    monkeypatch.setattr(generator_receiver, "client", FakeClient())
    monkeypatch.setattr(
        generator_receiver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(generator_receiver, "latest_wav", lambda: "/tmp/current.wav")
    generator_receiver._run_render(
        "D1",
        {"root": 0, "scale": "free", "confidence": 0.0},
    )

    addresses = [address for address, _ in messages]
    assert addresses.index("/generator/path") < addresses.index("/generator/ready")
    assert messages[0] == ("/generator/busy", 1)
    assert messages[-1] == ("/generator/busy", 0)
