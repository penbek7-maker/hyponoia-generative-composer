from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import subprocess
import glob
import os
import sys
import threading

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 7401

MAX_IP = "127.0.0.1"
MAX_PORT = 7402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_SCRIPT = os.path.join(BASE_DIR, "generator_v3_memory_bloom_smooth.py")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

client = SimpleUDPClient(MAX_IP, MAX_PORT)
render_lock = threading.Lock()
state_lock = threading.Lock()

# Latest harmonic state received from Max/MSP.
# Defaults keep the system working even before the Max harmony module is added.
HARMONY_STATE = {
    "root": 0,           # 0=C, 1=C#, ... 11=B
    "scale": "free",     # e.g. major, minor, dorian, phrygian, free
    "confidence": 0.0
}


def latest_wav():
    wavs = glob.glob(os.path.join(OUTPUT_FOLDER, "*.wav"))
    if not wavs:
        return None
    return max(wavs, key=os.path.getmtime)


def root_handler(address, root):
    try:
        root_value = int(round(float(root))) % 12
        with state_lock:
            HARMONY_STATE["root"] = root_value
        print(f"→ Harmony root updated: {root_value}")
    except (TypeError, ValueError):
        print(f"Invalid harmony root received: {root}")


def scale_handler(address, scale):
    scale_value = str(scale).strip().lower()
    if not scale_value:
        scale_value = "free"

    with state_lock:
        HARMONY_STATE["scale"] = scale_value
    print(f"→ Harmony scale updated: {scale_value}")


def confidence_handler(address, confidence):
    try:
        value = float(confidence)
        value = max(0.0, min(1.0, value))
        with state_lock:
            HARMONY_STATE["confidence"] = value
        print(f"→ Harmony confidence updated: {value:.3f}")
    except (TypeError, ValueError):
        print(f"Invalid harmony confidence received: {confidence}")


def _run_render(dream_depth, harmony_state):
    if not render_lock.acquire(blocking=False):
        print("Generator is already rendering; request rejected without blocking OSC.")
        client.send_message("/generator/busy", 1)
        return
    try:
        level = dream_depth.replace("D", "").replace("d", "")
        root = harmony_state["root"]
        scale = harmony_state["scale"]
        confidence = harmony_state["confidence"]

        print(
            f"→ Generating {dream_depth} using dream level {level}, "
            f"root {root}, scale {scale}, confidence {confidence:.3f}..."
        )
        client.send_message("/generator/busy", 1)
        result = subprocess.run([
            sys.executable,
            GENERATOR_SCRIPT,
            level,
            str(root),
            scale,
            str(confidence)
        ], cwd=BASE_DIR, check=False)

        if result.returncode != 0:
            print(f"Generator failed with exit code {result.returncode}.")
            client.send_message("/generator/error", result.returncode)
            return

        wav_path = latest_wav()

        if wav_path:
            abs_path = os.path.abspath(wav_path)
            print(f"→ Ready: {abs_path}")
            # Max must receive and preload the concrete path before it receives
            # the ready trigger. This ordering prevents a local UDP race.
            client.send_message("/generator/path", abs_path)
            client.send_message("/generator/ready", 1)
        else:
            print("No WAV found after generation.")
            client.send_message("/generator/error", -1)
    finally:
        client.send_message("/generator/busy", 0)
        render_lock.release()


def render_handler(address, dream_depth):
    print(f"Received from Max: {address} {dream_depth}")
    dream_depth = str(dream_depth)
    with state_lock:
        harmony_state = dict(HARMONY_STATE)
    threading.Thread(target=_run_render, args=(dream_depth, harmony_state), daemon=True).start()


def build_server():
    dispatcher = Dispatcher()
    dispatcher.map("/harmony/root", root_handler)
    dispatcher.map("/harmony/scale", scale_handler)
    dispatcher.map("/harmony/confidence", confidence_handler)
    dispatcher.map("/generator/render", render_handler)
    return ThreadingOSCUDPServer((LISTEN_IP, LISTEN_PORT), dispatcher)


def main():
    server = build_server()
    print(f"Listening for Max on {LISTEN_IP}:{LISTEN_PORT}")
    print("Waiting for harmony messages:")
    print("  /harmony/root <0-11>")
    print("  /harmony/scale <name>")
    print("  /harmony/confidence <0.0-1.0>")
    print("Waiting for /generator/render D1, D3, or D5...")
    server.serve_forever()


if __name__ == "__main__":
    main()
