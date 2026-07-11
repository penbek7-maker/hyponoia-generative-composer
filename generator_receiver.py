from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import subprocess
import glob
import os

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 7401

MAX_IP = "127.0.0.1"
MAX_PORT = 7402

GENERATOR_SCRIPT = "generator_v3_memory_bloom_smooth.py"
OUTPUT_FOLDER = "output"

client = SimpleUDPClient(MAX_IP, MAX_PORT)

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
        HARMONY_STATE["root"] = root_value
        print(f"→ Harmony root updated: {root_value}")
    except (TypeError, ValueError):
        print(f"Invalid harmony root received: {root}")


def scale_handler(address, scale):
    scale_value = str(scale).strip().lower()
    if not scale_value:
        scale_value = "free"

    HARMONY_STATE["scale"] = scale_value
    print(f"→ Harmony scale updated: {scale_value}")


def confidence_handler(address, confidence):
    try:
        value = float(confidence)
        value = max(0.0, min(1.0, value))
        HARMONY_STATE["confidence"] = value
        print(f"→ Harmony confidence updated: {value:.3f}")
    except (TypeError, ValueError):
        print(f"Invalid harmony confidence received: {confidence}")


def render_handler(address, dream_depth):
    print(f"Received from Max: {address} {dream_depth}")

    dream_depth = str(dream_depth)
    level = dream_depth.replace("D", "").replace("d", "")

    root = HARMONY_STATE["root"]
    scale = HARMONY_STATE["scale"]
    confidence = HARMONY_STATE["confidence"]

    print(
        f"→ Generating {dream_depth} using dream level {level}, "
        f"root {root}, scale {scale}, confidence {confidence:.3f}..."
    )

    result = subprocess.run([
        "python3",
        GENERATOR_SCRIPT,
        level,
        str(root),
        scale,
        str(confidence)
    ])

    if result.returncode != 0:
        print(f"Generator failed with exit code {result.returncode}.")
        client.send_message("/generator/error", result.returncode)
        return

    wav_path = latest_wav()

    if wav_path:
        abs_path = os.path.abspath(wav_path)
        print(f"→ Ready: {abs_path}")

        client.send_message("/generator/ready", 1)
        client.send_message("/generator/path", abs_path)
    else:
        print("No WAV found after generation.")
        client.send_message("/generator/error", -1)


dispatcher = Dispatcher()
dispatcher.map("/harmony/root", root_handler)
dispatcher.map("/harmony/scale", scale_handler)
dispatcher.map("/harmony/confidence", confidence_handler)
dispatcher.map("/generator/render", render_handler)

server = BlockingOSCUDPServer((LISTEN_IP, LISTEN_PORT), dispatcher)

print(f"Listening for Max on {LISTEN_IP}:{LISTEN_PORT}")
print("Waiting for harmony messages:")
print("  /harmony/root <0-11>")
print("  /harmony/scale <name>")
print("  /harmony/confidence <0.0-1.0>")
print("Waiting for /generator/render D1, D3, or D5...")

server.serve_forever()
