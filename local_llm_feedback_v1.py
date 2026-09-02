"""Local natural-language feedback interpretation through Ollama.

The language model may select only known musical intents.  Hyponoia, not the
model, owns the bounded control deltas and D-level routing.  This keeps free
language useful while preserving preview-before-apply and deterministic safety.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hyponoia_stability import feedback_target_scope


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"
MIN_CONFIDENCE = 0.55

INTENT_CONTROL_DELTAS: dict[str, dict[str, float]] = {
    "increase_musicality": {"musicality_weight": 0.06},
    "increase_rhythmicity": {"activity_weight": 0.05},
    "increase_bloom": {"bloom_weight": 0.07},
    "increase_synthetic_material": {"synthetic_material_weight": 0.08},
    "increase_arpeggios": {"arpeggio_weight": 0.10},
    "increase_layer_clarity": {"layer_clarity_weight": 0.08},
    "diversify_long_layers": {"long_layer_diversity_weight": 0.10},
    "increase_library_exploration": {
        "exploration_weight": 0.10,
        "repetition_control": 0.05,
    },
    "increase_palette_variety": {
        "exploration_weight": 0.06,
        "repetition_control": 0.03,
    },
    "decrease_repetition": {
        "repetition_control": 0.10,
        "exploration_weight": 0.06,
    },
    "increase_smoothness": {"transition_smoothness_weight": 0.08},
    "increase_richness": {"richness_weight": 0.07},
    "increase_activity": {"activity_weight": 0.08},
    "increase_material_development": {"material_development_weight": 0.08},
    "reduce_low_frequency_masking": {"low_frequency_control": 0.08},
}

ALLOWED_INTENTS = tuple(INTENT_CONTROL_DELTAS)

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary_el", "intents", "confidence", "ambiguities"],
    "properties": {
        "summary_el": {"type": "string"},
        "intents": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(ALLOWED_INTENTS)},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT = """You are the local language interpreter for Hyponoia, an
experimental music composition system. Read informal Greek or English listener
feedback and select only requested or clearly implied future changes.

Intent meanings:
- increase_musicality: more musically convincing phrasing/relationships
- increase_rhythmicity: more pulse or rhythmic motion
- increase_bloom: a stronger large-scale rise, expansion or climax
- increase_synthetic_material: more synthesizer/electronic material
- increase_arpeggios: more arpeggiated figures
- increase_layer_clarity: clearer mix, separation, less mud/buried layers
- diversify_long_layers: avoid reusing the same long drones/layers
- increase_library_exploration: use more/different library sounds
- increase_palette_variety: a more varied/plural sound palette
- decrease_repetition: fewer repeated sounds or ideas
- increase_smoothness: smoother transitions, entrances or exits
- increase_richness: less empty/thin, richer texture or more layers
- increase_activity: more energy/activity/speed
- increase_material_development: evolve ideas rather than merely place/repeat them
- reduce_low_frequency_masking: less bass masking or excessive low frequencies

Important distinctions:
- Praise or preservation such as 'keep the energy', 'the synth is good',
  'κρατάμε την ενέργεια' does NOT request an increase.
- A complaint such as 'λίγο άδειο' usually requests richer layers and material
  development; add energy only if activity/energy is actually implied.
- Abrupt exits/cuts request smoother transitions.
- Muddy/hidden/buried sounds request layer clarity; excessive bass also requests
  reduced low-frequency masking.
- Do not invent an intent. Put uncertainty into ambiguities and lower confidence.
- Return no intents for pure praise or comments unrelated to sound.

Examples:
- "Αυτό είναι βασικά λίγο άδειο" -> increase_richness,
  increase_material_development. It does NOT mean repetition.
- "Κρατάμε την ενέργεια και το synth, αλλά οι ήχοι φεύγουν απότομα" ->
  increase_smoothness only.
- "The middle feels buried and repetitive, but keep the pulse" ->
  increase_layer_clarity, decrease_repetition. Do not change pulse/activity.
- "Πολύ καλό, κράτα το έτσι" -> no intents.
- "Θέλω περισσότερο synth και arpeggios" -> increase_synthetic_material,
  increase_arpeggios.

Allowed intent identifiers:
{intents}

Write summary_el and ambiguities in concise Greek. Return only schema-valid JSON.
""".format(intents=", ".join(ALLOWED_INTENTS))


class LocalLLMUnavailable(RuntimeError):
    """Raised when the local Ollama service/model cannot provide a safe result."""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LocalLLMUnavailable(f"Local language model is unavailable: {exc}") from exc


def _validate_model_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise LocalLLMUnavailable("Local language model returned a non-object result")
    intents = raw.get("intents")
    if not isinstance(intents, list) or any(intent not in ALLOWED_INTENTS for intent in intents):
        raise LocalLLMUnavailable("Local language model returned an unsupported intent")
    if len(intents) != len(set(intents)):
        raise LocalLLMUnavailable("Local language model returned duplicate intents")
    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise LocalLLMUnavailable("Local language model returned invalid confidence")
    if not 0.0 <= float(confidence) <= 1.0:
        raise LocalLLMUnavailable("Local language model confidence is out of range")
    summary = raw.get("summary_el")
    ambiguities = raw.get("ambiguities")
    if not isinstance(summary, str) or not isinstance(ambiguities, list):
        raise LocalLLMUnavailable("Local language model returned incomplete explanation")
    if any(not isinstance(item, str) for item in ambiguities):
        raise LocalLLMUnavailable("Local language model returned invalid ambiguities")
    return {
        "summary_el": summary.strip(),
        "intents": intents,
        "confidence": float(confidence),
        "ambiguities": [item.strip() for item in ambiguities if item.strip()],
    }


def interpret_with_local_llm(
    comment: str,
    default_target_level: Any = None,
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float = 90.0,
    request_json: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Interpret free language while retaining deterministic routing and deltas."""
    text = str(comment).strip()
    target = feedback_target_scope(text, default_target_level)
    chosen_model = model or os.environ.get("HYPONOIA_LOCAL_LLM_MODEL", DEFAULT_MODEL)
    if not text:
        return {
            "schema_version": 1,
            "original_text": text,
            "status": "empty",
            "confidence": 0.0,
            "actions": [],
            "combined_control_deltas": {},
            "summary_el": "Δεν δόθηκε σχόλιο.",
            "ambiguities": [],
            "interpreter": "local_llm",
            "model": chosen_model,
            **target,
        }

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 256},
    }
    response = request_json(ollama_url or DEFAULT_OLLAMA_URL, payload, timeout)
    try:
        content = response["message"]["content"]
        result = _validate_model_result(json.loads(content))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LocalLLMUnavailable("Local language model returned invalid JSON") from exc

    confident = result["confidence"] >= MIN_CONFIDENCE
    intents = result["intents"] if confident else []
    actions = [
        {
            "intent": intent,
            "matched_pattern": None,
            "matched_language": "local_multilingual_model",
            "control_deltas": dict(INTENT_CONTROL_DELTAS[intent]),
            "implementation_note": "Intent proposed by the local language model; deltas remain Hyponoia-owned and bounded.",
        }
        for intent in intents
    ]
    combined: dict[str, float] = {}
    for action in actions:
        for control, delta in action["control_deltas"].items():
            combined[control] = combined.get(control, 0.0) + float(delta)

    return {
        "schema_version": 1,
        "original_text": text,
        "status": "interpreted" if actions else "unrecognised",
        "confidence": result["confidence"],
        "actions": actions,
        "combined_control_deltas": combined,
        "summary_el": result["summary_el"],
        "ambiguities": result["ambiguities"],
        "interpreter": "local_llm",
        "model": chosen_model,
        **target,
    }
