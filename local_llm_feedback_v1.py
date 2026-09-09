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
from local_language_model_v1 import DEFAULT_MODEL


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MIN_CONFIDENCE = 0.55

INTENT_CONTROL_DELTAS: dict[str, dict[str, float]] = {
    "increase_musicality": {"musicality_weight": 0.06},
    "increase_coherence": {"coherence_weight": 0.07},
    "increase_rhythmicity": {"activity_weight": 0.05},
    "increase_bloom": {"bloom_weight": 0.07},
    "increase_synthetic_material": {"synthetic_material_weight": 0.08},
    "increase_instrument_material": {"instrument_material_weight": 0.08},
    "bring_musical_material_forward": {
        "foreground_presence_weight": 0.08,
        "layer_clarity_weight": 0.04,
    },
    "increase_arpeggios": {"arpeggio_weight": 0.10},
    "decrease_arpeggios": {"arpeggio_weight": -0.20},
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
    "increase_looping": {"repetition_control": -0.08},
    "decrease_repetition": {
        "repetition_control": 0.10,
        "exploration_weight": 0.06,
    },
    "increase_smoothness": {"transition_smoothness_weight": 0.08},
    "increase_richness": {"richness_weight": 0.07},
    "increase_activity": {"activity_weight": 0.08},
    "increase_material_development": {"material_development_weight": 0.08},
    "strengthen_overall_form": {
        "bloom_weight": 0.05,
        "material_development_weight": 0.05,
        "coherence_weight": 0.03,
    },
    "reduce_low_frequency_masking": {"low_frequency_control": 0.08},
    "increase_structured_granulation": {
        "structured_granulation_weight": 0.10,
        "material_development_weight": 0.03,
    },
}

ALLOWED_INTENTS = tuple(INTENT_CONTROL_DELTAS)

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary_el", "intents", "confidence", "ambiguities"],
    "properties": {
        "summary_el": {"type": "string", "maxLength": 180},
        "intents": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(ALLOWED_INTENTS)},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "ambiguities": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 120},
        },
    },
}

SYSTEM_PROMPT = """You are the local language interpreter for Hyponoia, an
experimental music composition system. Read informal Greek or English listener
feedback and select only requested or clearly implied future changes.

Intent meanings:
- increase_musicality: more musically convincing phrasing/relationships
- increase_coherence: stronger meaningful connections between materials
- increase_rhythmicity: more pulse or rhythmic motion
- increase_bloom: a stronger large-scale rise, expansion or climax
- increase_synthetic_material: more synthesizer/electronic material
- increase_instrument_material: more confirmed instrument-hybrid source material
- bring_musical_material_forward: clearer foreground presence for musical material
- increase_arpeggios: more arpeggiated figures
- decrease_arpeggios: fewer or no arpeggiated figures
- increase_layer_clarity: clearer mix, separation, less mud/buried layers
- diversify_long_layers: avoid reusing the same long drones/layers
- increase_library_exploration: use more/different library sounds
- increase_palette_variety: a more varied/plural sound palette
- increase_looping: more deliberate musical loops or motif recurrence
- decrease_repetition: fewer repeated sounds or ideas
- increase_smoothness: smoother transitions, entrances or exits
- increase_richness: less empty/thin, richer texture or more layers
- increase_activity: more energy/activity/speed
- increase_material_development: evolve ideas rather than merely place/repeat them
- strengthen_overall_form: clearer whole-piece direction, arc or arrival
- reduce_low_frequency_masking: less bass masking or excessive low frequencies
- increase_structured_granulation: more organised, rhythmic granular movement

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
- Understand paraphrases and metaphors from the whole sentence. Do not require
  the listener to use an intent's technical words.
- Sparse, naked, thin, "γυμνό", or unable to bloom asks for increase_richness
  and/or increase_material_development; it never means decrease_repetition.
- Ideas returning, being remembered, "να ξαναγυρίζουν ιδέες", or recurring
  motifs asks for increase_looping; it never means decrease_repetition.
- Disconnected sounds ask for increase_coherence. Startling or abrupt exits ask
  for increase_smoothness. Do not infer repetition from either complaint.

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
- "Περισσότερα όργανα και τα μουσικά στοιχεία πιο μπροστά" ->
  increase_instrument_material, bring_musical_material_forward.
- "Δεν χρειάζεται arpeggio εδώ" -> decrease_arpeggios.
- "Θέλω περισσότερες συνδέσεις μεταξύ των ήχων" -> increase_coherence.
- "Θέλω λίγες λούπες και πιο πολλά arpeggios" -> increase_looping,
  increase_arpeggios. Here "λίγες λούπες" asks to add some loops; it does NOT
  mean fewer loops.
- "Είναι κάπως γυμνό και δεν ανθίζει" -> increase_richness,
  increase_material_development.
- "Θέλω να ξαναγυρίζουν κάποιες ιδέες ώστε να νιώθω ότι θυμάται" ->
  increase_looping.
- "The sounds feel disconnected and the exits make me jump" ->
  increase_coherence, increase_smoothness.
- "Οι ήχοι μπαίνουν και βγαίνουν σαν διακόπτες" -> increase_smoothness.
- "Δεν οδηγεί κάπου σαν συνολική σύνθεση" -> strengthen_overall_form.
- "Θέλω περισσότερο οργανωμένο granulation" -> increase_structured_granulation.

Allowed intent identifiers:
{intents}

Write summary_el as natural listener-facing language, never as intent IDs, in
at most 18 words. Include an ambiguity only when it changes which intent should
be selected; do not merely repeat or define the comment. Write each ambiguity
in at most 12 words, in
the same language as the listener's comment (Greek for Greek input, English for
English input). Return one complete schema-valid JSON object only. Do not add
reasoning or Markdown.
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
    # Small local models sometimes repeat a valid label. Repetition is not an
    # unsafe interpretation, so normalise it instead of discarding the whole
    # contextual result and silently falling back to keyword rules.
    intents = list(dict.fromkeys(intents))
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
        "options": {"temperature": 0, "num_predict": 384, "repeat_penalty": 1.18},
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
