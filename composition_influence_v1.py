"""Explain which whole-composition domains are affected by feedback controls."""

from __future__ import annotations

from typing import Any


DOMAIN_CONTROLS = {
    "material_selection": {
        "musicality_weight",
        "coherence_weight",
        "exploration_weight",
        "repetition_control",
        "synthetic_material_weight",
        "long_layer_diversity_weight",
    },
    "synth_and_arpeggios": {
        "synthetic_material_weight",
        "arpeggio_weight",
        "gesture_weight",
    },
    "energy_and_density": {
        "activity_weight",
        "richness_weight",
        "gesture_weight",
        "ambient_weight",
    },
    "layers_and_mix": {
        "layer_clarity_weight",
        "low_frequency_control",
        "richness_weight",
        "ambient_weight",
        "noise_penalty",
        "impact_penalty",
    },
    "material_development": {
        "material_development_weight",
        "repetition_control",
        "coherence_weight",
        "musicality_weight",
    },
    "transitions": {
        "transition_smoothness_weight",
        "coherence_weight",
    },
    "overall_form": {
        "bloom_weight",
        "material_development_weight",
        "activity_weight",
        "richness_weight",
        "coherence_weight",
    },
}

DOMAIN_LABELS_EL = {
    "material_selection": "επιλογή και σχέση υλικών",
    "synth_and_arpeggios": "synth και arpeggios",
    "energy_and_density": "ενέργεια και πυκνότητα",
    "layers_and_mix": "ηχητικά επίπεδα και μίξη",
    "material_development": "ανάπτυξη του υλικού",
    "transitions": "μεταβάσεις και αποχωρήσεις",
    "overall_form": "συνολική μορφή",
}

DOMAIN_LABELS_EN = {
    "material_selection": "material selection and relationships",
    "synth_and_arpeggios": "synth and arpeggios",
    "energy_and_density": "energy and density",
    "layers_and_mix": "sonic layers and mix",
    "material_development": "material development",
    "transitions": "transitions and sound departures",
    "overall_form": "overall form",
}


def describe_composition_influence(control_deltas: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable domain map for the requested non-zero controls."""
    active_controls = {
        str(control)
        for control, value in dict(control_deltas).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) != 0.0
    }
    domains = []
    for domain, domain_controls in DOMAIN_CONTROLS.items():
        matched = sorted(active_controls & domain_controls)
        if not matched:
            continue
        domains.append({
            "domain": domain,
            "label_el": DOMAIN_LABELS_EL[domain],
            "label_en": DOMAIN_LABELS_EN[domain],
            "controls": matched,
        })
    return {
        "schema_version": "composition_influence_v1",
        "active_controls": sorted(active_controls),
        "domains": domains,
        "domain_ids": [item["domain"] for item in domains],
        "domain_count": len(domains),
        "total_domain_count": len(DOMAIN_CONTROLS),
    }
