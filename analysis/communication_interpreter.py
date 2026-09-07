"""
AVIARA Communication Interpreter

Purpose:
    Convert bird species + acoustic evidence into a cautious,
    human-readable behavioral interpretation.

Important:
    This is NOT literal bird-to-human translation.
    The communication context is an acoustic/behavioral estimate.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================
# DISPLAY LABELS
# ============================================================

COMMUNICATION_DISPLAY = {
    "song": "Song",
    "contact_social": "Contact / Social Call",
    "alarm_warning": "Alarm / Warning",
    "begging_distress": "Begging / Distress Call",
    "uncertain": "Uncertain",
}


# ============================================================
# SPECIES-SPECIFIC HUMAN CONTEXT
# ============================================================

SPECIES_CONTEXT = {

    "Eudynamys scolopaceus": {
        "song":
            "advertising my presence, attracting a mate, "
            "or maintaining a territory",

        "contact_social":
            "keeping in contact with nearby birds "
            "or signaling where I am",

        "alarm_warning":
            "alerting nearby birds to a possible disturbance "
            "or threat",

        "begging_distress":
            "seeking attention, food, or support",
    },

    "Acridotheres tristis": {
        "song":
            "displaying my vocal repertoire, advertising my presence, "
            "or maintaining social or territorial space",

        "contact_social":
            "keeping contact with nearby mynas "
            "and coordinating social activity",

        "alarm_warning":
            "warning nearby birds that something unusual "
            "or potentially dangerous is nearby",

        "begging_distress":
            "asking for food, attention, or assistance",
    },

    "Pycnonotus cafer": {
        "song":
            "advertising my presence, defending territory, "
            "or attracting a mate",

        "contact_social":
            "maintaining social contact with nearby bulbuls",

        "alarm_warning":
            "warning nearby birds about a possible predator "
            "or disturbance",

        "begging_distress":
            "seeking food, attention, or help",
    },

    "Orthotomus sutorius": {
        "song":
            "advertising my presence, defending territory, "
            "or attracting a mate",

        "contact_social":
            "keeping track of nearby individuals during normal activity",

        "alarm_warning":
            "alerting nearby birds to a disturbance or possible threat",

        "begging_distress":
            "seeking food, attention, or assistance",
    },

    "Psittacula krameri": {
        "song":
            "advertising my presence or participating in social display",

        "contact_social":
            "maintaining contact with my flock "
            "and coordinating movement",

        "alarm_warning":
            "warning flock members about a possible threat",

        "begging_distress":
            "requesting food, attention, or support",
    },

    "Pavo cristatus": {
        "song":
            "signaling my presence during display "
            "or territory-related behavior",

        "contact_social":
            "maintaining contact with nearby peafowl",

        "alarm_warning":
            "alerting nearby birds to a possible predator "
            "or disturbance",

        "begging_distress":
            "seeking attention, food, or help",
    },

    "Centropus sinensis": {
        "song":
            "advertising my presence, territory, "
            "or reproductive status",

        "contact_social":
            "maintaining contact with nearby individuals",

        "alarm_warning":
            "warning nearby birds about a possible disturbance "
            "or threat",

        "begging_distress":
            "seeking attention or responding to stress",
    },

    "Halcyon smyrnensis": {
        "song":
            "signaling my presence or occupying my local territory",

        "contact_social":
            "maintaining contact with nearby individuals",

        "alarm_warning":
            "alerting others to a possible threat or disturbance",

        "begging_distress":
            "requesting food, attention, or assistance",
    },

    "Milvus migrans": {
        "song":
            "advertising my presence or maintaining social "
            "and territorial spacing",

        "contact_social":
            "coordinating with nearby kites "
            "and maintaining social contact",

        "alarm_warning":
            "warning nearby birds about a possible threat "
            "or disturbance",

        "begging_distress":
            "seeking attention or support",
    },

    "Corvus splendens": {
        "song":
            "broadcasting my presence and maintaining social spacing",

        "contact_social":
            "keeping the group informed about my location and activity",

        "alarm_warning":
            "warning other crows that something potentially dangerous "
            "is nearby",

        "begging_distress":
            "requesting food, attention, or help",
    },
}


GENERIC_CONTEXT = {
    "song":
        "advertising my presence, territory, "
        "or social/reproductive status",

    "contact_social":
        "maintaining contact or coordinating with nearby birds",

    "alarm_warning":
        "warning nearby birds about a possible disturbance or threat",

    "begging_distress":
        "requesting food, attention, or assistance",
}



# ============================================================
# RICHER BEHAVIORAL INTENT LAYER
# ============================================================

# This layer converts the broad communication context into a more
# human-readable probable behavioral intent. It does not claim that
# the model has literally decoded a bird's thoughts.

BEHAVIOR_INTENT_DISPLAY = {
    "food_seeking": "Food Seeking",
    "attention_seeking": "Attention Seeking",
    "threat_response": "Threat Response",
    "territorial_defense": "Territorial Defense",
    "mate_attraction": "Mate Attraction",
    "social_contact": "Social Contact",
    "group_coordination": "Group Coordination",
    "distress_help": "Distress / Help Seeking",
    "presence_display": "Presence / Social Display",
    "uncertain": "Uncertain Intent",
}

BEHAVIOR_MESSAGES = {
    "food_seeking": (
        "I may be hungry and looking for food.",
        "The vocalization is being interpreted as a possible food-seeking signal, especially when the source context is begging or feeding related.",
    ),
    "attention_seeking": (
        "Please pay attention to me — I may need food, care, or help.",
        "A begging/distress context can reflect attention or care seeking as well as food seeking.",
    ),
    "threat_response": (
        "I may feel threatened. Stay alert — there could be danger nearby.",
        "An alarm/warning context is consistent with a response to a possible predator, disturbance, or other threat.",
    ),
    "territorial_defense": (
        "This is my territory. Keep your distance — I may be defending my space.",
        "A song or warning in a territorial context may function as a spacing or territorial-defense signal.",
    ),
    "mate_attraction": (
        "I am looking for a mate and announcing that I am here.",
        "A song/display context may be associated with courtship or mate attraction.",
    ),
    "social_contact": (
        "Where are you? I am keeping in contact with the other birds around me.",
        "A contact/social call can help maintain proximity and social awareness among nearby birds.",
    ),
    "group_coordination": (
        "Stay connected with me — I am signaling where I am or coordinating with the group.",
        "Group-oriented contact calls can help birds maintain contact and coordinate movement.",
    ),
    "distress_help": (
        "I may be stressed or in trouble. I may need help or attention.",
        "Begging/distress context can indicate stress, discomfort, separation, or a request for assistance.",
    ),
    "presence_display": (
        "I am here — I am announcing my presence to nearby birds.",
        "A song/display context commonly fits a presence or social-display interpretation.",
    ),
    "uncertain": (
        "I am communicating, but the exact reason is unclear from the available evidence.",
        "The recording does not provide enough reliable context for a more specific behavioral interpretation.",
    ),
}

# Species-specific wording for the same broad intent. The values are
# deliberately subtle: species change the natural-language framing,
# while the underlying intent remains conservative.
SPECIES_INTENT_TONES = {
    "Eudynamys scolopaceus": {
        "mate_attraction": "I am calling to announce myself and may be looking for a mate.",
        "territorial_defense": "This is my space. Keep your distance while I defend my territory.",
        "social_contact": "Where are you? I am keeping contact with nearby birds.",
    },
    "Acridotheres tristis": {
        "social_contact": "Where are you? I am keeping in touch with nearby mynas.",
        "attention_seeking": "Pay attention to me — I may be asking for food, care, or help.",
        "territorial_defense": "Keep your distance. I am defending my space.",
    },
    "Pycnonotus cafer": {
        "mate_attraction": "I am announcing myself and may be trying to attract a mate.",
        "territorial_defense": "This is my space. I am warning others to keep away.",
        "social_contact": "Stay in touch with me — I am keeping contact with nearby bulbuls.",
    },
    "Orthotomus sutorius": {
        "mate_attraction": "I am calling to attract a mate and make my presence known.",
        "territorial_defense": "Please stay away from my nesting or territorial space.",
        "social_contact": "Where are the others? I am keeping track of nearby birds.",
    },
    "Psittacula krameri": {
        "social_contact": "Where are you? I am staying connected with my flock.",
        "group_coordination": "Stay with the flock — I am signaling my position.",
        "threat_response": "Watch out. I may have detected a threat and I am alerting the flock.",
    },
    "Pavo cristatus": {
        "mate_attraction": "I am displaying and calling because I may be looking for a mate.",
        "social_contact": "Where are you? I am keeping contact with other peafowl.",
        "threat_response": "Stay alert. I may have detected danger nearby.",
    },
    "Centropus sinensis": {
        "territorial_defense": "This is my space. I am announcing my presence and defending it.",
        "mate_attraction": "I am announcing myself and may be looking for a mate.",
        "social_contact": "I am keeping contact with nearby birds.",
    },
    "Halcyon smyrnensis": {
        "territorial_defense": "Keep your distance. I am defending my local space.",
        "threat_response": "Something may be wrong. I am warning nearby birds to stay alert.",
        "social_contact": "Where are you? I am keeping contact with nearby birds.",
    },
    "Milvus migrans": {
        "social_contact": "Stay connected. I am signaling my location to nearby kites.",
        "group_coordination": "Keep track of me — I am coordinating with nearby kites.",
        "threat_response": "Stay alert. I may be warning nearby birds about a possible threat.",
    },
    "Corvus splendens": {
        "social_contact": "Where are you? I am keeping the group informed about where I am.",
        "group_coordination": "Stay together. I am signaling my location or coordinating with the group.",
        "threat_response": "Watch out. I may have detected something dangerous nearby.",
    },
}


def infer_behavioral_intent(
    species: str,
    communication_type: str,
    annotation_text: str = "",
) -> str:
    """Map broad vocalization context to a probable behavioral intent."""
    text = str(annotation_text or "").lower()

    if communication_type == "begging_distress":
        if any(k in text for k in ("food", "feeding", "hungry", "begging", "chick", "juvenile", "young")):
            return "food_seeking"
        if any(k in text for k in ("distress", "injur", "trapped", "attack", "stress", "separation")):
            return "distress_help"
        return "attention_seeking"

    if communication_type == "alarm_warning":
        if any(k in text for k in ("territor", "intrud", "defend", "chase", "aggress")):
            return "territorial_defense"
        return "threat_response"

    if communication_type == "contact_social":
        if any(k in text for k in ("flock", "group", "coord", "movement", "flight")):
            return "group_coordination"
        return "social_contact"

    if communication_type == "song":
        has_mate = any(k in text for k in ("mate", "courtship", "breeding", "display", "reproduct"))
        has_territory = any(k in text for k in ("territor", "defend", "claim", "advertis"))
        if has_mate and has_territory:
            # Use the stronger, user-facing reproductive interpretation.
            return "mate_attraction"
        if has_mate:
            return "mate_attraction"
        if has_territory:
            return "territorial_defense"
        return "presence_display"

    return "uncertain"


def _intent_message(
    species: str,
    intent: str,
) -> Tuple[str, str]:
    """Return species-aware message and explanation for an intent."""
    base_message, explanation = BEHAVIOR_MESSAGES.get(
        intent,
        BEHAVIOR_MESSAGES["uncertain"],
    )
    species_variant = SPECIES_INTENT_TONES.get(species, {}).get(intent)
    return species_variant or base_message, explanation


# ============================================================
# RESULT
# ============================================================

@dataclass
class InterpretationResult:

    communication_display: str

    evidence_level: str

    interpretation: str

    message: str

    message_title: str

    acoustic_summary: str

    caution: str

    estimated: bool = False

    intent_label: str = "Uncertain Intent"

    intent_confidence: float = 0.0

    evidence_source: str = "Acoustic interpretation"

    annotation_text: str = ""


# ============================================================
# SAFE FEATURE READING
# ============================================================

def _safe_float(
    features: Dict[str, float],
    key: str,
    default: float = 0.0,
) -> float:

    try:

        value = float(
            features.get(
                key,
                default,
            )
        )

        if value != value:
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# NORMALIZE A FEATURE
# ============================================================

def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:

    if value < low:
        return low

    if value > high:
        return high

    return value


# ============================================================
# BALANCED COMMUNICATION ESTIMATOR
# ============================================================

def infer_vocalization_type(
    acoustic_features: Dict[str, float],
) -> Tuple[str, str, Dict[str, float]]:

    """
    Estimate communication context from acoustic features.

    Four candidates are scored independently:

        Song
        Contact / Social Call
        Alarm / Warning
        Begging / Distress

    The estimator deliberately allows "uncertain" when the
    evidence is weak or multiple classes are too close.

    This is a heuristic, not a trained semantic classifier.
    """

    rms = _safe_float(
        acoustic_features,
        "rms_energy_mean",
    )

    zcr = _safe_float(
        acoustic_features,
        "zero_crossing_rate",
    )

    centroid = _safe_float(
        acoustic_features,
        "spectral_centroid_hz",
    )

    bandwidth = _safe_float(
        acoustic_features,
        "spectral_bandwidth_hz",
    )

    rolloff = _safe_float(
        acoustic_features,
        "spectral_rolloff_hz",
    )

    flatness = _safe_float(
        acoustic_features,
        "spectral_flatness",
    )

    # --------------------------------------------------------
    # Acoustic proxies
    # --------------------------------------------------------

    tonal = 1.0 - _clamp(
        flatness,
        0.0,
        1.0,
    )

    noisy = _clamp(
        flatness,
        0.0,
        1.0,
    )

    low_zcr = 1.0 - _clamp(
        zcr / 0.30,
        0.0,
        1.0,
    )

    high_zcr = _clamp(
        zcr / 0.30,
        0.0,
        1.0,
    )

    mid_centroid = 1.0 - min(
        abs(centroid - 3500.0) / 5000.0,
        1.0,
    )

    high_centroid = _clamp(
        (centroid - 4500.0) / 6000.0,
        0.0,
        1.0,
    )

    broad_spectrum = _clamp(
        bandwidth / 7000.0,
        0.0,
        1.0,
    )

    high_rolloff = _clamp(
        (rolloff - 5000.0) / 9000.0,
        0.0,
        1.0,
    )

    strong_energy = _clamp(
        rms / 0.15,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Initialize equal priors
    # --------------------------------------------------------

    scores = {
        "song": 1.0,
        "contact_social": 1.0,
        "alarm_warning": 1.0,
        "begging_distress": 1.0,
    }

    # --------------------------------------------------------
    # SONG
    #
    # Structured / tonal / moderately stable signals
    # --------------------------------------------------------

    scores["song"] += (
        2.2 * tonal
        + 1.0 * low_zcr
        + 0.8 * mid_centroid
        + 0.3 * (1.0 - broad_spectrum)
    )

    # --------------------------------------------------------
    # CONTACT / SOCIAL
    #
    # Shorter/simple-looking social signals often have
    # intermediate structure rather than extremely tonal
    # or extremely noisy characteristics.
    # --------------------------------------------------------

    contact_balance = (
        1.0
        - abs(tonal - 0.55)
    )

    contact_energy = (
        1.0
        - abs(strong_energy - 0.45)
    )

    scores["contact_social"] += (
        1.6 * contact_balance
        + 1.0 * mid_centroid
        + 0.7 * contact_energy
        + 0.5 * (1.0 - high_zcr)
    )

    # --------------------------------------------------------
    # ALARM
    #
    # Noisy / broadband / high-frequency / higher-ZCR patterns
    # --------------------------------------------------------

    scores["alarm_warning"] += (
        2.0 * noisy
        + 1.4 * high_zcr
        + 1.2 * broad_spectrum
        + 0.9 * high_centroid
        + 0.7 * high_rolloff
        + 0.4 * strong_energy
    )

    # --------------------------------------------------------
    # BEGGING / DISTRESS
    #
    # High-energy and relatively noisy signals receive
    # additional support, but not enough to dominate.
    # --------------------------------------------------------

    distress_energy = _clamp(
        (rms - 0.03) / 0.12,
        0.0,
        1.0,
    )

    distress_noise = (
        0.7 * noisy
        + 0.5 * broad_spectrum
    )

    distress_pitch = 1.0 - _clamp(
        abs(centroid - 5000.0) / 7000.0,
        0.0,
        1.0,
    )

    scores["begging_distress"] += (
        1.4 * distress_energy
        + 1.1 * distress_noise
        + 0.7 * distress_pitch
        + 0.3 * strong_energy
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_type = ordered[0][0]

    best_score = ordered[0][1]

    second_score = ordered[1][1]

    total_score = sum(
        scores.values()
    )

    relative_confidence = (
        best_score / total_score
        if total_score > 0
        else 0.0
    )

    margin = (
        best_score - second_score
    )

    # --------------------------------------------------------
    # Don't force a class when evidence is weak
    # --------------------------------------------------------

    if (
        margin < 0.35
        or relative_confidence < 0.31
    ):

        return (
            "uncertain",
            "low",
            scores,
        )

    if (
        margin < 0.80
        or relative_confidence < 0.36
    ):

        return (
            best_type,
            "low",
            scores,
        )

    if (
        margin >= 1.40
        and relative_confidence >= 0.45
    ):

        return (
            best_type,
            "moderate",
            scores,
        )

    return (
        best_type,
        "low",
        scores,
    )


# ============================================================
# HUMAN READABLE MESSAGE
# ============================================================

def _build_human_message(
    species: str,
    communication_type: str,
    annotation_text: str = "",
) -> Tuple[str, str, str]:

    intent = infer_behavioral_intent(
        species,
        communication_type,
        annotation_text,
    )

    message, explanation = _intent_message(
        species,
        intent,
    )

    return message, explanation, intent


# ============================================================
# ACOUSTIC SUMMARY
# ============================================================

def _build_acoustic_summary(
    acoustic_features: Dict[str, float],
) -> str:

    rms = _safe_float(
        acoustic_features,
        "rms_energy_mean",
    )

    zcr = _safe_float(
        acoustic_features,
        "zero_crossing_rate",
    )

    centroid = _safe_float(
        acoustic_features,
        "spectral_centroid_hz",
    )

    bandwidth = _safe_float(
        acoustic_features,
        "spectral_bandwidth_hz",
    )

    flatness = _safe_float(
        acoustic_features,
        "spectral_flatness",
    )

    evidence = []

    if flatness < 0.18:

        evidence.append(
            "strong tonal structure"
        )

    elif flatness < 0.30:

        evidence.append(
            "moderately tonal structure"
        )

    elif flatness > 0.45:

        evidence.append(
            "relatively noise-like structure"
        )

    else:

        evidence.append(
            "mixed tonal and noisy structure"
        )

    if zcr < 0.10:

        evidence.append(
            "low zero-crossing activity"
        )

    elif zcr > 0.22:

        evidence.append(
            "high zero-crossing activity"
        )

    else:

        evidence.append(
            "moderate zero-crossing activity"
        )

    if centroid > 0:

        evidence.append(
            f"spectral centroid around {centroid:,.0f} Hz"
        )

    if bandwidth > 0:

        evidence.append(
            f"spectral bandwidth around {bandwidth:,.0f} Hz"
        )

    if rms > 0:

        evidence.append(
            f"mean RMS energy of {rms:.4f}"
        )

    return (
        "The interpretation is based on "
        + ", ".join(
            evidence[:5]
        )
        + "."
    )


# ============================================================
# MAIN INTERPRETER
# ============================================================

def interpret_birdsong(
    species: str,
    communication_type: Optional[str] = None,
    confidence: float = 0.0,
    acoustic_features: Optional[Dict[str, float]] = None,
    source: str = "unknown",
    annotation_text: str = "",
) -> InterpretationResult:

    acoustic_features = (
        acoustic_features
        or {}
    )

    estimated = False

    # --------------------------------------------------------
    # Automatic estimation
    # --------------------------------------------------------

    if (
        communication_type is None
        or communication_type == "uncertain"
    ):

        (
            communication_type,
            estimated_level,
            scores,
        ) = infer_vocalization_type(
            acoustic_features
        )

        estimated = True

        evidence_level = (
            estimated_level
        )

    else:

        # Manual / externally known context
        if confidence >= 0.80:

            evidence_level = "high"

        elif confidence >= 0.60:

            evidence_level = "moderate"

        else:

            evidence_level = "low"

    # --------------------------------------------------------
    # Display name
    # --------------------------------------------------------

    display_name = COMMUNICATION_DISPLAY.get(
        communication_type,
        "Uncertain",
    )

    # --------------------------------------------------------
    # Message
    # --------------------------------------------------------

    message, interpretation, intent = (
        _build_human_message(
            species,
            communication_type,
            annotation_text,
        )
    )

    # --------------------------------------------------------
    # Acoustic summary
    # --------------------------------------------------------

    acoustic_summary = (
        _build_acoustic_summary(
            acoustic_features
        )
    )

    # --------------------------------------------------------
    # Caution
    # --------------------------------------------------------

    if communication_type == "uncertain":

        caution = (
            "The available acoustic evidence does not strongly "
            "separate the main vocalization categories. The "
            "human-readable statement is therefore intentionally "
            "non-specific and should not be treated as a literal "
            "translation of the bird's intent."
        )

    elif estimated:

        caution = (
            "The vocalization context was estimated from acoustic "
            "features using a heuristic scoring model. The message "
            "describes a plausible behavioral interpretation; it "
            "does not directly measure or decode the bird's exact intent."
        )

    else:

        caution = (
            "The vocalization category was supplied as known or "
            "observed context. The message describes a plausible "
            "behavioral meaning and does not establish the bird's "
            "exact intent."
        )

    if communication_type == "uncertain":
        intent_confidence = 0.0
    elif estimated:
        intent_confidence = 0.45 if intent != "uncertain" else 0.0
    else:
        intent_confidence = 0.82 if intent != "uncertain" else 0.35

    intent_label = BEHAVIOR_INTENT_DISPLAY.get(
        intent,
        BEHAVIOR_INTENT_DISPLAY["uncertain"],
    )

    return InterpretationResult(

        communication_display=display_name,

        evidence_level=evidence_level.title(),

        interpretation=interpretation,

        message=message,

        message_title=(
            "What the bird may be communicating"
        ),

        acoustic_summary=acoustic_summary,

        caution=caution,

        estimated=estimated,

        intent_label=intent_label,

        intent_confidence=intent_confidence,

        evidence_source=(
            source if source and source != "unknown"
            else ("Acoustic fallback" if estimated else "Known vocalization context")
        ),

        annotation_text=annotation_text or "",
    )


# ============================================================
# DICTIONARY COMPATIBILITY
# ============================================================

def interpret_to_dict(
    *args,
    **kwargs,
) -> Dict[str, object]:

    result = interpret_birdsong(
        *args,
        **kwargs,
    )

    return {

        "communication_display":
            result.communication_display,

        "evidence_level":
            result.evidence_level,

        "interpretation":
            result.interpretation,

        "message":
            result.message,

        "message_title":
            result.message_title,

        "acoustic_summary":
            result.acoustic_summary,

        "caution":
            result.caution,

        "estimated":
            result.estimated,
    }