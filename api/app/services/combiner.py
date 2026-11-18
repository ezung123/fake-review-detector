from typing import Literal

VerdictType = Literal["genuine_human", "deceptive_human", "ai_genuine", "ai_deceptive"]

VERDICT_DISPLAY_TEXT: dict[VerdictType, str] = {
    "genuine_human": "Likely genuine, human-written",
    "deceptive_human": "Likely deceptive, human-written",
    "ai_genuine": "Likely AI-written, no sign of deception",
    "ai_deceptive": "Likely AI-generated and deceptive",
}

def combine_predictions(
    authorship_label: Literal["human", "ai"],
    deception_label: Literal["genuine", "deceptive"],
) -> VerdictType:
    """
    Combine independent authorship and deception predictions into one of four distinct verdicts.
    Crucially maintains independent dimensions and does not collapse them into a single score.
    """
    if authorship_label == "human":
        return "genuine_human" if deception_label == "genuine" else "deceptive_human"
    else:  # authorship_label == "ai"
        return "ai_genuine" if deception_label == "genuine" else "ai_deceptive"

