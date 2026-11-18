import hashlib
import re
from typing import Literal
from api.app.config import settings
from api.app.schemas import (
    AnalyzeResponse,
    AuthorshipResult,
    DeceptionResult,
    Explanations,
    ModelInfo,
    TokenExplanation,
)
from api.app.services.combiner import combine_predictions

# Heuristic vocabulary hints for realistic deterministic stub explanations
AI_INDICATORS = {
    "delve", "testament", "nestled", "tapestry", "crucial", "seamless", "seamlessly",
    "breathtaking", "unparalleled", "exceptional", "quintessential", "furthermore",
    "moreover", "invaluable", "exemplary", "boasts", "curated", "harmonious"
}

HUMAN_INDICATORS = {
    "smelled", "squeaked", "cracked", "stains", "refund", "receipt", "frontdesk",
    "dirty", "noisy", "elevators", "parking", "towel", "spotted", "broken",
    "waited", "yelled", "complained", "cardkey", "ac"
}

DECEPTIVE_INDICATORS = {
    "perfection", "flawless", "heavenly", "unbelievable", "magnificent", "must-visit",
    "miraculous", "spectacular", "godsend", "sublime", "unmatched", "world-class"
}

GENUINE_INDICATORS = {
    "however", "although", "pricey", "small", "adequate", "decent", "slow",
    "cramped", "walked", "walk", "minutes", "average", "street", "shower", "room"
}

def detect_runtime_mode() -> Literal["stub", "classical", "transformer"]:
    """
    Detect which models are available in the models directory.
    Defaults to 'stub' mode if trained weights are missing.
    """
    models_dir = settings.MODELS_DIR
    if not models_dir.exists():
        return "stub"

    has_transformer = (
        (models_dir / "authorship_distilbert").is_dir()
        and (models_dir / "deception_distilbert").is_dir()
    )
    if has_transformer:
        return "transformer"

    has_classical = (
        (models_dir / "authorship_classical.joblib").is_file()
        and (models_dir / "deception_classical.joblib").is_file()
    )
    if has_classical:
        return "classical"

    return "stub"

def _extract_stub_token_weights(
    tokens: list[str],
    task: Literal["authorship", "deception"],
    text_seed: int,
) -> list[TokenExplanation]:
    """
    Generate deterministic token attribution weights for stub mode.
    Positive weight pushes towards AI / deceptive.
    Negative weight pushes towards human / genuine.
    """
    explanations: list[TokenExplanation] = []
    seen = set()

    for idx, raw_word in enumerate(tokens):
        word = raw_word.lower()
        if word in seen or len(word) < 3:
            continue
        seen.add(word)

        # Base deterministic weight derived from token and global text seed
        token_hash = int.from_bytes(
            hashlib.sha256(f"{word}:{task}:{text_seed}".encode("utf-8")).digest()[:4],
            "big",
        )
        pseudo_random_weight = ((token_hash % 1000) - 500) / 1000.0  # [-0.5, 0.5]

        # Apply heuristic bias if keyword is known
        if task == "authorship":
            if word in AI_INDICATORS:
                weight = 0.45 + abs(pseudo_random_weight) * 0.45  # [+0.45, +0.90]
            elif word in HUMAN_INDICATORS:
                weight = -0.45 - abs(pseudo_random_weight) * 0.45  # [-0.90, -0.45]
            else:
                weight = pseudo_random_weight
        else:  # deception
            if word in DECEPTIVE_INDICATORS:
                weight = 0.45 + abs(pseudo_random_weight) * 0.45  # [+0.45, +0.90]
            elif word in GENUINE_INDICATORS:
                weight = -0.40 - abs(pseudo_random_weight) * 0.40  # [-0.80, -0.40]
            else:
                weight = pseudo_random_weight

        explanations.append(TokenExplanation(token=raw_word, weight=round(weight, 3)))

    # Sort tokens by absolute weight descending and take top 8
    explanations.sort(key=lambda x: abs(x.weight), reverse=True)
    return explanations[:8]

def analyze_review(text: str) -> AnalyzeResponse:
    """
    Analyze input review text across both Authorship and Deception dimensions.
    Returns probabilities, labels, combined verdict, and token-level explanations.
    """
    mode = detect_runtime_mode()
    warnings: list[str] = []

    if mode == "stub":
        warnings.append("Demo mode: models not loaded. Showing deterministic heuristic estimates.")

        # Compute deterministic probabilities from cryptographic hash of the input text
        digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
        seed_val = int.from_bytes(digest[:4], "big")
        
        # Scale hash values to [0.05, 0.95] for realistic probabilities
        hash_val_ai = int.from_bytes(digest[4:8], "big") / 0xFFFFFFFF
        hash_val_dec = int.from_bytes(digest[8:12], "big") / 0xFFFFFFFF
        
        # Keyword influence to make demo intuitive
        lower_text = text.lower()
        ai_boost = sum(0.12 for w in AI_INDICATORS if w in lower_text) - sum(0.12 for w in HUMAN_INDICATORS if w in lower_text)
        dec_boost = sum(0.12 for w in DECEPTIVE_INDICATORS if w in lower_text) - sum(0.12 for w in GENUINE_INDICATORS if w in lower_text)

        raw_ai_prob = 0.05 + 0.90 * hash_val_ai + ai_boost
        raw_dec_prob = 0.05 + 0.90 * hash_val_dec + dec_boost

        ai_probability = round(min(max(raw_ai_prob, 0.02), 0.98), 3)
        deceptive_probability = round(min(max(raw_dec_prob, 0.02), 0.98), 3)

        authorship_label: Literal["human", "ai"] = (
            "ai" if ai_probability >= settings.AUTHORSHIP_THRESHOLD else "human"
        )
        deception_label: Literal["genuine", "deceptive"] = (
            "deceptive" if deceptive_probability >= settings.DECEPTION_THRESHOLD else "genuine"
        )

        verdict = combine_predictions(authorship_label, deception_label)

        # Extract words for explanation highlights
        words = re.findall(r"[a-zA-Z]+", text)
        authorship_tokens = _extract_stub_token_weights(words, "authorship", seed_val)
        deception_tokens = _extract_stub_token_weights(words, "deception", seed_val)

        return AnalyzeResponse(
            authorship=AuthorshipResult(
                ai_probability=ai_probability,
                label=authorship_label,
            ),
            deception=DeceptionResult(
                deceptive_probability=deceptive_probability,
                label=deception_label,
            ),
            verdict=verdict,
            explanations=Explanations(
                authorship=authorship_tokens,
                deception=deception_tokens,
            ),
            model_info=ModelInfo(mode=mode),
            warnings=warnings,
        )

    # In future phases, classical and transformer loading branches are plugged here
    raise NotImplementedError(f"Runtime mode '{mode}' will be activated in later phases.")

