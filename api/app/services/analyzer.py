import hashlib
import re
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

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
from ml.src.data.cleaning import clean_text

REQUIRED_WARNINGS = [
    "Authorship model trained on Amazon-style and movie reviews",
    "Deception model trained on Chicago hotel reviews only; treat as indicative",
]

# Cached model instances
_AUTH_MODEL = None
_DEC_MODEL = None

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


def get_classical_models():
    """Load and cache classical pipelines from api/models/."""
    global _AUTH_MODEL, _DEC_MODEL
    if _AUTH_MODEL is None:
        auth_path = settings.MODELS_DIR / "authorship_classical.joblib"
        if auth_path.is_file():
            _AUTH_MODEL = joblib.load(auth_path)
    if _DEC_MODEL is None:
        dec_path = settings.MODELS_DIR / "deception_classical.joblib"
        if dec_path.is_file():
            _DEC_MODEL = joblib.load(dec_path)
    return _AUTH_MODEL, _DEC_MODEL


def _extract_model_token_explanations(pipeline, cleaned_text: str, top_k: int = 8) -> list[TokenExplanation]:
    """
    Extract token weights directly from the fitted TF-IDF + Linear model pipeline.
    Positive weight indicates class 1 (AI / Deceptive); negative indicates class 0 (Human / Genuine).
    """
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]

        if isinstance(clf, CalibratedClassifierCV):
            coef_list = []
            for cal_clf in clf.calibrated_classifiers_:
                base = getattr(cal_clf, "estimator", cal_clf)
                if hasattr(base, "coef_"):
                    coef_list.append(base.coef_)
            if coef_list:
                coefs = np.mean(coef_list, axis=0).ravel()
            else:
                return []
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_.ravel()
        else:
            return []

        vocab = tfidf.vocabulary_
        words = re.findall(r"\b\w+\b", cleaned_text.lower())
        seen = set()
        explanations = []

        for w in words:
            if w in seen or len(w) < 2:
                continue
            seen.add(w)
            if w in vocab:
                idx = vocab[w]
                weight = float(coefs[idx])
                explanations.append(TokenExplanation(token=w, weight=round(weight, 3)))

        explanations.sort(key=lambda x: abs(x.weight), reverse=True)
        return explanations[:top_k]
    except Exception:
        return []


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

        token_hash = int.from_bytes(
            hashlib.sha256(f"{word}:{task}:{text_seed}".encode("utf-8")).digest()[:4],
            "big",
        )
        pseudo_random_weight = ((token_hash % 1000) - 500) / 1000.0  # [-0.5, 0.5]

        if task == "authorship":
            if word in AI_INDICATORS:
                weight = 0.45 + abs(pseudo_random_weight) * 0.45
            elif word in HUMAN_INDICATORS:
                weight = -0.45 - abs(pseudo_random_weight) * 0.45
            else:
                weight = pseudo_random_weight
        else:
            if word in DECEPTIVE_INDICATORS:
                weight = 0.45 + abs(pseudo_random_weight) * 0.45
            elif word in GENUINE_INDICATORS:
                weight = -0.40 - abs(pseudo_random_weight) * 0.40
            else:
                weight = pseudo_random_weight

        explanations.append(TokenExplanation(token=raw_word, weight=round(weight, 3)))

    explanations.sort(key=lambda x: abs(x.weight), reverse=True)
    return explanations[:8]


def analyze_review(text: str) -> AnalyzeResponse:
    """
    Analyze input review text across both Authorship and Deception dimensions.
    Returns probabilities, labels, combined verdict, and token-level explanations.
    """
    # 1. Clean input text using shared cleaning pipeline
    cleaned_text = clean_text(text)
    mode = detect_runtime_mode()

    # Always include the required disclaimer warnings
    warnings: list[str] = list(REQUIRED_WARNINGS)

    if mode == "classical":
        auth_model, dec_model = get_classical_models()
        if auth_model is not None and dec_model is not None:
            # Inference on cleaned text
            ai_prob = float(auth_model.predict_proba([cleaned_text])[0, 1])
            dec_prob = float(dec_model.predict_proba([cleaned_text])[0, 1])

            ai_probability = round(min(max(ai_prob, 0.0), 1.0), 3)
            deceptive_probability = round(min(max(dec_prob, 0.0), 1.0), 3)

            authorship_label: Literal["human", "ai"] = (
                "ai" if ai_probability >= settings.AUTHORSHIP_THRESHOLD else "human"
            )
            deception_label: Literal["genuine", "deceptive"] = (
                "deceptive" if deceptive_probability >= settings.DECEPTION_THRESHOLD else "genuine"
            )

            verdict = combine_predictions(authorship_label, deception_label)

            auth_tokens = _extract_model_token_explanations(auth_model, cleaned_text)
            dec_tokens = _extract_model_token_explanations(dec_model, cleaned_text)

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
                    authorship=auth_tokens,
                    deception=dec_tokens,
                ),
                model_info=ModelInfo(mode=mode),
                warnings=warnings,
            )

    # Fallback to stub mode
    warnings.insert(0, "Demo mode: models not loaded. Showing deterministic heuristic estimates.")

    digest = hashlib.sha256(cleaned_text.strip().encode("utf-8")).digest()
    seed_val = int.from_bytes(digest[:4], "big")

    hash_val_ai = int.from_bytes(digest[4:8], "big") / 0xFFFFFFFF
    hash_val_dec = int.from_bytes(digest[8:12], "big") / 0xFFFFFFFF

    lower_text = cleaned_text.lower()
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

    words = re.findall(r"[a-zA-Z]+", cleaned_text)
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
        model_info=ModelInfo(mode="stub"),
        warnings=warnings,
    )
