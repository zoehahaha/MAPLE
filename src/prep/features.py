from __future__ import annotations

import re

FEATURE_TOKEN_CAP = 64


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", text.lower())[:FEATURE_TOKEN_CAP])


def input_features(text: str) -> frozenset[str]:
    """Bounded token, four-character-prefix, and structural input features."""
    tokens = _tokenize(text)
    token_set = set(tokens)
    result = {f"tok:{token}" for token in token_set}
    result.update(f"pre4:{token[:4]}" for token in token_set if len(token) >= 4)
    result.add("cue:negation" if token_set & {"no", "not", "never", "without", "dont", "don't"}
               else "cue:no_negation")
    result.add("cue:question" if "?" in text else "cue:no_question")
    result.add("cue:numeric" if any(token.isdigit() for token in tokens) else "cue:no_numeric")
    result.add("cue:short" if len(tokens) <= 12 else "cue:not_short")
    result.add("cue:long" if len(tokens) >= FEATURE_TOKEN_CAP else "cue:not_long")
    return frozenset(result)


def guard_fires(guard: tuple[tuple[str, bool], ...], features: frozenset[str]) -> bool:
    return all((name in features) is expected for name, expected in guard)

