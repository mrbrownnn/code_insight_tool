"""
Token counting utility for chunk size estimation.
Simple heuristic: ~4 characters per token (for code).
"""


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic for code: average ~4 characters per token.
    This is a rough estimate; actual tokenization varies by model.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def is_within_token_limit(text: str, max_tokens: int) -> bool:
    """Check if text is within the specified token limit.

    Args:
        text: The text to check.
        max_tokens: Maximum allowed tokens.

    Returns:
        True if within limit.
    """
    return estimate_tokens(text) <= max_tokens
