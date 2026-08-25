"""Neutral phone-normalization helper shared by all repositories."""


def normalize_customer_phone(value: str) -> str:
    """Normalize a customer phone used as a storage key.

    Exact current semantics: collapse all whitespace runs to single spaces.
    """
    return " ".join(value.split())
