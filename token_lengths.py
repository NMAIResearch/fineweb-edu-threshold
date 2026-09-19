"""Convert frozen single-sequence encoded lengths to content-only token counts."""

import numpy as np

CONTENT_BUDGET = 510
SPECIAL_TOKENS = 2


def content_lengths(encoded_lengths):
    """The frozen bert_tokens column includes one CLS and one SEP token."""
    values = np.asarray(encoded_lengths)
    if values.dtype.kind not in "iu" or np.any(values < SPECIAL_TOKENS):
        raise ValueError("encoded lengths must be integers including both special tokens")
    return values.astype(np.int64) - SPECIAL_TOKENS
