"""Boundary and invalid-input regressions for frozen encoded token lengths."""

import unittest
import numpy as np
from token_lengths import content_lengths, CONTENT_BUDGET


class TokenLengthTests(unittest.TestCase):
    def test_exact_window_and_first_truncated_document(self):
        content = content_lengths([2, 511, 512, 513])
        np.testing.assert_array_equal(content, [0, 509, 510, 511])
        np.testing.assert_array_equal(content > CONTENT_BUDGET, [False, False, False, True])
        np.testing.assert_array_equal(np.minimum(content, CONTENT_BUDGET), [0, 509, 510, 510])

    def test_invalid_encoded_lengths_fail_closed(self):
        for value in [[1], [-1], [512.5], [True], [float("nan")]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                content_lengths(value)


if __name__ == "__main__":
    unittest.main()
