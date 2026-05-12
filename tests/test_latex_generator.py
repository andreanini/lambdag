"""Tests for LaTeX generator functions."""

import pytest
import pandas as pd
from lambdag.visualize.latex_generator import (
    color_coding_latex,
    color_coding_latex_unmasked,
)


def test_color_coding_latex():
    """Test color_coding_latex function."""
    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
        }
    )

    latex = color_coding_latex(df, scale="absolute")
    assert "\\code" in latex
    assert "hello" in latex
    assert "world" in latex


def test_color_coding_latex_unmasked():
    """Test color_coding_latex_unmasked function."""
    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
        }
    )

    # Note: sent_id is 1-indexed, so unmasked_sents[1] is accessed for sent_id=1
    # We need at least 2 elements in unmasked_sents (index 0 and 1)
    unmasked_sents = [["dummy"], ["hi", "there"]]
    latex_unmasked = color_coding_latex_unmasked(df, unmasked_sents, scale="absolute")
    assert "\\code" in latex_unmasked
    assert "hi" in latex_unmasked
    assert "there" in latex_unmasked


def test_latex_generator_with_eos():
    """Test LaTeX generator with EOS token."""
    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "___EOS___"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
        }
    )

    latex = color_coding_latex(df, scale="absolute")
    assert "\\code" in latex
    assert "[EOS]" in latex
