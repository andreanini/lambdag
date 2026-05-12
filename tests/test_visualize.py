"""Tests for the lambdaG visualization functionality."""

import pytest
import numpy as np
import pandas as pd
from lambdag.visualize import lambdaG_visualize
from lambdag.visualize.lambdaG_visualize import _sentence_split
from lambdag.visualize.color_coding import (
    pos_color_abs,
    neg_color_abs,
    pos_color_rel,
    neg_color_rel,
)
from lambdag.visualize.llr_computation import (
    loglikelihood_one_rep,
    loglikelihood_table_avgllrs,
)


def test_lambdaG_visualize_basic():
    """Test basic functionality of lambdaG_visualize."""
    # Simple example texts
    q_text = "The quick brown fox jumps over the lazy dog."
    k_text = "A quick brown fox jumps over a lazy dog."
    ref_text = "The quick brown fox jumps over the lazy dog. The dog is lazy but the fox is quick."

    # Compute visualization
    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,  # small N for testing
        r=2,  # few replications for speed
        output="html",
        scale="absolute",
        negative=False,
    )

    # Check that we get the expected keys
    assert "table" in result
    assert "colourcoded_text" in result

    # Check that the table is a DataFrame with expected columns
    import pandas as pd

    assert isinstance(result["table"], pd.DataFrame)
    expected_columns = {
        "sentence_id",
        "token_id",
        "t",
        "lambdaG",
        "sentence_lambdaG",
        "zlambdaG",
        "token_contribution",
        "sent_contribution",
    }
    assert expected_columns.issubset(set(result["table"].columns))

    # Check that colourcoded_text is a string
    assert isinstance(result["colourcoded_text"], str)
    assert len(result["colourcoded_text"]) > 0


def test_lambdaG_visualize_with_unmasked():
    """Test lambdaG_visualize with unmasked sentences."""
    q_text = "Hello world. How are you?"
    k_text = "Hello world. How are you doing?"
    ref_text = "Hello world. How are you? I am fine thank you."
    unmasked_sentences = [["Hello", "world"], ["How", "are", "you"]]

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="absolute",
        negative=False,
        unmasked_sentences=unmasked_sentences,
    )

    assert "table" in result
    assert "colourcoded_text" in result
    assert isinstance(result["colourcoded_text"], str)


def test_lambdaG_visualize_latex_output():
    """Test lambdaG_visualize with LaTeX output."""
    q_text = "Hello world."
    k_text = "Hello world."
    ref_text = "Hello world. This is a test."

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="latex",
        scale="absolute",
        negative=False,
    )

    assert "table" in result
    assert "colourcoded_text" in result
    assert isinstance(result["colourcoded_text"], str)
    assert len(result["colourcoded_text"]) > 0


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Empty texts
    with pytest.raises(ValueError, match="q_text must contain at least one sentence"):
        lambdaG_visualize("", "k text", "ref text")

    with pytest.raises(ValueError, match="k_text must contain at least one sentence"):
        lambdaG_visualize("q text", "", "ref text")

    with pytest.raises(ValueError, match="ref_text must contain at least one sentence"):
        lambdaG_visualize("q text", "k text", "")

    # Invalid parameters
    with pytest.raises(ValueError, match="r must be >= 1"):
        lambdaG_visualize("q", "k", "ref", r=0)

    with pytest.raises(ValueError, match="N must be >= 1"):
        lambdaG_visualize("q", "k", "ref", N=0)

    with pytest.raises(ValueError, match="output must be 'html' or 'latex'"):
        lambdaG_visualize("q", "k", "ref", output="invalid")

    with pytest.raises(ValueError, match="scale must be 'absolute' or 'relative'"):
        lambdaG_visualize("q", "k", "ref", scale="invalid")

    # Mismatched unmasked sentences
    with pytest.raises(
        ValueError, match="unmasked_sentences has.*sentences, but q_text has"
    ):
        lambdaG_visualize(
            "q text.",
            "k text.",
            "ref text.",
            unmasked_sentences=[["too", "many"], ["sentences"]],
        )


def test_deterministic_output():
    """Test that the same seed produces identical results."""
    q_text = "Hello world. How are you?"
    k_text = "Hello world. How are you doing?"
    ref_text = "Hello world. How are you? I am fine thank you."

    # Run twice with same seed
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=5,
        output="html",
        rng=rng1,
    )

    result2 = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=5,
        output="html",
        rng=rng2,
    )

    # Tables should be identical
    pd.testing.assert_frame_equal(result1["table"], result2["table"])


def test_sentence_split():
    """Test the _sentence_split helper function."""
    # Basic sentence splitting
    result = _sentence_split("Hello world. How are you?")
    assert len(result) == 2
    assert result[0] == ["Hello", "world"]
    assert result[1] == ["How", "are", "you"]

    # Multiple punctuation
    result = _sentence_split("Hello! How are you? I am fine.")
    assert len(result) == 3

    # Single sentence
    result = _sentence_split("Hello world")
    assert len(result) == 1
    assert result[0] == ["Hello", "world"]


def test_color_coding_functions():
    """Test color coding functions with boundary values."""
    # Test pos_color_abs
    assert pos_color_abs(4.0) == "#943126; color: white"
    assert pos_color_abs(3.0) == "#cb4335; color: white"
    assert pos_color_abs(2.0) == "#ec7063"
    assert pos_color_abs(1.0) == "#f5b7b1"
    assert pos_color_abs(0.5) == "#fdedec"
    assert pos_color_abs(0.3) == ""

    # Test neg_color_abs
    assert neg_color_abs(-4.0) == "#053061; color: white"
    assert neg_color_abs(-3.0) == "#2166ac; color: white"
    assert neg_color_abs(-2.0) == "#4393c3"
    assert neg_color_abs(-1.0) == "#92c5de"
    assert neg_color_abs(-0.5) == "#d1e5f0"
    assert neg_color_abs(-0.3) == ""

    # Test pos_color_rel (note: conditions are strict inequalities)
    assert pos_color_rel(2.1) == "#E74C3C; color: white"
    assert pos_color_rel(1.5) == "#F1948A"
    assert pos_color_rel(0.7) == "#FADBD8"
    assert pos_color_rel(0.3) == ""

    # Test neg_color_rel
    assert neg_color_rel(-2.1) == "#2166ac; color: white"
    assert neg_color_rel(-1.5) == "#67a9cf"
    assert neg_color_rel(-0.7) == "#d1e5f0"
    assert neg_color_rel(-0.3) == ""


def test_lambdaG_visualize_relative_scale():
    """Test lambdaG_visualize with relative (z-score) scale."""
    q_text = "Hello world. How are you?"
    k_text = "Hello world. How are you doing?"
    ref_text = "Hello world. How are you? I am fine thank you."

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="relative",
        negative=True,
    )

    assert "table" in result
    assert "colourcoded_text" in result


def test_lambdaG_visualize_negative_coloring():
    """Test lambdaG_visualize with negative value coloring."""
    q_text = "Hello world."
    k_text = "Hello world."
    ref_text = "Hello world. This is a test."

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="absolute",
        negative=True,
    )

    assert "table" in result
    assert "colourcoded_text" in result


if __name__ == "__main__":
    pytest.main([__file__])
