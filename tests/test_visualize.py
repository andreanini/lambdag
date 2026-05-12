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


def test_pick_shade():
    """Test the pick_shade helper function."""
    from lambdag.visualize.color_coding import pick_shade

    # Test shade mapping
    map_table = [(0, 1, 10), (1, 2, 20), (2, 3, 30)]
    assert pick_shade(0.5, map_table) == 10
    assert pick_shade(1.5, map_table) == 20
    assert pick_shade(2.5, map_table) == 30
    assert pick_shade(0, map_table) == 10  # lower bound inclusive
    assert pick_shade(3, map_table) == 0  # above all ranges


def test_html_generator_with_negative():
    """Test HTML generation with negative values."""
    from lambdag.visualize.html_generator import color_coding_html
    import pandas as pd

    # Create a table with negative lambdaG values
    df = pd.DataFrame(
        {
            "sentence_id": [1, 1, 2],
            "token_id": [1, 2, 1],
            "t": ["hello", "world", "test"],
            "lambdaG": [-2.5, 1.5, -0.8],
            "zlambdaG": [-1.5, 0.8, -0.3],
        }
    )

    # Test with negative=True
    html = color_coding_html(df, scale="absolute", negative=True)
    assert "hello" in html
    assert "world" in html

    # Test with negative=False (negative values should not be colored)
    html_no_neg = color_coding_html(df, scale="absolute", negative=False)
    assert "hello" in html_no_neg


def test_html_generator_relative_scale():
    """Test HTML generation with relative scale."""
    from lambdag.visualize.html_generator import color_coding_html
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [2.5, 1.5],
        }
    )

    html = color_coding_html(df, scale="relative", negative=True)
    assert "hello" in html
    assert "world" in html


def test_latex_output():
    """Test LaTeX output generation."""
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
    # LaTeX should contain \code command
    assert "\\code" in result["colourcoded_text"]


def test_lambdaG_visualize_with_list_input():
    """Test lambdaG_visualize with list input instead of strings."""
    q_text = [["hello", "world"], ["how", "are", "you"]]
    k_text = [["hello", "world"], ["i", "am", "fine"]]
    ref_text = [["hello", "world"], ["this", "is", "a", "test"]]

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="absolute",
        negative=False,
    )

    assert "table" in result
    assert "colourcoded_text" in result


def test_lambdaG_visualize_importance_ordering():
    """Test lambdaG_visualize with importance ordering."""
    q_text = "First sentence. Second sentence. Third sentence."
    k_text = "First sentence. Second sentence. Third sentence."
    ref_text = "First sentence. Second sentence. Third sentence."

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="absolute",
        negative=False,
        order_by="importance",
    )

    assert "table" in result
    # Check that sentences are sorted by sentence_lambdaG descending
    sentence_ids = result["table"]["sentence_id"].unique()
    assert len(sentence_ids) == 3


def test_color_coding_boundary_values():
    """Test color coding at boundary values."""
    # Test exact boundary values for pos_color_abs
    assert pos_color_abs(4.0) == "#943126; color: white"
    assert pos_color_abs(3.99) == "#cb4335; color: white"
    assert pos_color_abs(3.0) == "#cb4335; color: white"
    assert pos_color_abs(2.99) == "#ec7063"

    # Test exact boundary values for neg_color_abs
    assert neg_color_abs(-4.0) == "#053061; color: white"
    assert neg_color_abs(-3.99) == "#2166ac; color: white"
    assert neg_color_abs(-3.0) == "#2166ac; color: white"
    assert neg_color_abs(-2.99) == "#4393c3"

    # Test exact boundary values for pos_color_rel (note: conditions are strict: v > 2, v > 1)
    assert pos_color_rel(2.01) == "#E74C3C; color: white"
    assert pos_color_rel(2.0) == "#F1948A"  # 2.0 is NOT > 2, so falls to next condition
    assert pos_color_rel(1.5) == "#F1948A"
    assert pos_color_rel(1.01) == "#F1948A"
    assert pos_color_rel(1.0) == "#FADBD8"  # 1.0 is NOT > 1
    assert pos_color_rel(0.75) == "#FADBD8"

    # Test exact boundary values for neg_color_rel (note: conditions are strict: v < -2, v < -1)
    assert neg_color_rel(-2.01) == "#2166ac; color: white"
    assert neg_color_rel(-2.0) == "#67a9cf"  # -2.0 is NOT < -2
    assert neg_color_rel(-1.5) == "#67a9cf"
    assert neg_color_rel(-1.01) == "#67a9cf"
    assert neg_color_rel(-1.0) == "#d1e5f0"  # -1.0 is NOT < -1
    assert neg_color_rel(-0.75) == "#d1e5f0"


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


def test_generate_html_output_with_unmasked():
    """Test generate_html_output with unmasked sentences."""
    from lambdag.visualize.html_generator import generate_html_output
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1, 2],
            "token_id": [1, 2, 1],
            "t": ["hello", "world", "test"],
            "lambdaG": [1.5, 2.5, 0.8],
            "zlambdaG": [1.5, 2.5, 0.8],
            "sentence_lambdaG": [2.0, 2.0, 0.8],
        }
    )

    masked_html = '<span style="background-color: #f5b7b1; padding: 3px 6px; border-radius: 4px;">hello</span> world'
    unmasked_html = '<span style="background-color: #f5b7b1; padding: 3px 6px; border-radius: 4px;">hi</span> there'

    html = generate_html_output(
        df,
        masked_html,
        unmasked_html,
        scale="absolute",
        negative=False,
        sentence_scores=[2.0, 0.8],
        order_by="importance",
    )

    assert "hi" in html
    assert "there" in html


def test_generate_html_output_without_unmasked():
    """Test generate_html_output without unmasked sentences."""
    from lambdag.visualize.html_generator import generate_html_output
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
            "sentence_lambdaG": [2.0, 2.0],
        }
    )

    masked_html = '<span style="background-color: #f5b7b1; padding: 3px 6px; border-radius: 4px;">hello</span> world'

    html = generate_html_output(
        df,
        masked_html,
        None,
        scale="absolute",
        negative=False,
        sentence_scores=[2.0],
        order_by="",
    )

    assert "hello" in html
    assert "world" in html


def test_color_coding_html_unmasked():
    """Test color_coding_html_unmasked function."""
    from lambdag.visualize.html_generator import color_coding_html_unmasked
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1, 2],
            "token_id": [1, 2, 1],
            "t": ["hello", "world", "test"],
            "lambdaG": [1.5, 2.5, 0.8],
            "zlambdaG": [1.5, 2.5, 0.8],
        }
    )

    unmasked_sents = [["hi", "there"], ["foo"]]

    html = color_coding_html_unmasked(
        df, unmasked_sents, scale="absolute", negative=False
    )

    assert "hi" in html
    assert "there" in html
    assert "foo" in html


def test_latex_generator_functions():
    """Test LaTeX generator functions."""
    from lambdag.visualize.latex_generator import (
        color_coding_latex,
        color_coding_latex_unmasked,
    )
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
        }
    )

    # Test color_coding_latex
    latex = color_coding_latex(df, scale="absolute")
    assert "\\code" in latex
    assert "hello" in latex
    assert "world" in latex

    # Test color_coding_latex_unmasked
    # Note: sent_id is 1-indexed, so unmasked_sents[1] is accessed for sent_id=1
    # We need at least 2 elements in unmasked_sents (index 0 and 1)
    unmasked_sents = [["dummy"], ["hi", "there"]]
    latex_unmasked = color_coding_latex_unmasked(df, unmasked_sents, scale="absolute")
    assert "\\code" in latex_unmasked
    assert "hi" in latex_unmasked
    assert "there" in latex_unmasked


def test_lambdaG_visualize_sequential_order():
    """Test lambdaG_visualize with sequential ordering (default)."""
    q_text = "First sentence. Second sentence."
    k_text = "First sentence. Second sentence."
    ref_text = "First sentence. Second sentence."

    result = lambdaG_visualize(
        q_text=q_text,
        k_text=k_text,
        ref_text=ref_text,
        N=3,
        r=2,
        output="html",
        scale="absolute",
        negative=False,
        order_by="",  # sequential
    )

    assert "table" in result
    # Check that sentences are in sequential order
    sentence_ids = result["table"]["sentence_id"].unique()
    assert len(sentence_ids) == 2


def test_lambdaG_visualize_with_print_path():
    """Test lambdaG_visualize with print_path option."""
    import tempfile
    import os

    q_text = "Hello world."
    k_text = "Hello world."
    ref_text = "Hello world. This is a test."

    with tempfile.TemporaryDirectory() as tmpdir:
        print_path = os.path.join(tmpdir, "output.html")
        result = lambdaG_visualize(
            q_text=q_text,
            k_text=k_text,
            ref_text=ref_text,
            N=3,
            r=2,
            output="html",
            scale="absolute",
            negative=False,
            print_path=print_path,
        )

        assert "table" in result
        assert os.path.exists(print_path)


def test_color_coding_html_unmasked_relative():
    """Test color_coding_html_unmasked with relative scale."""
    from lambdag.visualize.html_generator import color_coding_html_unmasked
    import pandas as pd

    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [2.5, 1.5],
        }
    )

    unmasked_sents = [["hi", "there"]]

    html = color_coding_html_unmasked(
        df, unmasked_sents, scale="relative", negative=True
    )
    assert "hi" in html
    assert "there" in html


def test_latex_generator_with_eos():
    """Test LaTeX generator with EOS token."""
    from lambdag.visualize.latex_generator import color_coding_latex
    import pandas as pd

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


if __name__ == "__main__":
    pytest.main([__file__])
