"""Tests for HTML generator functions."""

import pytest
import pandas as pd
from lambdag.visualize.html_generator import (
    color_coding_html,
    color_coding_html_unmasked,
    generate_html_output,
)


def test_color_coding_html_basic():
    """Test basic HTML color coding."""
    df = pd.DataFrame(
        {
            "sentence_id": [1, 1],
            "token_id": [1, 2],
            "t": ["hello", "world"],
            "lambdaG": [1.5, 2.5],
            "zlambdaG": [1.5, 2.5],
        }
    )

    html = color_coding_html(df, scale="absolute", negative=False)
    assert "hello" in html
    assert "world" in html


def test_html_generator_with_negative():
    """Test HTML generation with negative values."""
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


def test_color_coding_html_unmasked():
    """Test color_coding_html_unmasked function."""
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


def test_color_coding_html_unmasked_relative():
    """Test color_coding_html_unmasked with relative scale."""
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


def test_generate_html_output_with_unmasked():
    """Test generate_html_output with unmasked sentences."""
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
