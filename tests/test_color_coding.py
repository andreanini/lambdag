"""Tests for color coding functions."""

import pytest
from lambdag.visualize.color_coding import (
    pos_color_abs,
    neg_color_abs,
    pos_color_rel,
    neg_color_rel,
    pick_shade,
)


def test_pos_color_abs():
    """Test pos_color_abs with various values."""
    assert pos_color_abs(4.0) == "#943126; color: white"
    assert pos_color_abs(3.0) == "#cb4335; color: white"
    assert pos_color_abs(2.0) == "#ec7063"
    assert pos_color_abs(1.0) == "#f5b7b1"
    assert pos_color_abs(0.5) == "#fdedec"
    assert pos_color_abs(0.3) == ""


def test_neg_color_abs():
    """Test neg_color_abs with various values."""
    assert neg_color_abs(-4.0) == "#053061; color: white"
    assert neg_color_abs(-3.0) == "#2166ac; color: white"
    assert neg_color_abs(-2.0) == "#4393c3"
    assert neg_color_abs(-1.0) == "#92c5de"
    assert neg_color_abs(-0.5) == "#d1e5f0"
    assert neg_color_abs(-0.3) == ""


def test_pos_color_rel():
    """Test pos_color_rel with various values."""
    # Note: conditions are strict inequalities (v > 2, v > 1)
    assert pos_color_rel(2.1) == "#E74C3C; color: white"
    assert pos_color_rel(1.5) == "#F1948A"
    assert pos_color_rel(0.7) == "#FADBD8"
    assert pos_color_rel(0.3) == ""


def test_neg_color_rel():
    """Test neg_color_rel with various values."""
    # Note: conditions are strict inequalities (v < -2, v < -1)
    assert neg_color_rel(-2.1) == "#2166ac; color: white"
    assert neg_color_rel(-1.5) == "#67a9cf"
    assert neg_color_rel(-0.7) == "#d1e5f0"
    assert neg_color_rel(-0.3) == ""


def test_pick_shade():
    """Test the pick_shade helper function."""
    map_table = [(0, 1, 10), (1, 2, 20), (2, 3, 30)]
    assert pick_shade(0.5, map_table) == 10
    assert pick_shade(1.5, map_table) == 20
    assert pick_shade(2.5, map_table) == 30
    assert pick_shade(0, map_table) == 10  # lower bound inclusive
    assert pick_shade(3, map_table) == 0  # above all ranges


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
    assert pos_color_rel(2.0) == "#F1948A"  # 2.0 is NOT > 2
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
