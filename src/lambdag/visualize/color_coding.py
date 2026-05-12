"""
color_coding.py — Color mapping functions for LambdaG visualization
====================================================================

Contains all functions for mapping lambdaG and zlambdaG values to colors
for HTML and LaTeX output.
"""

from __future__ import annotations


def pos_color_abs(v: float) -> str:
    """Absolute positive lambdaG → red shades (from R code)."""
    if v >= 4:
        return "#943126; color: white"
    if v >= 3:
        return "#cb4335; color: white"
    if v >= 2:
        return "#ec7063"
    if v >= 1:
        return "#f5b7b1"
    if v >= 0.5:
        return "#fdedec"
    return ""  # 0 < v < 0.5 → no color


def neg_color_abs(v: float) -> str:
    """Absolute negative lambdaG → blue shades (from R code)."""
    if v <= -4:
        return "#053061; color: white"
    if v <= -3:
        return "#2166ac; color: white"
    if v <= -2:
        return "#4393c3"
    if v <= -1:
        return "#92c5de"
    if v <= -0.5:
        return "#d1e5f0"
    return ""  # -0.5 < v < 0 → no color (handled by default)


def pos_color_rel(v: float) -> str:
    """Relative positive zlambdaG → red shades (from R code)."""
    if v > 2:
        return "#E74C3C; color: white"
    if v > 1:
        return "#F1948A"
    if v > 0.5:
        return "#FADBD8"
    return ""


def neg_color_rel(v: float) -> str:
    """Relative negative zlambdaG → blue shades (from R code)."""
    if v < -2:
        return "#2166ac; color: white"
    if v < -1:
        return "#67a9cf"
    if v < -0.5:
        return "#d1e5f0"
    return ""


def pick_shade(val: float, map_table: list) -> int:
    for lo, hi, shade in map_table:
        if lo <= val < hi:
            return shade
    return 0
