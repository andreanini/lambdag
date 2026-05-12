"""
latex_generator.py — LaTeX output generation for LambdaG visualization
====================================================================

Contains functions for generating LaTeX output with color-coded tokens.
"""

from __future__ import annotations
from typing import cast
import pandas as pd
from .color_coding import pick_shade


def color_coding_latex(llr_table: pd.DataFrame, scale: str = "absolute") -> str:
    """Generate LaTeX colour-coded text from the LLR table.

    Replaces the R `color_coding_latex()` function.

    Uses the `colorlow` LaTeX command pattern from the R code:
      \\code{colorlow!<shade>}{\\strut <token>}\\allowbreak

    Args:
        llr_table: DataFrame from loglikelihood_table_avgllrs.
        scale:     "absolute" (raw lambdaG) or "relative" (z-score).

    Returns:
        LaTeX string with colour-coded tokens.
    """
    # Absolute: map lambdaG ranges to shade percentages (0-100)
    _ABS_SHADE_MAP = [
        (0, 1, 20),
        (1, 2, 40),
        (2, 3, 60),
        (3, 4, 80),
        (4, float("inf"), 100),
    ]
    # Relative: map zlambdaG ranges to shade percentages
    _REL_SHADE_MAP = [
        (0.5, 1.0, 20),
        (1.0, 2.0, 50),
        (2.0, float("inf"), 70),
    ]

    lines = []
    for _sid, grp in llr_table.groupby("sentence_id"):
        parts = []
        for _, row in grp.iterrows():
            token = row["t"]
            lam = row["lambdaG"]
            zlam = row["zlambdaG"]

            if token == "___EOS___":
                token_display = "[EOS]"
            else:
                token_display = token

            if scale == "absolute":
                shade = pick_shade(abs(lam), _ABS_SHADE_MAP) if lam > 0 else 0
            elif scale == "relative":
                shade = pick_shade(abs(zlam), _REL_SHADE_MAP) if zlam > 0 else 0
            else:
                shade = 0

            if token == "___EOS___":
                parts.append(
                    f"\\code{{colorlow!{shade}}}{{\\strut {token_display}}}"
                    f"\\allowbreak\\newline"
                )
            else:
                parts.append(
                    f"\\code{{colorlow!{shade}}}{{\\strut {token_display}}}\\allowbreak"
                )
        lines.append("".join(parts))

    return "\n".join(lines)


def color_coding_latex_unmasked(
    llr_table: pd.DataFrame,
    unmasked_sents: list[list[str]],
    scale: str = "absolute",
) -> str:
    """LaTeX colour-coding applied to real (unmasked) words.

    Uses the lambdaG scores from the masked tokens to colour the
    corresponding unmasked words.

    Args:
        llr_table:    DataFrame from loglikelihood_table_avgllrs.
        unmasked_sents: List of token-lists (real words, one per sentence).
        scale:        "absolute" or "relative".

    Returns:
        LaTeX string with colour-coded unmasked tokens, one sentence per line.
    """
    _ABS_SHADE_MAP = [
        (0, 1, 20),
        (1, 2, 40),
        (2, 3, 60),
        (3, 4, 80),
        (4, float("inf"), 100),
    ]
    _REL_SHADE_MAP = [
        (0.5, 1.0, 20),
        (1.0, 2.0, 50),
        (2.0, float("inf"), 70),
    ]

    lines = []
    for sent_id_raw, grp in llr_table.groupby("sentence_id"):
        sent_id: int = cast(int, sent_id_raw)
        if sent_id >= len(unmasked_sents):
            break
        unmasked_tokens = unmasked_sents[sent_id]
        parts = []
        for (_, row), uword in zip(grp.iterrows(), unmasked_tokens):
            lam = row["lambdaG"]
            zlam = row["zlambdaG"]

            if scale == "absolute":
                shade = pick_shade(abs(lam), _ABS_SHADE_MAP) if lam > 0 else 0
            elif scale == "relative":
                shade = pick_shade(abs(zlam), _REL_SHADE_MAP) if zlam > 0 else 0
            else:
                shade = 0

            parts.append(f"\\code{{colorlow!{shade}}}{{\\strut {uword}}}\\allowbreak")
        lines.append("".join(parts))

    return "\n".join(lines)
