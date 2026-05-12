"""
lambdaG_visualize.py — Port of R lambdaG_visualize.R to Python
==============================================================

Ports the token-level LambdaG visualization from Nini et al. (under review)
to Python, using the `lambdag` package for language model fitting.

The R source: https://github.com/andreanini/idiolect/blob/main/R/lambdaG_visualize.R

Key differences from the R version:
  - Uses `lambdag.KneserNeyLanguageModel` instead of `kgrams::language_model`
  - Accepts raw text strings instead of `quanteda` tokens objects
  - Returns a pandas DataFrame instead of an R data.frame
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional
import numpy as np

# Import from submodules
from .llr_computation import loglikelihood_table_avgllrs
from .html_generator import (
    color_coding_html,
    color_coding_html_unmasked,
    generate_html_output,
)
from .latex_generator import color_coding_latex, color_coding_latex_unmasked

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lambdaG_visualize(
    q_text: str | list[str] | list[list[str]],
    k_text: str | list[str] | list[list[str]],
    ref_text: str | list[str] | list[list[str]],
    N: int = 10,
    r: int = 30,
    output: str = "html",
    print_path: str = "",
    scale: str = "absolute",
    negative: bool = False,
    order_by: str = "importance",
    unmasked_sentences: Optional[list[str] | list[list[str]]] = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Visualize LambdaG scores at the token level.

    Ports the R `lambdaG_visualize()` function from
    https://github.com/andreanini/idiolect/blob/main/R/lambdaG_visualize.R

    This function:
      1. Tokenizes input texts into sentences
      2. Runs r replications of token-level LLR computation
      3. Averages LLRs across replications
      4. Colour-codes tokens by their lambdaG score
      5. Returns the table and colour-coded text

    Args:
        q_text:           Questioned/disputed text (raw string or list of
                          sentences). If a string, it will be sentence-split.
        k_text:           Known text from the candidate translator (string or
                          sentences).
        ref_text:         Reference corpus text (string or sentences).
        N:                N-gram order for language models. Default: 10.
        r:                Number of replications (reference resamplings).
                          Default: 30.
        output:           Output format: "html" or "latex". Default: "html".
        print_path:       File path to save colour-coded text. Empty = don't
                          save.
        scale:            Colour scale: "absolute" (raw lambdaG) or "relative"
                          (z-score).
        negative:         If True, colour negative lambdaG values in blue
                          (HTML only).
        order_by:         If "importance", sort sentences by lambdaG descending.
        unmasked_sentences: Optional list of plain-text sentences (one per
                          sentence in q_text). If provided, the colour-coding
                          is applied to the real words instead of the masked
                          POS tags. The number of unmasked sentences must match
                          the number of sentences in q_text.
        rng:              Optional numpy random Generator for deterministic
                          sampling. If provided, the same seed will produce
                          identical results across runs.

    Returns:
        dict with keys:
          - "table": pd.DataFrame with per-token and per-sentence lambdaG scores
          - "colourcoded_text": HTML or LaTeX string of colour-coded text
    """
    # --- Tokenize / sentence-split ---
    # If strings are passed, do basic sentence splitting
    if isinstance(q_text, str):
        q_sents = _sentence_split(q_text)
    else:
        q_sents = [s.split() if isinstance(s, str) else list(s) for s in q_text]

    if isinstance(k_text, str):
        k_sents = _sentence_split(k_text)
    else:
        k_sents = [s.split() if isinstance(s, str) else list(s) for s in k_text]

    if isinstance(ref_text, str):
        ref_sents = _sentence_split(ref_text)
    else:
        ref_sents = [s.split() if isinstance(s, str) else list(s) for s in ref_text]

    # --- Process unmasked sentences if provided ---
    unmasked_sents = None
    if unmasked_sentences is not None:
        if isinstance(unmasked_sentences, str):
            unmasked_sents = _sentence_split(unmasked_sentences)
        else:
            unmasked_sents = [
                s.split() if isinstance(s, str) else list(s) for s in unmasked_sentences
            ]
        if len(unmasked_sents) != len(q_sents):
            raise ValueError(
                f"unmasked_sentences has {len(unmasked_sents)} sentences, "
                f"but q_text has {len(q_sents)} sentences. They must match."
            )

    # --- Validate ---
    if len(q_sents) == 0:
        raise ValueError("q_text must contain at least one sentence.")
    if len(k_sents) == 0:
        raise ValueError("k_text must contain at least one sentence.")
    if len(ref_sents) == 0:
        raise ValueError("ref_text must contain at least one sentence.")
    if r < 1:
        raise ValueError("r must be >= 1.")
    if N < 1:
        raise ValueError("N must be >= 1.")
    if output not in ("html", "latex"):
        raise ValueError("output must be 'html' or 'latex'.")
    if scale not in ("absolute", "relative"):
        raise ValueError("scale must be 'absolute' or 'relative'.")

    # --- Compute token-level LLR table ---
    llr_table = loglikelihood_table_avgllrs(
        q_sents,
        k_sents,
        ref_sents,
        r=r,
        order=N,
        order_by=order_by,
        rng=rng,
    )

    # --- Colour-code ---
    masked_html = ""
    unmasked_html = None

    if unmasked_sents is not None:
        # Reorder unmasked_sents to match the order of sentence_id in the (sorted) table
        # This ensures masked and unmasked sentences are in the same order
        unique_sentence_ids = llr_table["sentence_id"].unique()
        if len(unique_sentence_ids) <= len(unmasked_sents):
            reordered_unmasked = [
                unmasked_sents[sid - 1] for sid in unique_sentence_ids
            ]
        else:
            reordered_unmasked = unmasked_sents[: len(unique_sentence_ids)]

        if output == "html":
            masked_html = color_coding_html(llr_table, scale, negative)
            unmasked_html = color_coding_html_unmasked(
                llr_table, reordered_unmasked, scale, negative
            )
            # Get sentence scores for coloring numbers (preserve table order)
            sentence_df = llr_table.drop_duplicates("sentence_id")
            # Don't sort by sentence_id - preserve the order_by ordering
            sentence_scores = sentence_df["sentence_lambdaG"].tolist()
            cc_text = generate_html_output(
                llr_table,
                masked_html,
                unmasked_html,
                scale,
                negative,
                sentence_scores,
                order_by,
            )
        else:
            cc_text = color_coding_latex_unmasked(llr_table, reordered_unmasked, scale)
    else:
        if output == "html":
            masked_html = color_coding_html(llr_table, scale, negative)
            # Get sentence scores for coloring numbers (preserve table order)
            sentence_df = llr_table.drop_duplicates("sentence_id")
            # Don't sort by sentence_id - preserve the order_by ordering
            sentence_scores = sentence_df["sentence_lambdaG"].tolist()
            cc_text = generate_html_output(
                llr_table, masked_html, None, scale, negative, sentence_scores, order_by
            )
        else:
            cc_text = color_coding_latex(llr_table, scale)

    # --- Optional file output ---
    if print_path:
        with open(print_path, "w", encoding="utf-8") as f:
            f.write(cc_text)

    return {
        "table": llr_table,
        "colourcoded_text": cc_text,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentence_split(text: str) -> list[list[str]]:
    """Split text into sentences, then tokenize each sentence.

    Basic sentence splitting on .!? followed by whitespace or end-of-string.
    This is intentionally simple; for production use, consider spaCy or similar.
    """
    # Split on punctuation followed by whitespace or end of string
    sentences = re.split(r"(?<=[.!?])(?:\s+|$)", text.strip())
    # Strip punctuation from end of each sentence and tokenize
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Remove trailing punctuation from the last word of each sentence
        tokens = sent.split()
        if tokens:
            # Strip trailing punctuation from last token
            tokens[-1] = tokens[-1].rstrip(".!?")
            if tokens[-1]:  # Only add if not empty after stripping
                result.append(tokens)
    return result
