"""
language_model.py — Kneser-Ney language model fitting for LambdaG visualization
==============================================================================

Contains functions for fitting Kneser-Ney language models used in LambdaG
computations.
"""

from __future__ import annotations
from typing import Iterable
from lambdag.language_models.kneser_ney import KneserNeyLanguageModel


def fit_kn_language_model(
    sentences: Iterable[Iterable[str]],
    order: int = 10,
    discount: float = 0.75,
) -> KneserNeyLanguageModel:
    """Fit a Kneser-Ney language model on the given sentences.

    This replaces the R `extract()` function which called
    `kgrams::kgram_freqs()` then `kgrams::language_model(smoother="kn", D=0.75)`.

    Args:
        sentences: Iterable of sentences, each an iterable of tokens.
        order:     N-gram order.
        discount:  KN discount parameter (default 0.75, matching the R code).

    Returns:
        A fitted KneserNeyLanguageModel.
    """
    lm = KneserNeyLanguageModel(
        order=order,
        discount=discount,
        special_handling_of_pad_start_element=True,
    )
    for sent in sentences:
        lm.fit(sent)
    return lm
