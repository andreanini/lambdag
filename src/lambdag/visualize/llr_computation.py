"""
llr_computation.py — Log-likelihood ratio computations for LambdaG visualization
================================================================================

Contains functions for computing token-level log-likelihood ratios and averaging
over replications.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from lambdag.language_models.kneser_ney import KneserNeyLanguageModel
from .language_model import fit_kn_language_model


def loglikelihood_one_rep(
    q_sents: list[list[str]],
    k_lm: KneserNeyLanguageModel,
    k_sents: list[list[str]],
    ref_sents: list[list[str]],
    order: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One replication of token-level log-likelihood ratios.

    Replaces the R `loglikelihood_one_rep()` function.

    For each sentence in q_sents:
      - Computes sentence-level LLR (sum log2 known_probs - sum log2 ref_probs)
      - For each token position, computes token-level LLR

    Args:
        q_sents:    List of token-lists (the questioned text, split by sentence).
        k_lm:       Fitted KN language model for the known author.
        k_sents:    List of token-lists (known author sentences, for sample sizing).
        ref_sents:  List of token-lists (reference corpus sentences).
        order:      N-gram order.
        rng:        Optional numpy random Generator for deterministic sampling.

    Returns:
        DataFrame with columns: sentence_id, token_id, t, k, ref, llr, sentence_llr
    """
    # Build a reference LM from a random sample (same size as known)
    if rng is None:
        rng = np.random.default_rng()
    sample_idx = rng.choice(len(ref_sents), size=len(k_sents), replace=False)
    sampled_ref = [ref_sents[i] for i in sample_idx]
    ref_lm = fit_kn_language_model(sampled_ref, order=order)

    rows = []
    for sent_idx, sent_tokens in enumerate(q_sents):
        # Pad the sentence for probability computation
        # KN model uses <s> BOS and </s> EOS
        # Use pad_sequence=True to add EOS token (matching R's kgrams::EOS())
        known_probs = list(k_lm.probabilities(sent_tokens, pad_sequence=True))
        ref_probs = list(ref_lm.probabilities(sent_tokens, pad_sequence=True))

        # Add EOS token to the token list for output table
        tokens_with_eos = sent_tokens + [k_lm.pad_end_element]

        # Sentence-level LLR (using log10 to match R's loglikelihood_one_rep)
        # Sum over all tokens INCLUDING EOS
        sentence_llr = float(
            np.sum(np.log10(known_probs)) - np.sum(np.log10(ref_probs))
        )

        # Token-level LLR (including EOS)
        # Use 1-indexed sentence_id to match R (which uses 1:length(q.sents))
        for token_idx, (tok, kp, rp) in enumerate(
            zip(tokens_with_eos, known_probs, ref_probs)
        ):
            token_llr = math.log10(kp / rp) if kp > 0 and rp > 0 else 0.0
            rows.append(
                {
                    "sentence_id": sent_idx + 1,  # 1-indexed to match R
                    "token_id": token_idx + 1,
                    "t": tok,
                    "k": kp,
                    "ref": rp,
                    "llr": token_llr,
                    "sentence_llr": round(sentence_llr, 3),
                }
            )

    return pd.DataFrame(rows)


def loglikelihood_table_avgllrs(
    q_sents: list[list[str]],
    k_sents: list[list[str]],
    ref_sents: list[list[str]],
    r: int = 30,
    order: int = 10,
    order_by: str = "importance",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Average token-level LLRs over r replications.

    Replaces the R `loglikelihood_table_avgllrs()` function.

    Args:
        q_sents:    Tokenized sentences of the questioned text.
        k_sents:    Tokenized sentences of the known author/translator.
        ref_sents:  Tokenized sentences of the reference corpus.
        r:          Number of replications (resampling of reference).
        order:      N-gram order.
        order_by:   If "importance", sort by sentence_lambdaG descending.
        rng:        Optional numpy random Generator for deterministic sampling.

    Returns:
        DataFrame with added columns: lambdaG, zlambdaG, token_contribution,
        sent_contribution.
    """
    # Fit the known language model once (deterministic)
    k_lm = fit_kn_language_model(k_sents, order=order)

    # Run r replications
    all_frames = []
    for _ in range(r):
        df = loglikelihood_one_rep(q_sents, k_lm, k_sents, ref_sents, order, rng)
        all_frames.append(df)

    # Concatenate and group by (sentence_id, token_id, t)
    combined = pd.concat(all_frames, ignore_index=True)
    grouped = (
        combined.groupby(["sentence_id", "token_id", "t"])
        .agg(
            lambdaG=("llr", "mean"),
            sentence_lambdaG=("sentence_llr", "mean"),
        )
        .reset_index()
    )

    # Round sentence_lambdaG
    grouped["sentence_lambdaG"] = grouped["sentence_lambdaG"].round(3)

    # Z-score of lambdaG
    grouped["zlambdaG"] = (
        (grouped["lambdaG"] - grouped["lambdaG"].mean()) / grouped["lambdaG"].std()
    ).round(3)

    # --- Relative contribution in percentage ---
    pos_total = grouped.loc[grouped["lambdaG"] > 0, "lambdaG"].sum()
    neg_total = grouped.loc[grouped["lambdaG"] < 0, "lambdaG"].sum()

    pos_s_total = (
        grouped.drop_duplicates("sentence_id")
        .loc[lambda df: df["sentence_lambdaG"] > 0, "sentence_lambdaG"]
        .sum()
    )
    neg_s_total = (
        grouped.drop_duplicates("sentence_id")
        .loc[lambda df: df["sentence_lambdaG"] < 0, "sentence_lambdaG"]
        .sum()
    )

    grouped["token_contribution"] = grouped["lambdaG"].apply(
        lambda x: (
            round(x / pos_total * 100, 2)
            if x > 0
            else round(-x / neg_total * 100, 2) if x < 0 else 0.0
        )
    )
    grouped["sent_contribution"] = grouped["sentence_lambdaG"].apply(
        lambda x: (
            round(x / pos_s_total * 100, 2)
            if x > 0
            else round(-x / neg_s_total * 100, 2) if x < 0 else 0.0
        )
    )

    # Handle edge cases where totals are 0
    if pos_total == 0 and neg_total == 0:
        grouped["token_contribution"] = 0.0
    if pos_s_total == 0 and neg_s_total == 0:
        grouped["sent_contribution"] = 0.0

    # Optional ordering
    if order_by == "importance":
        grouped = grouped.sort_values(
            ["sentence_lambdaG", "sentence_id", "token_id"],
            ascending=[False, True, True],
        )

    return grouped.reset_index(drop=True)
