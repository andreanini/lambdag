"""
html_generator.py — HTML output generation for LambdaG visualization
====================================================================

Contains functions for generating HTML output with color-coded tokens.
"""

from __future__ import annotations
import pandas as pd
from .color_coding import pos_color_abs, neg_color_abs, pos_color_rel, neg_color_rel
from typing import cast


def color_coding_html(
    llr_table: pd.DataFrame,
    scale: str = "absolute",
    negative: bool = False,
) -> str:
    """Generate HTML colour-coded text from the LLR table.

    Replaces the R `color_coding_html()` function.

    Args:
        llr_table: DataFrame from loglikelihood_table_avgllrs.
        scale:     "absolute" (raw lambdaG) or "relative" (z-score).
        negative:  If True, colour negative values in blue shades.

    Returns:
        HTML string with <span> elements coloured by lambdaG magnitude.
    """
    # Group tokens by sentence, colour-code each, then join sentences
    # on separate lines. Use sort=False to preserve table order (e.g., sorted by sentence_lambdaG).
    lines = []
    for _sid, grp in llr_table.groupby("sentence_id", sort=False):
        parts = []
        for _, row in grp.iterrows():
            lam = row["lambdaG"]
            zlam = row["zlambdaG"]
            token = row["t"]

            # Display EOS token visibly (stored as </s> in table)
            display_token = "EOS" if token == "</s>" else token

            color = ""

            if scale == "absolute":
                if lam > 0:
                    color = pos_color_abs(lam)
                elif negative and lam < 0:
                    color = neg_color_abs(lam)
            elif scale == "relative":
                if zlam > 0:
                    color = pos_color_rel(zlam)
                elif negative and zlam < 0:
                    color = neg_color_rel(zlam)

            if color:
                parts.append(
                    f'<span style="background-color: {color}; padding: 3px 6px; border-radius: 4px;">{display_token}</span>'
                )
            else:
                # Don't display EOS token when there's no background color
                if token != "</s>":
                    parts.append(token)
        lines.append(" ".join(parts))

    return "<br>\n".join(lines)


def color_coding_html_unmasked(
    llr_table: pd.DataFrame,
    unmasked_sents: list[list[str]],
    scale: str = "absolute",
    negative: bool = False,
) -> str:
    """HTML colour-coding applied to real (unmasked) words.

    Uses the lambdaG scores from the masked tokens to colour the
    corresponding unmasked words.  The number of unmasked tokens per
    sentence must match the number of masked tokens.

    Args:
        llr_table:    DataFrame from loglikelihood_table_avgllrs.
        unmasked_sents: List of token-lists (real words, one per sentence).
        scale:        "absolute" or "relative".
        negative:     If True, colour negative values in blue shades.

    Returns:
        HTML string with <span> elements coloured by lambdaG magnitude,
        one sentence per line.
    """
    # Iterate through table in order (preserving sentence_lambdaG sorting)
    # Use sort=False to maintain the order of the table
    # unmasked_sents is already reordered to match the table's sentence_id order
    lines = []
    seen_ids = set()
    sent_idx = 0  # Index into the reordered unmasked_sents list.

    for sent_id_raw, grp in llr_table.groupby("sentence_id", sort=False):
        sent_id: int = cast(int, sent_id_raw)
        if sent_id in seen_ids:
            continue  # Skip duplicates (shouldn't happen, but safety)
        seen_ids.add(sent_id)

        if sent_idx >= len(unmasked_sents):
            break
        # unmasked_sents[sent_idx] corresponds to the sentence at position sent_idx in the sorted table
        unmasked_tokens = unmasked_sents[sent_idx]
        parts = []
        for (_, row), uword in zip(grp.iterrows(), unmasked_tokens):
            lam = row["lambdaG"]
            zlam = row["zlambdaG"]

            color = ""
            if scale == "absolute":
                if lam > 0:
                    color = pos_color_abs(lam)
                elif negative and lam < 0:
                    color = neg_color_abs(lam)
            elif scale == "relative":
                if zlam > 0:
                    color = pos_color_rel(zlam)
                elif negative and zlam < 0:
                    color = neg_color_rel(zlam)

            if color:
                # Add padding and border-radius for better visual appearance
                parts.append(
                    f'<span style="background-color: {color}; padding: 3px 6px; border-radius: 4px;">{uword}</span>'
                )
            else:
                parts.append(uword)
        lines.append(" ".join(parts))
        sent_idx += 1

    return "<br>\n".join(lines)


def generate_html_output(
    llr_table: pd.DataFrame,
    masked_html: str,
    unmasked_html: str | None = None,
    scale: str = "absolute",
    negative: bool = False,
    sentence_scores: list[float] | None = None,
    order_by: str = "",
) -> str:
    """Generate a complete HTML document with poster-style layout.

    Args:
        llr_table: DataFrame with lambdaG scores.
        masked_html: HTML string for masked (POS-tagged) sentences.
        unmasked_html: Optional HTML string for unmasked (real) sentences.
        scale: Colour scale used ("absolute" or "relative").
        negative: Whether negative values are coloured.
        order_by: Ordering method (e.g., "importance", "" for sequential).

    Returns:
        Complete HTML document as a string.
    """
    # --- Summary Statistics ---
    sentence_df = llr_table.drop_duplicates("sentence_id")
    sentence_stats = {
        "sentence_lambdaG_mean": float(sentence_df["sentence_lambdaG"].mean()),
        "sentence_lambdaG_std": float(sentence_df["sentence_lambdaG"].std()),
        "sentence_lambdaG_min": float(sentence_df["sentence_lambdaG"].min()),
        "sentence_lambdaG_max": float(sentence_df["sentence_lambdaG"].max()),
    }

    n_sentences = llr_table["sentence_id"].nunique()
    n_tokens = len(llr_table)
    n_positive = (llr_table["lambdaG"] > 0).sum()
    n_negative = (llr_table["lambdaG"] < 0).sum()
    mean_lambdaG = llr_table["lambdaG"].mean()
    std_lambdaG = llr_table["lambdaG"].std()

    # --- Color scale legend ---
    if scale == "absolute":
        legend_items = [
            ("#943126", "λ ≥ 4"),
            ("#cb4335", "3 ≤ λ < 4"),
            ("#ec7063", "2 ≤ λ < 3"),
            ("#f5b7b1", "1 ≤ λ < 2"),
            ("#fdedec", "0.5 ≤ λ < 1"),
            ("#ffffff", "0 < λ < 0.5"),
        ]
        if negative:
            legend_items += [
                ("#d1e5f0", "-1 < λ ≤ -0.5"),
                ("#92c5de", "-2 < λ ≤ -1"),
                ("#4393c3", "-3 < λ ≤ -2"),
                ("#2166ac", "-4 < λ ≤ -3"),
                ("#053061", "λ ≤ -4"),
            ]
    else:  # relative
        legend_items = [
            ("#E74C3C", "z ≥ 2"),
            ("#F1948A", "1 ≤ z < 2"),
            ("#FADBD8", "0.5 ≤ z < 1"),
            ("#ffffff", "0 ≤ z < 0.5"),
        ]
        if negative:
            legend_items += [
                ("#d1e5f0", "-0.5 < z ≤ 0"),
                ("#67a9cf", "-1 < z ≤ -0.5"),
                ("#2166ac", "z ≤ -1"),
            ]

    legend_html = "".join(
        '<div style="display:inline-block;margin-right:15px;">'
        '<span style="background-color:'
        + color
        + ';padding:2px 8px;border-radius:3px;">&nbsp;</span> '
        '<span style="font-size:12px;color:#666;">' + label + "</span>"
        "</div>"
        for color, label in legend_items
    )

    # --- Build HTML ---
    has_unmasked = unmasked_html is not None

    # Convert order_by to display text
    if order_by == "importance":
        order_display = ", Order=Importance"
    else:
        order_display = ", Order=Sequential"

    # CSS with escaped braces for f-string
    css = """
        body {
            font-family: 'Roboto Condensed', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-weight: 300;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .summary-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .summary-container h2 {
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
            font-weight: 400;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-box {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .stat-box h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            font-weight: 400;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #4CAF50;
        }
        .stat-detail {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .color-scale {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            flex-wrap: wrap;
        }
        .color-scale span {
            font-size: 12px;
            color: #666;
        }
        .content-container {
            display: block;
            margin-bottom: 30px;
        }
        .column {
            flex: 1;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        .column h2 {
            text-align: center;
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
            font-weight: 400;
        }
        .sentence {
            margin-bottom: 12px;
            line-height: 1.6;
            font-size: 15px;
        }
        .sentence-number {
            display: inline-block;
            min-width: 30px;
            color: #000;
            font-size: 13px;
            font-weight: 400;
        }
        .sentence-number-circle {
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            border-radius: 50%;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
            color: #000;
        }
        .sentence-row {
            display: flex;
            align-items: flex-start;
            margin-bottom: 12px;
            gap: 10px;
        }
        .sentence-column {
            flex: 1;
            line-height: 1.6;
            font-size: 15px;
        }
        .heading-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 10px;
        }
        .heading-spacer {
            width: 38px;
            flex-shrink: 0;
        }
        .column-heading {
            flex: 1;
            text-align: center;
            color: #555;
            font-size: 1.5em;
            font-weight: 400;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LambdaG Visualization</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@300;400;700&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <h1>LambdaG Visualization</h1>
    <p class="subtitle">Token-level LambdaG scores with {scale} scale</p>
    
    <div class="summary-container">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <h3>Number of Sentences</h3>
                <div class="stat-value">{n_sentences}</div>
            </div>
            <div class="stat-box">
                <h3>Number of Tokens</h3>
                <div class="stat-value">{n_tokens}</div>
            </div>
            <div class="stat-box">
                <h3>Mean Sentence λG</h3>
                <div class="stat-value">{sentence_stats["sentence_lambdaG_mean"]:.3f}</div>
                <div class="stat-detail">Std: {sentence_stats["sentence_lambdaG_std"]:.3f}</div>
            </div>
            <div class="stat-box">
                <h3>Mean Token λG</h3>
                <div class="stat-value">{mean_lambdaG:.3f}</div>
                <div class="stat-detail">Std: {std_lambdaG:.3f}</div>
            </div>
            <div class="stat-box">
                <h3>Positive Tokens</h3>
                <div class="stat-value" style="color: #E74C3C;">{n_positive}</div>
                <div class="stat-detail">{n_positive/n_tokens*100:.1f}% of total</div>
            </div>
            <div class="stat-box">
                <h3>Negative Tokens</h3>
                <div class="stat-value" style="color: #2166ac;">{n_negative}</div>
                <div class="stat-detail">{n_negative/n_tokens*100:.1f}% of total</div>
            </div>
        </div>
        
        <div class="color-scale">
            <span><strong>Color Scale:</strong></span>
            {legend_html}
        </div>
    </div>
    
    <div class="content-container">
        <div class="heading-row">
            <div class="heading-spacer"></div>
            <div class="column-heading">Masked Sentences (POS Tags{order_display})</div>
            {'<div class="column-heading">Unmasked Sentences (Original Text)</div>' if has_unmasked else ''}
        </div>
        {format_paired_sentences_for_display(masked_html, unmasked_html, sentence_scores)}
    </div>
</body>
</html>
"""

    return html


def format_sentences_for_display(html_text: str) -> str:
    """Format HTML text with sentence numbers and proper spacing.

    Splits the HTML by <br> tags and wraps each sentence in a div.
    """
    # Split by <br> tags (with or without newline)
    import re

    parts = re.split(r"<br>\s*", html_text)
    result = []
    for i, part in enumerate(parts, 1):
        if part.strip():
            result.append(
                f'<div class="sentence">'
                f'<span class="sentence-number">{i}</span> '
                f"{part}"
                f"</div>"
            )
    return "\n".join(result)


def format_paired_sentences_for_display(
    masked_html: str,
    unmasked_html: str | None = None,
    sentence_scores: list[float] | None = None,
) -> str:
    """Format masked and unmasked sentences in paired rows for vertical alignment.

    Splits both HTML strings by <br> tags and pairs them in flexbox rows.
    If unmasked_html is None, only shows masked sentences.
    If sentence_scores is provided, colors the sentence numbers based on z-score.
    """
    import re
    import numpy as np

    masked_parts = re.split(r"<br>\s*", masked_html or "")
    unmasked_parts = re.split(r"<br>\s*", unmasked_html or "") if unmasked_html else []

    # Compute colors for sentence numbers if scores provided
    sentence_colors = []
    if sentence_scores is not None and len(sentence_scores) > 0:
        scores_array = np.array(sentence_scores)
        mean = scores_array.mean()
        std = scores_array.std()
        if std > 0:
            z_scores = (scores_array - mean) / std
        else:
            z_scores = np.zeros_like(scores_array)

        # Map z-scores to colors (same as token-level relative scale)
        for z in z_scores:
            if z < -2:
                sentence_colors.append("#2166ac; color: white")
            elif z < -1:
                sentence_colors.append("#67a9cf")
            elif z < -0.5:
                sentence_colors.append("#d1e5f0")
            elif z > 2:
                sentence_colors.append("#E74C3C; color: white")
            elif z > 1:
                sentence_colors.append("#F1948A")
            elif z > 0.5:
                sentence_colors.append("#FADBD8")
            else:
                sentence_colors.append("")

    result = []
    for i, masked in enumerate(masked_parts, 1):
        if masked.strip():
            masked_content = masked.strip()

            # Color the sentence number if we have scores
            num_style = ""
            if i <= len(sentence_colors) and sentence_colors[i - 1]:
                num_style = f"background-color: {sentence_colors[i-1]};"

            # Only include unmasked column if unmasked_html was provided
            unmasked_div = ""
            if unmasked_html is not None and i <= len(unmasked_parts):
                unmasked_content = (
                    unmasked_parts[i - 1].strip() if i <= len(unmasked_parts) else ""
                )
                if unmasked_content:
                    unmasked_div = (
                        f'<div class="sentence-column">{unmasked_content}</div>'
                    )

            result.append(
                f'<div class="sentence-row">'
                f'<span class="sentence-number"><span class="sentence-number-circle" style="{num_style}">{i}</span></span>'
                f'<div class="sentence-column">{masked_content}</div>'
                f"{unmasked_div}"
                f"</div>"
            )
    return "\n".join(result)
