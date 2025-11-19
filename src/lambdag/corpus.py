import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm.auto import tqdm


def load_corpus(
    path: str | Path,
    process_strings: Callable[[str], str] | None = None,
    with_progress: bool = False,
) -> tuple[list[tuple[str, str, str, str]], np.ndarray, dict[str, set[str]]]:
    """
    Loads a AV-corpus in (old) PAN-webis format. The given Path must contain problem folders and a truth.txt file which contains lines consisting of a problem folder name and the corresponding label (Y or N).
    The problem folders must contain one or multiple known and unknown files named whose file names contain the author name in []-brackets. Example structure of path:
    - path
      - problem1
        - known_a[author1].txt
        - known_b[author1].txt
        - unknown_a[author2].txt
        - unknown_b[author2].txt
      - problem2
        - known[author3].txt
        - unknown[author3].txt
      - truth.txt

    where truth.txt holds:
    problem1 N
    problem2 Y

    Args:
        path: the path of the corpus to load
        process_strings: A transformation function that will be applied to all texts
        with_progress: if true, loading progress is shown using tqdm

    Returns:
        problems: A list of (known-text, known-author-name, unknown-text, unknown-author-name)-tuples
        labels: An array of 0's (N-Problem, i.e. different authors) and 1's (Y-Problem, i.e. same authors)
        author_texts: A dictionary mapping author names to a set of all their corresponding texts in the corpus
    """
    path = path if isinstance(path, Path) else Path(path)
    truth_dict = {
        line[:-2]: line[-1] == "Y"
        for line in (path / "truth.txt").read_text("utf-8").splitlines()
        if line
    }
    problems = []
    truths = []
    author_texts = defaultdict(set)
    problem_paths = list(path.glob("**/"))[1:]
    if with_progress:
        problem_paths = tqdm(problem_paths, leave=False)
    for problem in problem_paths:
        if problem.name in truth_dict:
            truths.append(truth_dict[problem.name])

            if all(" vs " in p for p in truth_dict.keys()):
                known_author, unknown_author = re.findall(
                    r"^(.+)\svs\s(.+)$", problem.name
                )[0]
            else:
                known_author = re.findall(
                    r"\[(\S+)", list(problem.glob("known*"))[0].name
                )[0]
                unknown_author = re.findall(
                    r"\[(\S+)", list(problem.glob("unknown*"))[0].name
                )[0]

            known_texts = [k.read_text("utf-8") for k in problem.glob("known*")]
            unknown_texts = [u.read_text("utf-8") for u in problem.glob("unknown*")]

            if process_strings:
                known_texts = list(map(process_strings, known_texts))
                unknown_texts = list(map(process_strings, unknown_texts))

            known_text = " ".join(known_texts)
            unknown_text = " ".join(unknown_texts)

            author_texts[known_author] |= set(known_texts)

            problems.append((known_text, known_author, unknown_text, unknown_author))
    return problems, np.array(truths).astype(np.float32), author_texts
