import re
from collections import Counter
from typing import Iterable, Literal, overload

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from lambdag.language_models.kneser_ney import KneserNeyLanguageModel, LanguageModel


class HapaxCalibrationModel:
    """Calibrates LambdaG scores using the Hapax Correction (Barlow et al., 2026).

    Multiplies raw scores by V1(Q)/N(Q), where V1(Q) is the number of hapax
    legomena (types appearing exactly once) and N(Q) is the total token count
    in the unknown document. This ratio is Baayen's P, a measure of
    linguistic productivity that scales down scores from repetitive texts.
    """

    def predict_proba(self, scores: np.ndarray, unknown_sentences_list: list[list[tuple[str, ...] | str]]) -> np.ndarray:
        # N(Q), i.e. total token count per unknown document
        n_q = np.array([sum(len(s) for s in sentences) for sentences in unknown_sentences_list])
        # V1(Q), i.e. number of hapax legomena (types appearing exactly once) per unknown document
        v1_q = np.array([sum(v == 1 for v in Counter(t for s in sentences for t in s).values()) for sentences in unknown_sentences_list])
        corrected = scores * v1_q / n_q
        p1 = 1.0 / (1.0 + np.power(2.0, -corrected))
        return np.column_stack([1.0 - p1, p1])


class SquareRootCalibrationModel:
    """Calibrates LambdaG scores using the Square Root Correction (Barlow et al., 2026).

    Divides raw scores by sqrt(N(Q)), where N(Q) is the total token count in
    the unknown document. This accounts for the diminishing evidential
    value of additional tokens in longer texts.
    """

    def predict_proba(self, scores: np.ndarray, unknown_sentences_list: list[list[tuple[str, ...] | str]]) -> np.ndarray:
        # N(Q), i.e. total token count per unknown document
        n_q = np.array([sum(len(s) for s in sentences) for sentences in unknown_sentences_list])
        corrected = scores / np.sqrt(n_q)
        p1 = 1.0 / (1.0 + np.power(2.0, -corrected))
        return np.column_stack([1.0 - p1, p1])


class LambdaGMethod:
    def __init__(
        self,
        basis: Literal["tokens", "characters"],
        order: int,
        smoothing: Literal["kneser_ney", "kneser_ney_shb_disabled"],
        lowercasing: bool,
        sentenize: bool,
        num_references: int,
        calibration: Literal["logistic_regression", "logistic_regression_no_intercept", "square_root", "hapax"] = "logistic_regression",
        random_seed: int | None = None,
    ):
        """
        The LambdaG authorship verification method.

        Args:
            randorandom_seedm_gen: A seed for numpy random generator (if None, a default generator is used)
            basis: The element type of the sequences (either 'tokens' or 'characters')
            order: The language model order
            smoothing: The smoothing (one of: 'kneser_ney', 'kneser_ney_shb_disabled')
            lowercasing: If true, all texts are lowercased
            sentenize: If true, split texts into sentences, so that language mode contexts can only span inside a single sentence
            num_references: Number of reference sentences that are sampled
            calibration: The calibration method (one of: 'logistic_regression',
                'logistic_regression_no_intercept', 'square_root', 'hapax')
            random_seed: A seed to initialize the internal random generator with
        """
        if basis not in ["tokens", "characters"]:
            raise ValueError(
                "Only 'tokens' or 'characters' allowed for argument 'basis'."
            )
        if smoothing not in ["kneser_ney", "kneser_ney_shb_disabled"]:
            raise ValueError(
                "Only 'kneser_ney' or 'kneser_ney_shb_disabled' allowed for argument 'smoothing'."
            )
        if calibration not in [
            "logistic_regression",
            "logistic_regression_no_intercept",
            "square_root",
            "hapax",
        ]:
            raise ValueError(
                "Calibration must be one of: 'logistic_regression', 'logistic_regression_no_intercept', 'square_root', 'hapax'."
            )

        self.basis = basis
        self.order = order
        self.smoothing = smoothing
        self.lowercasing = lowercasing
        self.sentenize = sentenize
        self.num_references = num_references
        self.calibration = calibration
        self.random_gen = np.random.default_rng(seed=random_seed)

        if calibration == "logistic_regression":
            self.calibration_model = LogisticRegression(class_weight="balanced", fit_intercept=True)
        elif calibration == "logistic_regression_no_intercept":
            self.calibration_model = LogisticRegression(class_weight="balanced", fit_intercept=False)
        elif calibration == "square_root":
            self.calibration_model = SquareRootCalibrationModel()
        elif calibration == "hapax":
            self.calibration_model = HapaxCalibrationModel()

    def _fit_new_language_model(
        self,
        sentences: Iterable[Iterable[str]],
    ) -> LanguageModel:
        """
        Creates a language model, fits it on the sentences and returns it.

        Args:
            sentences: An iterable of sentences, where each sentence is an iterable of tokens
        """
        if self.smoothing == "kneser_ney":
            lm = KneserNeyLanguageModel(
                discount=0.75,
                order=self.order,
                special_handling_of_pad_start_element=True,
            )
        elif self.smoothing == "kneser_ney_shb_disabled":
            lm = KneserNeyLanguageModel(
                discount=0.75,
                order=self.order,
                special_handling_of_pad_start_element=False,
            )

        for sentence in sentences:
            lm.fit(sentence)
        return lm

    def _preprocess_text(self, text: str) -> list[tuple[str, ...] | str]:
        """
        Preprocesses and extracts sentences from the given text, where each sentence represents an iterable of tokens.
        """
        if self.lowercasing:
            text = text.lower()

        if self.sentenize:
            sentences = [
                s[0]
                for s in re.findall(r"(.+? ([.?!\n]+ |$))", text, re.VERBOSE)
                if len(s[0].strip()) > 0
            ]  # extract sentences (somewhat naive, could be more refined)
        else:
            sentences = [text]

        if self.basis == "tokens":
            return [
                tuple(t for t in re.split(r"\s+", sent) if len(t) > 0)
                for sent in sentences
            ]  # extract tokens (somewhat naive, could be more refined)
        elif self.basis == "characters":
            return list(sentences)

    def _lambdag_score(
        self,
        known_sentences: list[tuple[str, ...] | str],
        unknown_sentences: list[tuple[str, ...] | str],
        reference_sentences: list[tuple[str, ...] | str],
    ) -> float:
        """
        Performs the LambdaG method for a single AV problem and returns the lambda score.
        Args:
            known_sentences: The sentences of the known document
            unknown_sentences: The sentences of the unknown document
            reference_sentences: The sentences used as references (should not contain any sentences written by known and unknown)
        """
        known_lm = self._fit_new_language_model(known_sentences)
        llrs = []
        for _ in range(self.num_references):
            sampled_reference_sentences = [
                reference_sentences[i]
                for i in self.random_gen.choice(
                    len(reference_sentences), size=len(known_sentences), replace=False
                )
            ]
            reference_lm = self._fit_new_language_model(sampled_reference_sentences)

            llr = 0
            for sentence in unknown_sentences:
                known_probs = known_lm.probabilities(sentence)
                reference_probs = reference_lm.probabilities(sentence)
                llr += np.sum(np.log10(known_probs))
                llr -= np.sum(np.log10(reference_probs))
            llrs.append(llr)
        return np.mean(llrs)

    @overload
    def lambdag(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_preprocessed: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def lambdag(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_preprocessed: Literal[True],
    ) -> tuple[np.ndarray, list[tuple[list[tuple[str, ...] | str], str, list[tuple[str, ...] | str], str]]]: ...

    def lambdag(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_preprocessed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[tuple[list[tuple[str, ...] | str], str, list[tuple[str, ...] | str], str]]]:
        """
        Performs the LambdaG method for a full corpus of AV problems and returns the lambda scores.
        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
            return_preprocessed: If true, also return the preprocessed problems
        """
        preprocessed = [
            (
                self._preprocess_text(known_text),
                known_author,
                self._preprocess_text(unknown_text),
                unknown_author,
            )
            for known_text, known_author, unknown_text, unknown_author in av_problems
        ]
        author_sentences = {
            name: [sent for text in texts for sent in self._preprocess_text(text)]
            for name, texts in author_texts.items()
        }

        scores = []
        for known_sentences, known_author, unknown_sentences, unknown_author in tqdm(
            preprocessed
        ):
            reference_sentences = [
                sent
                for name, sents in author_sentences.items()
                if name not in [known_author, unknown_author]
                for sent in sents
            ]
            score = self._lambdag_score(
                known_sentences, unknown_sentences, reference_sentences
            )
            scores.append(score)

        scores = np.array(scores)
        if return_preprocessed:
            return scores, preprocessed
        return scores

    def fit(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        labels: Iterable[int],
    ) -> None:
        """
        Calculates lambdag scores for the given AV problems and fits the calibration model using the labels.
        For square_root and hapax corrections, this is a no-op.

        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
            labels: The labels of the AV problems, where 0 and 1 represents different and same authorship, respectively
        """
        if isinstance(self.calibration_model, (HapaxCalibrationModel, SquareRootCalibrationModel)):
            return
        scores = self.lambdag(av_problems, author_texts)
        self.calibration_model.fit(scores[:, None], labels)

    @overload
    def predict_proba(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_scores: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def predict_proba(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_scores: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_proba(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculates lambdag scores and calibrated probabilities for the given AV problems.
        Returns the probabilities and optionally the raw lambdag scores, too.

        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
            return_scores: If true, return a tuple of probabilities and raw lambdag scores
        """
        if isinstance(self.calibration_model, (HapaxCalibrationModel, SquareRootCalibrationModel)):
            scores, preprocessed = self.lambdag(av_problems, author_texts, return_preprocessed=True)
            unknown_sentences_list = [unknown_sentences for _, _, unknown_sentences, _ in preprocessed]
            probas = self.calibration_model.predict_proba(scores, unknown_sentences_list)
        else:
            scores = self.lambdag(av_problems, author_texts)
            probas = self.calibration_model.predict_proba(scores[:, None])

        if return_scores:
            return probas, scores
        else:
            return probas

    def predict(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
    ) -> np.ndarray:
        """
        Calculates lambdag scores and returns predicted classes (0 or 1 for different or same authorship) for the given AV problems.

        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
        """
        probas = self.predict_proba(av_problems, author_texts)
        return (probas[:, 1] >= 0.5).astype(int)
