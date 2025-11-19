import re
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from lambdag.language_models.kneser_ney import KneserNeyLanguageModel, LanguageModel


class LambdaGMethod:
    def __init__(
        self,
        basis: str,
        order: int,
        smoothing: str,
        lowercasing: bool,
        sentenize: bool,
        num_references: int,
        random_seed: any = None,
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
        self.basis = basis
        self.order = order
        self.smoothing = smoothing
        self.lowercasing = lowercasing
        self.sentenize = sentenize
        self.num_references = num_references
        self.random_gen = np.random.default_rng(seed=random_seed)

    def fit_new_language_model(
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

    def preprocess_text(self, text: str) -> Iterable[Iterable[str]]:
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
            return sentences

    def lambdag_score(
        self,
        known_sentences: Iterable[Iterable[str]],
        unknown_sentences: Iterable[Iterable[str]],
        reference_sentences: list[Iterable[str]],
    ) -> float:
        """
        Performs the LambdaG method for a single AV problem and returns the lambda score.
        Args:
            known_sentences: The sentences of the known document
            unknown_sentences: The sentences of the unknown document
            reference_sentences: The sentences used as references (should not contain any sentences written by known and unknown)
        """
        known_lm = self.fit_new_language_model(known_sentences)
        llrs = []
        for _ in range(self.num_references):
            sampled_reference_sentences = [
                reference_sentences[i]
                for i in self.random_gen.choice(
                    len(reference_sentences), size=len(known_sentences), replace=False
                )
            ]
            reference_lm = self.fit_new_language_model(sampled_reference_sentences)

            llr = 0
            for sentence in unknown_sentences:
                known_probs = known_lm.probabilities(sentence)
                reference_probs = reference_lm.probabilities(sentence)
                llr += np.sum(np.log2(known_probs))
                llr -= np.sum(np.log2(reference_probs))
            llrs.append(llr)
        return np.mean(llrs)

    def lambdag(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
    ) -> np.ndarray:
        """
        Performs the LambdaG method for a full corpus of AV problems and returns the lambda scores.
        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
        """
        av_problems = [
            (
                self.preprocess_text(known_text),
                known_author,
                self.preprocess_text(unknown_text),
                unknown_author,
            )
            for known_text, known_author, unknown_text, unknown_author in av_problems
        ]
        author_sentences = {
            name: [sent for text in texts for sent in self.preprocess_text(text)]
            for name, texts in author_texts.items()
        }

        scores = []
        for known_sentences, known_author, unknown_sentences, unknown_author in tqdm(
            av_problems
        ):
            reference_sentences = [
                sent
                for name, sents in author_sentences.items()
                if name not in [known_author, unknown_author]
                for sent in sents
            ]
            score = self.lambdag_score(
                known_sentences, unknown_sentences, reference_sentences
            )
            scores.append(score)

        return np.array(scores)

    def fit(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        labels: Iterable[int],
    ) -> None:
        """
        Calculates lambdag scores for the given AV problems and fits a logistic regression model using the labels.

        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
            labels: The labels of the AV problems, where 0 and 1 represents different and same authorship, respectively
        """
        scores = self.lambdag(av_problems, author_texts)
        self.calibration_model = LogisticRegression(class_weight="balanced")
        self.calibration_model.fit(scores[:, None], labels)

    def predict_proba(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        return_scores: bool = False,
    ) -> np.ndarray:
        """
        Calculates lambdag scores and calibrated probabilites (using the fitted logistic regression model)
        for the given AV problems. Returns the probabilities and optionally scores, too.

        Args:
            av_problems: A list of AV problems which are tuples of (known_text, known_author, unknown_text, unknown_author)
            author_texts: A dictionary mapping author names to lists of their corresponding texts
            return_scores: If true, return a tuple of probabilities and lambdag scores
        """
        if not self.calibration_model:
            raise Exception("LambdaGMethod not yet fitted.")
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