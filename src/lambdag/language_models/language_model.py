import math
from abc import ABC, abstractmethod
from collections import deque
from itertools import chain, islice, repeat
from typing import Generic, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


def ngrams(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """
    Yields all n-grams of a sequence
    """
    it = iter(iterable)
    window = deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


def everygrams(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """
    Yields all 1-, 2-, ..., n-grams of a sequence
    """
    window = deque(maxlen=n)
    for x in iterable:
        window.append(x)
        t = tuple(window)
        for i in range(n):
            if i < len(t):
                yield t[i:]


class LanguageModel(ABC, Generic[T]):
    """
    Generic base class for n-gram style language models.
    """

    def __init__(
        self,
        order: int,
        pad_start_element: T = "<s>", # BOS
        pad_end_element: T = "</s>",  # EOS
        unk_element: T = "<unk>",
        pad_start_count: int | None = None,
        pad_end_count: int = 1,
    ) -> None:
        assert order > 0
        assert pad_start_count is None or pad_start_count >= 0
        assert pad_end_count >= 0

        self.order = order
        self.pad_start_element = pad_start_element
        self.pad_end_element = pad_end_element
        self.unk_element = unk_element
        self.pad_start_count = order - 1 if pad_start_count is None else pad_start_count
        self.pad_end_count = pad_end_count

    def pad_sequence(self, sequence: Iterable[T]) -> Iterator[T]:
        """
        Pads a sequence according to the settings.
        Args:
            sequence: the sequence to be padded
        """
        return chain(
            repeat(self.pad_start_element, self.pad_start_count),
            sequence,
            repeat(self.pad_end_element, self.pad_end_count),
        )

    @abstractmethod
    def fit(self, sequence: Iterable[T], pad_sequence: bool = True) -> None:
        """
        Trains/Fits the language model on a single sequence.
        Args:
            sequence: the sequence to be fitted
            pad_sequence: if true, pad the given sequence
        """
        raise NotImplementedError

    @abstractmethod
    def probability(self, element: T, context: Iterable[T] = ()) -> float:
        """
        Calculates and returns the probability of the given element to follow the given context according to the model.
        Args:
            element: a single element (such as a token)
            context: a sequence of elements (such as a token sequence)
        """
        raise NotImplementedError

    @abstractmethod
    def vocabulary(self) -> Iterable[T]:
        """
        Return an iterable over the model's vocabulary.
        """
        raise NotImplementedError

    def probabilities(
        self, sequence: Iterable[T], pad_sequence: bool = True
    ) -> list[float]:
        """
        Calculates and returns the probabilities of the elements in the given sequence according to the model.
        Args:
            sequence: a sequence of elements (such as a token sequence) of which the perplexity is calculated
            pad_sequence: if true, pad the given sequence
        """
        if pad_sequence:
            sequence = self.pad_sequence(sequence)
        return [
            self.probability(gram[-1], gram[:-1])
            for gram in ngrams(sequence, self.order)
        ]

    def perplexity(self, sequence: Iterable[T], pad_sequence: bool = True) -> float:
        """
        Calculates and returns the perplexity of the given sequence according to the model.
        Args:
            sequence: a sequence of elements (such as a token sequence) of which the perplexity is calculated
            pad_sequence: if true, pad the given sequence
        """
        probs = self.probabilities(sequence, pad_sequence=pad_sequence)
        log_probs = [math.log2(prob) for prob in probs]
        return 2 ** (-np.mean(log_probs))

    def generate(
        self,
        max_element_count: int,
        start_sequence: Iterable[T] | None = None,
        return_pad: bool = False,
        force_max_element_count: bool = False
    ) -> tuple[T]:
        """
        Generates a sequence or continues a given start sequence according to the model.
        Args:
            max_element_count: the maximum number of elements to be generated
            start_sequence: a sequence from which the generation process is started (if None, a sequence of multiple pad_start_element is used)
            return_pad: if true, the returned sequence will include pad_start_element and pad_end_element
            force_max_element_count: if true, the generated sequence will always have max_element_count elements
        """
        sequence = (
            (self.pad_start_element,) * self.pad_start_count
            if start_sequence is None
            else tuple(start_sequence)
        )
        vocab = list(self.vocabulary())
        for c in range(max_element_count):
            vocab_probas = [self.probability(v, sequence) for v in vocab]
            if force_max_element_count and c < max_element_count - 1:
                pad_end_index = vocab.index(self.pad_end_element)
                pad_end_proba = vocab_probas[pad_end_index]
                delta = pad_end_proba / (len(vocab) - 1)
                vocab_probas = [
                    0 if i == pad_end_index else vocab_probas[i] + delta
                    for i in range(len(vocab_probas))
                ]
            next_element = np.random.choice(vocab, p=vocab_probas)
            sequence = sequence + (next_element,)
            if next_element == self.pad_end_element:
                break
        if return_pad:
            return sequence
        else:
            return tuple(
                t
                for t in sequence
                if t not in [self.pad_start_element, self.pad_end_element]
            )
