from collections import defaultdict
from typing import Iterable, TypeVar

from lambdag.language_models.language_model import LanguageModel, everygrams

T = TypeVar("T")


class SafeDefaultDict(defaultdict):
    def __init__(self, default_factory=None, *args, autofill=False, **kwargs):
        """
        A defaultdict variant, which allows toggling lookups to not update the underlying dictionary.
        Args:
            autofill: If true, insert and return value from factory to dictionary on lookup miss, if false only return the value
        """
        super().__init__(default_factory, *args, **kwargs)
        self.autofill = autofill

    def __missing__(self, key):
        if not self.autofill:
            if self.default_factory is None:
                raise KeyError(key)
            return self.default_factory()
        return super().__missing__(key)


class KneserNeyLanguageModel(LanguageModel[T]):
    def __init__(
        self,
        order: int,
        discount: float = 0.5,
        pad_start_element: T = "<s>",
        pad_end_element: T = "</s>",
        unk_element: T = "<unk>",
        pad_start_count: int | None = None,
        pad_end_count: int = 1,
        special_handling_of_pad_start_element: bool = True,
    ) -> None:
        """
        A language model with 'kneser-ney' smoothing.
        Args:
            order: the maximum size of ngrams that will be considered (a context size is order-1 or lower)
            discount: the discount value (between 0 and 1) of the kneser-ney smoothing
            pad_start_element: a special element (aka. BOS-token), that is added to the start of a sequence while training
            pad_end_element: a special element (aka. EOS-token), that is added to the end of a sequence while training
            unk_element: a special element (aka. OOV-token), that is used in inference when an element is seen, that has not been part of trainings sequences
            pad_start_count: the number of `pad_start_element` which are added to the start of a sequence while training (if None, order-1 is used)
            pad_start_count: the number of `pad_end_element` which are added to the end of a sequence while training
            special_handling_of_pad_start_element: if true, the pad_start_element is handled differently, by excluding it from the vocabulary and forcing its probability to be zero
        """
        assert 1 >= discount >= 0

        super().__init__(
            order=order,
            pad_start_element=pad_start_element,
            pad_end_element=pad_end_element,
            unk_element=unk_element,
            pad_start_count=pad_start_count,
            pad_end_count=pad_end_count,
        )

        self.discount = discount
        self.special_handling_of_pad_start_element = (
            special_handling_of_pad_start_element
        )

        self.voc = set()
        self.voc_size = 0                        # this is V
        self.context_element_count = SafeDefaultDict(lambda: SafeDefaultDict(int)) # this is used for c(gw)
        self.gram_count = SafeDefaultDict(int)   # this is used for c(g)
        self.pre_voc = SafeDefaultDict(set)      # this is used for N1+(•g)
        self.suc_voc = SafeDefaultDict(set)      # this is used for N1+(g•)
        self.pre_suc_voc = SafeDefaultDict(set)  # this is used for N1+(•g•)
        self.total_element_count = 0             # this is 𝚺_w c(w)

    def vocabulary(self) -> Iterable[T]:
        """
        Return an iterable over the model's vocabulary.
        """
        return self.voc

    def fit(self, sequence: Iterable[T], pad_sequence: bool = True) -> None:
        """
        Trains/Fits the language model on a single sequence, which gets padded internally. Note, that the internal model is updated progressively (i.e. the model is not cleared beforehand).
        Args:
            sequence: the sequence to be fitted
            pad_sequence: if true, pad the given sequence
        """
        if pad_sequence:
            sequence = self.pad_sequence(sequence)

        self.context_element_count.autofill = True
        self.gram_count.autofill = True
        self.pre_voc.autofill = True
        self.suc_voc.autofill = True
        self.pre_suc_voc.autofill = True

        for gram in everygrams(sequence, self.order):
            if self.special_handling_of_pad_start_element:
                if gram[-1] == self.pad_start_element:
                    # Do not include the BOS in the vocabulary
                    continue

            self.gram_count[gram] += 1
            self.context_element_count[gram[:-1]][gram[-1]] += 1
            self.pre_voc[gram[1:]].add(gram[0])
            self.suc_voc[gram[:-1]].add(gram[-1])
            if len(gram) >= 2:
                self.pre_suc_voc[gram[1:-1]].add((gram[0], gram[-1]))
            if len(gram) == 1:
                self.voc.add(gram[0])
                self.total_element_count += 1
        self.voc.add(self.unk_element)
        self.voc_size = len(self.voc)

        if self.special_handling_of_pad_start_element:
            # Add +1 counts to <s>, <s><s>, etc. counters
            g = (self.pad_start_element,)
            for _ in range(self.order):
                self.gram_count[g] += 1
                g += (self.pad_start_element,)
        
        self.context_element_count.autofill = False
        self.gram_count.autofill = False
        self.pre_voc.autofill = False
        self.suc_voc.autofill = False
        self.pre_suc_voc.autofill = False

    def probability(self, element: T, context: Iterable[T] = ()) -> float:
        """
        Calculates and returns the probability of the given element to follow the given context according to the model.
        Args:
            element: a single element (such as a token)
            context: a sequence of elements (such as a token sequence)
        """
        if (
            self.special_handling_of_pad_start_element
            and element == self.pad_start_element
        ):
            return 0.0
        if not isinstance(context, tuple):
            context = tuple(context)
        if len(context) >= self.order:
            context = context[
                -self.order + 1 :
            ]  # trim the context if it is longer than order-1. Is this correct?

        def prob_star(element, context):
            if len(context) >= 1 and self.gram_count[context] == 0:
                return prob_star(element, context[1:])

            p_disc = max(0, len(self.pre_voc[context + (element,)]) - self.discount) / len(self.pre_suc_voc[context])
            interp = (self.discount * len(self.suc_voc[context])) / len(self.pre_suc_voc[context])

            if len(context) == 0:
                p_interp = 1.0 / self.voc_size
            else:
                p_interp = prob_star(element, context[1:])

            return p_disc + interp * p_interp

        if len(context) >= 1 and self.gram_count[context] == 0:
            return prob_star(element, context[1:])
        elif len(context) == 0:
            return (
                max(0, self.gram_count[(element,)] - self.discount)
                / self.total_element_count
                + (self.discount * len(self.suc_voc[()]))
                / self.total_element_count
                / self.voc_size
            )
        else:
            return (
                max(0, self.gram_count[context + (element,)] - self.discount) 
                / self.gram_count[context] 
                + (self.discount * len(self.suc_voc[context])) 
                / self.gram_count[context] * prob_star(element, context[1:])
            )
