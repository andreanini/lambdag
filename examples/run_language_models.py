from lambdag.language_models import KneserNeyLanguageModel

if __name__ == "__main__":  # Example usage
    corpus = """The cat sat on the mat and looked at the window .
The dog barked loudly at the mailman near the gate .
The bird flew over the tree and landed on the branch .
The child played with the ball in the garden .
The cat chased the mouse through the kitchen and under the table .
The dog ran across the yard and jumped over the fence .
The bird sang in the morning when the sun was rising .
The child laughed and called for the dog to come closer .
The cat slept in the sunlight and dreamed of fish .
The dog waited patiently for the child to throw the ball .
The mouse found some cheese and carried it into a hole .
The child sat under the tree and read a small book .
The bird spread its wings and flew into the blue sky .
The cat climbed onto the chair and watched the dog .
The dog barked at the shadow and wagged its tail ."""
    corpus_sequences = [sentence.split(" ") for sentence in corpus.splitlines()]

    print(f"Corpus text:\n\n{corpus}\n")

    lms = [
        KneserNeyLanguageModel[str](
            order=4, discount=0.5, special_handling_of_pad_start_element=True
        )
    ]

    for lm in lms:
        print(f"{'-' * 30}\n{type(lm).__name__}\n{'-' * 30}")

        for sequence in corpus_sequences:
            lm.fit(sequence, pad_sequence=True)

        print("\nProbabilities:")

        print("P(cat)     =", lm.probability("cat"))
        print("P(chair)   =", lm.probability("chair"))
        print("P(asdf)    =", lm.probability("asdf"))
        print("P(<unk>)   =", lm.probability("<unk>"))
        print("P(<s>)     =", lm.probability("<s>"))
        print("P(</s>)    =", lm.probability("</s>"))
        print()
        print("P(cat|The) =", lm.probability("cat", ("The",)))
        print("P(dog|The) =", lm.probability("dog", ("The",)))
        print("P(at|The)  =", lm.probability("dog", ("at",)))
        print("P(</s>|.)  =", lm.probability("</s>", (".",)))

        print("\nPerplexities:")
        for test_sentence in [
            "The cat sat on the mat and looked at the window .",
            "The cat looked at the mat and under the cat sky .",
            "window The looked the at . . chair and wagged sat",
        ]:
            print(
                f"PPL('{test_sentence}') = {lm.perplexity(test_sentence.split(), pad_sequence=False):.3f}"
            )

        print("\nGenerated sequences:")
        for _ in range(5):
            print(f"- '{' '.join(lm.generate(50))}'")
