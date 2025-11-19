import argparse

from sklearn.metrics import accuracy_score, roc_auc_score
from lambdag.corpus import load_corpus
from lambdag import LambdaGMethod

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("train", help="Path to the training corpus directory")
    parser.add_argument("test", help="Path to the test corpus directory")
    parser.add_argument("-b", "--basis", default="tokens", choices=["characters", "tokens"], help="The element type of the sequences (either 'characters' or 'tokens')")
    parser.add_argument("-o", "--order", type=int, default=10, help="The language model order")
    parser.add_argument("-s", "--smoothing", default="kneser_ney", help="The smoothing (one of: kneser_ney, kneser_ney_shb_disabled)")
    parser.add_argument("-l", "--lowercasing", type=bool, default=False, help="If true, all texts are lowercased")
    parser.add_argument("-n", "--num_references", type=int, default=100, help="Number of reference sentences that are sampled")
    parser.add_argument("--sentenize", type=bool, default=True, help="If true, split texts into sentences, so that language mode contexts can only span inside a single sentence")

    args = parser.parse_args()

    # load corpora
    train_problems, train_labels, train_author_texts = load_corpus(args.train)
    test_problems, test_labels, test_author_texts = load_corpus(args.test)

    # train LambdaG
    method = LambdaGMethod(
        basis=args.basis, 
        order=args.order, 
        smoothing=args.smoothing, 
        lowercasing=args.lowercasing, 
        sentenize=args.sentenize, 
        num_references=args.num_references
    )
    method.fit(train_problems, train_author_texts, train_labels)

    # evaluate on test corpus
    test_probas, test_scores = method.predict_proba(test_problems, test_author_texts, return_scores=True)

    # print results
    for i, (proba, score, label) in enumerate(
        zip(test_probas, test_scores, test_labels)
    ):
        print(
            f"Problem #{i+1:04}:", 
            f"label = {label}", 
            f"lambda_G score = {score: 8.3f}", 
            f"Y-probability = {proba[1]:.3f}",
            sep=" | "
        )

    print(f"\nAccuracy: {accuracy_score(test_labels, test_probas[:,1]>=0.5):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(test_labels, test_probas[:,1]):.3f}")