# LambdaG - Grammar as a behavioral biometric: Using cognitively motivated grammar models for authorship verification

This is the official repository for the paper "[Grammar as a behavioral biometric: Using cognitively motivated grammar models for authorship verification](https://arxiv.org/abs/2403.08462)". The paper proposes an authorship verification (AV) method - called **LambdaG** - which seeks to answer the question of whether two given documents are written by the same author, or not. In contrast to existing AV methods which often suffer from high complexity, low explainability and especially from a lack of clear scientific justification, LambdaG represents a simpler method based on modeling the grammar of an author following Cognitive Linguistics principles.

Given two documents, 𝒟<sub>𝒜</sub> and 𝒟<sub>𝒰</sub> as well as some reference documents $𝔻_{\text{ref}}$, the ratio of the likelihoods of 𝒟<sub>𝒰</sub> given 𝒜's grammar versus the grammar of a reference population’s grammar is calculated. Based on this ratio a final prediction of whether 𝒜=𝒰 holds is made.

## Usage

### Python implementation

To use LambdaG with Python install the LambdaG package from PyPI:
```
pip install lambdag
```

or alternatively directly from this repository:

```
pip install git+https://github.com/AndreaNini/LambdaG
```

Afterwards you can use LambdaG as follows, for example:

```python
from sklearn.metrics import accuracy_score, roc_auc_score
from lambdag.corpus import load_corpus
from lambdag import LambdaGMethod

# load train and test corpus
train_problems, train_labels, train_author_texts = load_corpus("corpus/path/train")
test_problems, test_labels, test_author_texts = load_corpus("corpus/path/test")

# instantiate method object
method = LambdaGMethod(basis="tokens", order=8)

# train method
method.fit(train_problems, train_author_texts, train_labels)

# evaluate on test corpus
test_probas = method.predict_proba(test_problems, test_author_texts)

print(f"Accuracy: {accuracy_score(test_labels, test_probas[:,1]>=0.5):.3f}")
```

For further examples see the [examples](examples) folder.

Please keep in mind that, for best results, LambdaG should be used on texts that have been preprocessed with [POSNoise](https://arxiv.org/abs/2005.06605), (implemented [here](https://github.com/Halvani/POSNoise) for example).

### R implementation
An alternative implementation in R is provided in the [`idiolect`](https://andreanini.github.io/idiolect/articles/idiolect.html) package.

## Data
We will publish the data used in our paper soon.

## Citation
If you use our code in your work, please consider citing:
```
@misc{nini2025grammarbehavioralbiometricusing,
    title={Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models for Authorship Verification}, 
    author={Andrea Nini and Oren Halvani and Lukas Graner and Valerio Gherardi and Shunichi Ishihara},
    year={2025},
    eprint={2403.08462},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2403.08462}, 
}
```
