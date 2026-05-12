# LambdaG - Grammar as a behavioral biometric: Using cognitively motivated grammar models for authorship verification

This is the official repository for the paper "[Grammar as a behavioral biometric: Using cognitively motivated grammar models for authorship verification](https://doi.org/10.1057/s41599-025-06340-3)". The paper proposes an authorship verification (AV) method - called **LambdaG** - which seeks to answer the question of whether two given documents are written by the same author, or not. In contrast to existing AV methods which often suffer from high complexity, low explainability and especially from a lack of clear scientific justification, LambdaG represents a simpler method based on modeling the grammar of an author following Cognitive Linguistics principles.

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

A text heatmap with colour-coded tokens depending on the $\lambda_G$ value can be made using the `lambdaG_visualize()` function (see [example](https://github.com/andreanini/lambdag/blob/main/examples/example_visualization.ipynb)).

For further examples see [examples](https://github.com/andreanini/lambdag/tree/main/examples).

Please keep in mind that, for best results, LambdaG should be used on texts that have been preprocessed with **POSNoise** ([paper](https://arxiv.org/abs/2005.06605), [original implementation](https://github.com/Halvani/POSNoise)).

### R implementation
An alternative implementation in R is provided in the [`idiolect`](https://andreanini.github.io/idiolect/articles/idiolect.html) package.

## Data
We will publish the data used in our paper soon.

## Citation
If you use our code in your work, please consider citing:
```
@article{nini2025grammarbehavioralbiometricusing,
    title={Grammar as a Behavioral Biometric: {{Using}} Cognitively Motivated Grammar Models for Authorship Verification}, 
    author={Nini, Andrea and Halvani, Oren and Graner, Lukas and Titze, Sophie and Gherardi, Valerio and Ishihara, Shunichi},
    year={2026},
    journal = {Humanities and Social Sciences Communications},
    volume = {13},
    number = {1},
    pages = {455},
    issn = {2662-9992},
    doi = {10.1057/s41599-025-06340-3},
}

```
