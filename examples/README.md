# Examples

Some illustrative example scripts.

### run_lambdag.py
Runs LambdaG on a given AV test corpus. When using `logistic_regression` as calibration method, the lambda_g scores are calibrated corresponding to a training dataset.  

Example output (logistic regression):
```
> python run_lambdag.py Enron_Posnoised_Training Enron_Posnoised_Test --num_references 50 --order 4 -c logistic_regression
Problem #0001: | label = 1.0 | lambda_G score =    3.025 | Y-probability = 0.572
Problem #0002: | label = 0.0 | lambda_G score = -237.635 | Y-probability = 0.015
Problem #0003: | label = 1.0 | lambda_G score =  447.686 | Y-probability = 1.000
Problem #0004: | label = 0.0 | lambda_G score = -206.086 | Y-probability = 0.027
Problem #0005: | label = 1.0 | lambda_G score =  377.552 | Y-probability = 0.999
Problem #0006: | label = 0.0 | lambda_G score =  -36.378 | Y-probability = 0.391
Problem #0007: | label = 1.0 | lambda_G score =   60.868 | Y-probability = 0.796
...
Problem #0090: | label = 0.0 | lambda_G score = -262.080 | Y-probability = 0.010
Problem #0091: | label = 1.0 | lambda_G score =   61.387 | Y-probability = 0.797
Problem #0092: | label = 0.0 | lambda_G score = -298.934 | Y-probability = 0.005
Problem #0093: | label = 1.0 | lambda_G score = -251.412 | Y-probability = 0.012
Problem #0094: | label = 0.0 | lambda_G score = -205.743 | Y-probability = 0.027
Problem #0095: | label = 0.0 | lambda_G score =  -39.561 | Y-probability = 0.377
Problem #0096: | label = 1.0 | lambda_G score =  244.625 | Y-probability = 0.992

Accuracy: 0.906
ROC-AUC:  0.941
Cllr:     0.441
```

When using the `hapax` or `square_root` calibration methods, **no** training corpus is needed and the training path argument can be omitted. Hereby calibration exclusively depends on the AV cases of the test dataset themselves.

Example output (hapax, no training corpus needed):
```
> python run_lambdag.py Enron_Posnoised_Test --num_references 30 --order 10 -c hapax
Problem #0001: | label = 1.0 | lambda_G score =    4.276 | Y-probability = 0.567
Problem #0002: | label = 0.0 | lambda_G score =  -73.024 | Y-probability = 0.042
Problem #0003: | label = 1.0 | lambda_G score =  140.036 | Y-probability = 0.998
Problem #0004: | label = 0.0 | lambda_G score =  -58.615 | Y-probability = 0.068
Problem #0005: | label = 1.0 | lambda_G score =  111.795 | Y-probability = 0.993
Problem #0006: | label = 0.0 | lambda_G score =   -8.233 | Y-probability = 0.380
Problem #0007: | label = 1.0 | lambda_G score =   20.539 | Y-probability = 0.772
...
Problem #0090: | label = 0.0 | lambda_G score =  -69.285 | Y-probability = 0.007
Problem #0091: | label = 1.0 | lambda_G score =   16.062 | Y-probability = 0.758
Problem #0092: | label = 0.0 | lambda_G score =  -89.283 | Y-probability = 0.005
Problem #0093: | label = 1.0 | lambda_G score =  -69.904 | Y-probability = 0.016
Problem #0094: | label = 0.0 | lambda_G score =  -60.986 | Y-probability = 0.062
Problem #0095: | label = 0.0 | lambda_G score =  -12.745 | Y-probability = 0.309
Problem #0096: | label = 1.0 | lambda_G score =   77.071 | Y-probability = 0.969

Accuracy: 0.906
ROC-AUC:  0.936
Cllr:     0.428
```

### run_language_models.py
Tests all available language model implementations by fitting them on tokens from a small example corpus, calculating probabilities, perplexities and generate some example outputs. 

Example output:
```
> python run_language_models.py
Corpus text:

The cat sat on the mat and looked at the window .
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
The dog barked at the shadow and wagged its tail .

------------------------------
KneserNeyLanguageModel
------------------------------

Probabilities:
P(cat)     = 0.021021671826625385
P(chair)   = 0.005232198142414861
P(asdf)    = 0.0026006191950464397
P(<unk>)   = 0.0026006191950464397
P(<s>)     = 0.0
P(</s>)    = 0.07891640866873065

P(cat|The) = 0.2345253209197348
P(dog|The) = 0.2357243616871209
P(at|The)  = 0.002391028353787558
P(</s>|.)  = 0.966905064183947

Perplexities:
PPL('The cat sat on the mat and looked at the window .') = 1.240
PPL('The cat looked at the mat and under the cat sky .') = 14.321
PPL('window The looked the at . . chair and wagged sat') = 84.255

Generated sequences:
- 'The dog ran cheese and looked at the shadow and wagged its tail .'
- 'The cat chased the mouse through the kitchen and under the table .'
- 'The bird spread its tail .'
- 'The cat slept in the sunlight and carried it into the tree and read a small book .'
- 'The dog barked and waited patiently for the under read a hole a small book .'
```