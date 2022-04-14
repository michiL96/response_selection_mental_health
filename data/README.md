# Data

## Training set

The **training set** file has the following format:
> `dialogue_context    correct_response    1`   
> `dialogue_context    wrong_response    0`

Each element is divided by a *tabulation*.

An example of training set is available in `train_set.csv`.

## Validation set

The **validation set** file has the following format:
> `dialogue_context    correct_response    distractor`

Each element is divided by a *tabulation*.

An example of validation set is available in `validation_set.csv`.

## Test set

The **test set** file has the following format:
> `dialogue_context    correct_response    distractor_1    distractor_2    ...    distractor_N-1`

The test set contains *N-1* distractors in a candidates' pool size of *N*.

Each element is divided by a *tabulation*.

An example of test set is available in `test_set_pool_size2.csv`.
