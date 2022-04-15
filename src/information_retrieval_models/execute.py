import os
import yaml
from train_test import train_predictors, load_test_set, make_prediction, recall_at_k

config_filepath = os.path.join('.', 'config.yaml')
with open(config_filepath) as file:
    tmp = yaml.safe_load_all(file)
    for t in tmp:
        config = t

training_filepath = config['General']['training_set_filepath']
model = train_predictors(training_filepath, config['General']['model_type'], k1=config['BM25']['k1'],
                         b=config['BM25']['b'], spacy_model=config['BM25']['spacy_model'])

if model is not None:
    test_filepath = config['General']['test_set_pool_size2_filepath']
    test_set = load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    print('1 in 2 => R@1 = ' + str(r1))

    test_filepath = config['General']['test_set_pool_size10_filepath']
    test_set = load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    r2 = round(recall_at_k(y_predicted, y, 2), 4)
    r3 = round(recall_at_k(y_predicted, y, 3), 4)
    r5 = round(recall_at_k(y_predicted, y, 5), 4)
    print('1 in 10 => R@1 = ' + str(r1) + '; R@2 = ' + str(r2) + '; R@3 = ' + str(r3) + '; R@5 = ' + str(r5))

    test_filepath = config['General']['test_set_pool_size50_filepath']
    test_set = load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    r2 = round(recall_at_k(y_predicted, y, 2), 4)
    r3 = round(recall_at_k(y_predicted, y, 3), 4)
    r5 = round(recall_at_k(y_predicted, y, 5), 4)
    print('1 in 50 => R@1 = ' + str(r1) + '; R@2 = ' + str(r2) + '; R@3 = ' + str(r3) + '; R@5 = ' + str(r5))
