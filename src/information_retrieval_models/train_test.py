import pickle
import numpy as np
from BM25 import BM25F
from TFIDF import TFIDF


def train_predictors(training_filepath: str, model_type: str):
    data_train = []
    data = open(training_filepath)
    for line in data:
        data_train.append(line)

    # Load the train set
    contexts = []
    responses = []
    count = 0
    for sample in data_train:
        count += 1
        sample = sample.strip().split('\t')

        # Consider only the correct responses
        if sample[2] == '1':
            contexts.append(sample[0])
            responses.append(sample[1])

    # Train the model selecting the correct model based on configurations
    print(f'Training {model_type} on {len(contexts)} histories')
    if model_type == 'tfidf':
        conversation_predictor = TFIDF()
    elif model_type == 'bm25':
        conversation_predictor = BM25F()
    else:
        return None
    conversation_predictor.train(contexts, responses)
    return conversation_predictor


def load_test_set(test_filepath):
    test_data = []
    test_data_file = open(test_filepath, 'r')
    for line in test_data_file:
        line = line.strip().split('\t')
        test_data.append(line)
    return test_data


def make_prediction(test_data: [], model):
    y_true = np.zeros(len(test_data))
    y_predicted = [model.predict(sample[0], sample[1:]) for sample in test_data]
    return y_true, y_predicted


def recall_at_k(y, y_predicted, k=1):
    correct_count = 0
    for predictions, label in zip(y, y_predicted):
        if label in predictions[:k]:
            correct_count += 1
    return correct_count/len(y)

