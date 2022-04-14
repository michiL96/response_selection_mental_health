import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def train(self, histories, responses):
        self._vectorizer.fit(np.append(histories, responses))

    def predict(self, history, responses):
        history_v = self._vectorizer.transform([history])
        responses_v = self._vectorizer.transform(responses)
        result = np.dot(responses_v, history_v.T).todense()
        result = np.asarray(result).flatten()
        sorted_indexes = np.argsort(result, axis=0)[::-1]
        return sorted_indexes

    def predict_co(self, history, responses):
        history_v = self._vectorizer.transform([history])
        responses_v = self._vectorizer.transform(responses)
        result = np.dot(responses_v, history_v.T).todense()
        result = np.asarray(result).flatten()
        return result

