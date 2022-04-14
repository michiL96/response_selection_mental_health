import math
import nltk
import spacy
import numpy as np


class BM25F:
    def __init__(self, k1: float = 1.5, b: float = 0.75, spacy_model: str = 'it_core_news_lg'):
        self.nlp = spacy.load(spacy_model)
        self.b = b
        self.k1 = k1
        self.boost = {
            'NOUN': 1.0,
            'VERB': 1.0
        }
        self.b_c = {
            'NOUN': 0.75,
            'VERB': 0.75
        }
        self.default_boost = 1.0
        self.default_b_c = 0.75

    def fit(self, corpus: []):
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term.text, 0) + 1
                frequencies[term.text] = term_count
            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def _score(self, query, tf):
        score = 0.0
        for term in query:
            if term in tf:
                score += self.idf_[term] * tf[term]/(self.k1+tf[term])
        return score

    def get_scores_pool(self, query, pool):
        full_occurs_c = []
        len_c = {}
        for document in pool:
            occurs_c = {}
            for token in document:
                if token.text in self.idf_:
                    if token.pos_ not in occurs_c:
                        occurs_c[token.pos_] = {}
                    occurs_c[token.pos_][token.text] = 1 if token.text not in occurs_c[token.pos_] else occurs_c[token.pos_][token.text]+1
            full_occurs_c.append(occurs_c)

            for field in occurs_c:
                len_field = sum(occurs_c[field].values())
                if field not in len_c:
                    len_c[field] = {
                        'count_len': 0,
                        'count_doc': 0
                    }
                len_c[field]['count_len'] += len_field
                len_c[field]['count_doc'] += 1

        pool_tf = []
        for doc in full_occurs_c:
            curr_tf_c = {}
            curr_tf = {}
            for field in doc:
                curr_tf_c[field] = {}
                l_d_c = sum(doc[field].values())
                for term in doc[field]:
                    occ = doc[field][term]
                    b_c = self.b_c[field] if field in self.b_c else self.default_b_c
                    l_c = len_c[field]['count_len']/len_c[field]['count_doc']

                    curr_tf_c[field][term] = occ / ((1-b_c) + b_c * (l_d_c/l_c))

                    boost = self.boost[field] if field in self.boost else self.default_boost
                    c_tf = boost * curr_tf_c[field][term]

                    curr_tf[term] = c_tf if term not in curr_tf else curr_tf[term]+c_tf

            pool_tf.append(curr_tf)

        scores = []
        for index in range(len(pool)):
            scores.append(self._score(query, pool_tf[index]))

        result = np.asarray(scores)
        sorted_indexes = np.argsort(result, axis=0)[::-1]
        return sorted_indexes

    def predict(self, query, pool):
        ita_stopwords = nltk.corpus.stopwords.words('italian')

        query = [token.text for token in self.nlp(query) if token.text not in ita_stopwords]
        pool = [
            [token for token in self.nlp(sent) if token.text not in ita_stopwords] for sent in pool
        ]

        return self.get_scores_pool(query, pool)

    def format_corpus(self, corpus):
        ita_stopwords = nltk.corpus.stopwords.words('italian')

        texts = [
            [token for token in self.nlp(document) if token.text not in ita_stopwords]
            for document in corpus
        ]

        word_count_dict = {}
        for text in texts:
            for token in text:
                word_count = word_count_dict.get(token.text, 0) + 1
                word_count_dict[token.text] = word_count

        texts = [[token for token in text if word_count_dict[token.text] > 1] for text in texts]

        return texts

    def train(self, corpus_histories, corpus_responses):
        corpus = corpus_histories + corpus_responses
        texts = self.format_corpus(corpus)
        self.fit(texts)
