class CustomVocabulary:
    def __init__(self, nlp, pad_token: int = 0, oov_token: int = 1):
        self.trimmed = False
        self.nlp = nlp
        self.word2index = {}
        self.word2count = {}
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.index2word = {self.pad_token: "pad", self.oov_token: 'oov'}
        self.num_words = len(self.index2word)  # Count PAD, OOV

    def add_sentence(self, sentence: str):
        sentence = [tok.text for tok in self.nlp.tokenizer(sentence)]
        for word in sentence:
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count: int):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "pad", self.oov_token: 'oov'}
        self.num_words = len(self.index2word)  # Count default tokens

        for word in keep_words:
            self.add_word(word)
