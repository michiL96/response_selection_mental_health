from vocabulary import CustomVocabulary
from utilities import text_to_word_sequence


def filter_pair(pair: [str], nlp, max_sentence_length: int = 152):
    """
    Returns True iff both sentences in a pair are under the maximum sentence length threshold
    :param pair: 
    :param nlp: 
    :param max_sentence_length: 
    :return: 
    """
    return len([tok.text for tok in nlp.tokenizer(pair[0])]) < max_sentence_length and \
           len([tok.text for tok in nlp.tokenizer(pair[1])]) < max_sentence_length


def filter_pairs(pairs: [[str]], nlp, max_sentence_length: int = 152):
    """
    Filter pairs using filter_pair condition
    :param pairs:
    :param nlp:
    :param max_sentence_length:
    :return:
    """
    return [pair for pair in pairs if filter_pair(pair, nlp, max_sentence_length=max_sentence_length)]


def load_training_set(train_filepath: str, nlp, pad_token: int = 0, oov_token: int = 1, max_sentence_length: int = 152):
    """
    Load the training set and build the respective vocabualry
    :param train_filepath: string, filepath of the training set
    :param nlp: Spacy tokenizer
    :param pad_token: index of the padding token
    :param oov_token: index of the out of vocabulary token
    :param max_sentence_length: maximum number of tokens per sentence
    :return: vocabulary of the train set and list of list in the format [[dialogue history, response, label], ...]
    """
    print("Start preparing training data ...")
    vocabulary = CustomVocabulary(nlp, pad_token=pad_token, oov_token=oov_token)
    pairs = []
    lines = open(train_filepath, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    for line in lines:
        pairs.append([s for s in line.split('\t')])

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs, nlp, max_sentence_length=max_sentence_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        vocabulary.add_sentence(pair[0])
        vocabulary.add_sentence(pair[1])

        pair[0] = text_to_word_sequence(pair[0], nlp)
        pair[1] = text_to_word_sequence(pair[1], nlp)
    print("Counted words:", vocabulary.num_words)
    return vocabulary, pairs


def load_validation_set(valid_filepath: str, nlp):
    """
    Load the validation set
    :param valid_filepath: string, path of the validation set
    :param nlp: Spacy tokenizer
    :return: list of list in the format [[dialogue history, response, label], ...]
    """
    dev_set = []
    data = open(valid_filepath, 'r')
    for line in data:
        line = line.strip().split('\t')
        history = text_to_word_sequence(line[0], nlp)
        dev_set.append([history, text_to_word_sequence(line[1], nlp), '1'])
        dev_set.append([history, text_to_word_sequence(line[2], nlp), '0'])
    return dev_set


def load_test_set(test_filepath: str, nlp):
    test_set = []
    data = open(test_filepath, 'r')
    for line in data:
        line = line.strip().split('\t')
        final_line = []
        for sample in line:
            final_line.append(text_to_word_sequence(sample, nlp))
            test_set.append(final_line)
    return test_set
