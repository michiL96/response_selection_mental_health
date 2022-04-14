from vocabulary import CustomVocabulary
from utilities import text_to_word_sequence


def check_distractors_length(distractors: [str], max_sentence_length: int = 152):
    """
    Check if all distractors are below the maximum threshold
    :param distractors:
    :param max_sentence_length:
    :return:
    """
    for distractor in distractors:
        if len(distractor) > max_sentence_length:
            return False
    return True


def load_training_set(train_filepath: str, nlp, pad_token: int = 0, oov_token: int = 1, max_sentence_length: int = 151):
    """
    Load the training set and obtain the respective vocabulary
    :param train_filepath:
    :param nlp:
    :param pad_token:
    :param oov_token:
    :param max_sentence_length:
    :return:
    """
    vocabulary = CustomVocabulary(nlp, pad_token, oov_token)
    training_set = []
    init_data = 0
    with open(train_filepath, encoding="utf8") as f:
        lines = f.readlines()
    print(f'Reading {train_filepath}')
    init_data += len(lines)

    for line in lines:
        parts = line.strip().split('\t')
        # divide each sequence in a list of words
        history = text_to_word_sequence(parts[0], nlp)
        response = text_to_word_sequence(parts[1], nlp)

        if len(history) <= max_sentence_length and len(response) <= max_sentence_length:
            vocabulary.add_sentence(parts[0])
            vocabulary.add_sentence(parts[1])

            training_set.append([int(parts[2]), [history], response])
    print(f'Read {init_data}. Trimmed to {len(training_set)}. {len(training_set)*100/init_data}')
    return training_set, vocabulary


def load_validation_test_set(filepath: str, nlp, max_sentence_length: int = 151):
    dataset = []
    init_data = 0
    with open(filepath, encoding="utf8") as f:
        lines = f.readlines()
    print(f'Reading {filepath}')
    init_data += len(lines)

    for line in lines:
        parts = line.strip().replace('_', '').split('\t')

        response = text_to_word_sequence(parts[1], nlp)
        history = text_to_word_sequence(parts[0], nlp)
        distractors = [text_to_word_sequence(utt, nlp) for utt in parts[2:]]

        if len(history) <= max_sentence_length and len(response) <= max_sentence_length and \
                check_distractors_length(distractors, max_sentence_length):
            dataset.append([1, [history], response])
            for distractor in distractors:
                dataset.append([0, [history], distractor])

    print(f'Read {init_data*2}. Trimmed to {len(dataset)}. {len(dataset)*100/(init_data*2)}')
    return dataset
