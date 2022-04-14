import torch
import random
import numpy as np
import torch.nn.functional as F
from vocabulary import CustomVocabulary
from utilities import text_to_word_sequence


def words_to_indexes(sequence: str, words_dict: dict, oov_token: int = 1):
    indexes = []
    for word in sequence:
        if word in words_dict:
            indexes.append(words_dict[word])
        else:
            indexes.append(oov_token)
    return indexes


def get_sequences_length(sequences: [str], nlp, max_sentence_length: int = 152):
    sequences_length = [min(len(nlp(sequence)), max_sentence_length) for sequence in sequences]
    return sequences_length


def sequences_padding(sequences: [str], vocabulary: CustomVocabulary, nlp, pad_token: int = 0,
                      max_sentence_length: int = 152):
    tens_list = []
    for i in range(len(sequences)):
        seq = words_to_indexes(text_to_word_sequence(sequences[i], nlp), vocabulary.word2index)
        if len(seq) < max_sentence_length:
            seq = seq + [pad_token]*(max_sentence_length-len(seq))
        tens_list.append(seq)
    return tens_list


def process_row(row: []):
    flag, context, response = row
    flag = int(flag)
    return context, response, flag


def core_convertor_test(batch):
    contexts = []
    responses = []
    labels = []
    for context, response, label in batch:
        contexts.append(context)
        responses.append(response)
        labels.append(label)
    return contexts, responses, labels


class SMNTrainValidTest:
    def __init__(self, vocabulary: CustomVocabulary, pad_token: int = 0, oov_token: int = 1, max_sentence_length: int = 151):
        self.vocabulary = vocabulary
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.max_sentence_length = max_sentence_length

    def pad_sentence(self, sentence: str):
        fin_sentence = []
        for token in sentence:
            if token in self.vocabulary.word2index:
                fin_sentence.append(self.vocabulary.word2index[token])
            else:
                fin_sentence.append(self.vocabulary.oov_token)

        if len(fin_sentence) < self.max_sentence_length:
            fin_sentence = fin_sentence + [self.vocabulary.pad_token]*(self.max_sentence_length-len(fin_sentence))
        return fin_sentence

    def pad_history(self, history: [str]):
        fin_history = []
        for h in history:
            fin_h = self.pad_sentence(h)
            fin_history.append(fin_h)
        return fin_history

    def core_convertor(self, batch):
        contexts = []
        responses = []
        labels = []
        for context, response, label in batch:
            contexts.append(self.pad_history(context))
            responses.append(self.pad_sentence(response))
            labels.append(label)
        return contexts, responses, labels

    def training_phase(self, model, train_set, loss_function, optimizer, config):
        num_batches = int(len(train_set) / config.batch_size)

        model.train()
        random.shuffle(train_set)
        tmp_train = list(map(process_row, train_set))
        train_histories, train_responses, train_labels = self.core_convertor(tmp_train)

        train_loss = []
        for num_batch in range(num_batches):
            start_index = num_batch * config.batch_size
            end_index = start_index + config.batch_size

            history = train_histories[start_index:end_index]
            responses = train_responses[start_index:end_index]
            y_true = train_labels[start_index:end_index]

            history = torch.LongTensor(history).cuda()
            responses = torch.LongTensor(responses).cuda()
            y_true = torch.LongTensor(y_true).cuda()

            y_pred = model(history, responses)
            loss = loss_function(y_pred, y_true)
            train_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

            optimizer.step()
        return train_loss

    def validation_phase(self, model, valid_set, loss_function, config):
        num_dev_batches = int(len(valid_set) / config.batch_size)

        model.eval()
        random.shuffle(valid_set)
        tmp_valid = list(map(process_row, valid_set))
        dev_histories, dev_responses, dev_labels = self.core_convertor(tmp_valid)

        dev_loss = []
        for num_batch in range(num_dev_batches):
            start_index = num_batch * config.batch_size
            end_index = start_index + config.batch_size

            history_valid = dev_histories[start_index:end_index]
            responses_valid = dev_responses[start_index:end_index]
            y_true_valid = dev_labels[start_index:end_index]

            history_valid = torch.LongTensor(history_valid).cuda()
            responses_valid = torch.LongTensor(responses_valid).cuda()
            y_true_valid = torch.LongTensor(y_true_valid).cuda()

            y_predicted_valid = model(history_valid, responses_valid)
            loss_valid = loss_function(y_predicted_valid, y_true_valid)
            dev_loss.append(loss_valid.item())
        return dev_loss

    def test_phase(self, model, test_set, loss_function):
        tmp_test = list(map(process_row, test_set))
        test_histories, test_responses, test_labels = self.core_convertor(tmp_test)

        num_test_batches = int(len(test_responses) / 8)

        model.eval()
        print('Testing on Test set')
        test_loss = []
        all_candidate_scores = []
        for num_batch in range(num_test_batches):
            start_index = num_batch * 8
            end_index = start_index + 8

            history = test_histories[start_index:end_index]
            responses = test_responses[start_index:end_index]
            y_true = test_labels[start_index:end_index]

            history = torch.LongTensor(history).cuda()
            responses = torch.LongTensor(responses).cuda()
            y_true = torch.LongTensor(y_true).cuda()

            y_predicted = model(history, responses)

            loss = loss_function(y_predicted, y_true)
            test_loss.append(loss.item())

            candidate_scores = F.softmax(y_predicted, 0).cpu().detach().numpy()
            all_candidate_scores.append(candidate_scores[:, 1])

        all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
        return all_candidate_scores, test_labels, round(sum(test_loss)/len(test_loss), 4)

