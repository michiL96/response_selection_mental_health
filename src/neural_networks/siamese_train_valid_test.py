import torch
import random
from siamese_nn import DualEncoder
from torch.autograd import Variable
from vocabulary import CustomVocabulary
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


def get_batch(batch_num: int, batch_size: int, pairs: []):
    start_index = batch_num*batch_size
    return pairs[start_index:start_index+batch_size]


def core_convertor(batch: []):
    """
    Convert the batch in Tensors
    :param batch:
    :return:
    """
    contexts = []
    responses = []
    labels = []
    for context, response, label in batch:
        contexts.append(torch.LongTensor(context))
        responses.append(torch.LongTensor(response))
        labels.append(label)

    contexts = Variable(torch.stack(contexts, 0)).cuda()
    responses = Variable(torch.stack(responses, 0)).cuda()
    labels = torch.FloatTensor(labels).cuda()

    return contexts, responses, labels


def recall_at_k(counts: [int], test_set_len: int, k: int, pool_size: int = 2):
    """
    Calculate Recall at k for the given counts
    :param counts: list that represents how many times the correct response has been selected in that position
    :param test_set_len:
    :param k:
    :param pool_size:
    :return: recall at k metric
    """
    recall = sum(counts[:k])/test_set_len
    print("1 in %0.2f ==> R@1: %0.4f" % (pool_size, recall))
    return recall


class SiameseTrainValidTest:
    def __init__(self, vocabulary: CustomVocabulary, pad_token: int = 0, oov_token: int = 1, max_sentence_length: int = 152):
        self.vocabulary = vocabulary
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.max_sentence_length = max_sentence_length

    def padded_words_to_indexes(self, sentence: str):
        """
        Convert words into their indexes or oov token if the word is not present in the vocabulary and pad the sequence
        :param sentence:
        :return:
        """
        padded_sequence = []
        for word in sentence:
            if self.vocabulary.word2index.get(word) is not None:
                padded_sequence.append(self.vocabulary.word2index[word])
            else:
                padded_sequence.append(self.oov_token)
        if len(padded_sequence) < self.max_sentence_length:
            padded_sequence = [self.pad_token]*(self.max_sentence_length-len(padded_sequence)) + padded_sequence
        return padded_sequence

    def process_row(self, row: []):
        """
        Convert the row into a sequence of padded sentences
        :param row:
        :return:
        """
        context, response, flag = row
        context = self.padded_words_to_indexes(context)
        response = self.padded_words_to_indexes(response)
        flag = int(flag)
        return context, response, flag

    def process_row_test(self, row: []):
        context = row[0]
        response = row[1]
        distractors = row[2:]
        context = self.padded_words_to_indexes(context)
        response = self.padded_words_to_indexes(response)
        distractors = [self.padded_words_to_indexes(distractor) for distractor in distractors]
        return context, response, distractors

    def training_phase(self, model: DualEncoder, training_set: [], batch_size: int, loss_function: _Loss,
                       optimizer: Optimizer, gradient_clipping: int):
        num_batches = int(len(training_set) / batch_size)
        random.shuffle(training_set)

        model.train()   # set train mode
        train_loss = []
        for batch_num in range(num_batches):
            batch = get_batch(batch_num, batch_size, training_set)  # extract the current batch
            batch = list(map(self.process_row, batch))
            contexts, responses, labels = core_convertor(batch)     # convert the current batch in Tensors

            label_predictions, responses = model(contexts, responses)   # feed the model and obtain predictions

            loss = loss_function(label_predictions, labels)     # compute loss
            train_loss.append(loss.tolist())

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)   # apply gradient clipping

            optimizer.step()
        return sum(train_loss)/len(train_loss)

    def validation_phase(self, model: DualEncoder, validation_set: [], batch_size: int, loss_function: _Loss):
        num_dev_batches = int(len(validation_set) / batch_size)
        model.eval()    # set evaluation mode
        print('\t\tEvaluating on Development set')
        loss_validation_list = []
        for dev_batch_num in range(num_dev_batches):
            dev_batch = get_batch(dev_batch_num, batch_size, validation_set)
            dev = list(map(self.process_row, dev_batch))
            contexts, responses, y_true = core_convertor(dev)
            y_predicted, responses_valid = model(contexts, responses)
            loss_validation = loss_function(y_predicted, y_true)
            loss_validation_list.append(round(loss_validation.tolist(), 5))

        return sum(loss_validation_list) / len(loss_validation_list)

    def test_phase(self, model: DualEncoder, test_set: [], pool_size: int = 2):
        test_set = list(map(self.process_row_test, test_set))
        count = [0] * pool_size

        for row in test_set:
            context, response, distractors = row
            with torch.no_grad():
                contexts = Variable(torch.stack([torch.LongTensor(context) for i in range(pool_size)], 0)).cuda()
                responses = [torch.LongTensor(response)]
                responses += [torch.LongTensor(distractor) for distractor in distractors]
                responses = Variable(torch.stack(responses, 0)).cuda()

            results, responses = model(contexts, responses)
            results = [res.data.cpu().numpy() for res in results]

            # calculate the position of the correct response and update the counter of that position
            better_count = sum(1 for value in results[1:] if value >= results[0])
            count[better_count] += 1
        return count

