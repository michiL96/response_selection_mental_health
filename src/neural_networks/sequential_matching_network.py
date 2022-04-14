import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from vocabulary import CustomVocabulary
from locked_dropout import LockedDropout
from weight_dropout import WeightDropout
from utilities import load_word_embeddings


class SMN(nn.Module):
    def __init__(self, config: dict, vocabulary: CustomVocabulary, pad_token: int = 0, oov_token: int = 1):
        super(SMN, self).__init__()

        self.embedding_filepath = config['General']['pretrained_word_embedding']
        self.max_sentence_len = config['General']['max_sentence_length']
        self.embedding_size = config['General']['word_embedding_size']
        self.rnn_units = config['General']['rnn_cell_type']
        self.voc_size = vocabulary.num_words
        self.batch_size = config['General']['batch_size']
        self.kernel_size = config['SMN']['kernel_size']
        self.stride = config['SMN']['stride']
        self.num_matrix = config['SMN']['num_matrices']
        self.conv_channels = config['SMN']['convolutional_channels']
        self.rnn_units_final = config['SMN']['final_rnn_units']
        self.output_labels = config['SMN']['output_labels']
        self.use_emb_dropout = config['SMN']['use_embedding_dropout']
        self.lock_drop_p = config['SMN']['embedding_locked_probability']
        self.use_lock_dropout = config['SMN']['use_locked_dropout']
        self.use_batch_norm = config['SMN']['use_batch_normalization']
        self.use_weight_dropout = config['SMN']['embedding_weight_probability']
        self.weight_drop_p = config['SMN']['weight_drop_p']
        self.vocabulary = vocabulary
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.padding = config['SMN']['padding']
        self.dilation = config['SMN']['dilation']
        self.stride_conv = config['SMN']['stride_conv']

        # NETWORK COMPONENTS
        self.word_embedding = nn.Embedding(self.voc_size, self.embedding_size)
        if self.use_emb_dropout:
            self.emb_dropout = nn.Dropout(config['General']['embedding_dropout_probability'])
        self.utterance_gru = nn.GRU(self.embedding_size, self.rnn_units, bidirectional=False, batch_first=True)

        if self.use_lock_dropout:
            self.lock_dropout = LockedDropout()

        if self.use_weight_dropout:
            self.weight_dropout_utt = WeightDropout(self.utterance_gru, ['weight_hh_l0'], dropout=self.weight_drop_p)

        self.conv2d = nn.Conv2d(self.num_matrix, config['General']['convolutional_channels'], kernel_size=self.kernel_size)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(config['General']['convolutional_channels'])

        self.maxPool = nn.MaxPool2d(self.kernel_size, self.stride)

        conv_h_out = math.floor(((self.max_sentence_len + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride_conv[0])+1)
        conv_w_out = math.floor(((self.max_sentence_len + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride_conv[1])+1)

        pool_h_out = math.floor(((conv_h_out + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1)
        pool_w_out = math.floor(((conv_w_out + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1)

        input_linear = pool_h_out * pool_w_out * self.conv_channels

        self.linear = nn.Linear(input_linear, self.rnn_units_final)
        self.A_matrix = torch.ones((self.rnn_units, self.rnn_units), requires_grad=True)
        self.final_gru = nn.GRU(self.rnn_units_final, self.rnn_units, bidirectional=False, batch_first=True)

        if self.use_weight_dropout:
            self.weight_dropout_fin = WeightDropout(self.final_gru, ['weight_hh_l0'], dropout=self.weight_drop_p)

        self.final_linear = nn.Linear(self.rnn_units, self.output_labels)

        self.init_weights()

        self.init_weights_embedding()

    def init_weights(self):
        """
        Initialize the weights of all the networks' components
        :return:
        """
        ih_u = (param.data for name, param in self.utterance_gru.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utterance_gru.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)

        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)

        init.xavier_uniform_(self.A_matrix)
        self.A_matrix = self.A_matrix.cuda()

        ih_f = (param.data for name, param in self.final_gru.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_gru.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)

        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)

    def init_weights_embedding(self):
        word_embeddings = load_word_embeddings(self.vocabulary.word2index, self.embedding_filepath)
        embedding_weights = torch.FloatTensor(self.vocabulary.num_words, self.embedding_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k,v in word_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[self.pad_token] = torch.FloatTensor([self.pad_token]*self.embedding_size)
        del self.word_embedding.weight
        self.word_embedding.weight = nn.Parameter(embedding_weights)
        self.word_embedding.weight.requires_grad = False

    def forward(self, history, response):
        """
            history:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        """
        history_embedding = self.word_embedding(history)
        response_embedding = self.word_embedding(response)

        if self.use_emb_dropout:
            history_embedding = self.emb_dropout(history_embedding)
            response_embedding = self.emb_dropout(response_embedding)

        history_embedding = history_embedding.permute(1, 0, 2, 3)

        response_gru_embedding, _ = self.utterance_gru(response_embedding)
        response_embedding = response_embedding.permute(0, 2, 1)
        response_gru_embedding = response_gru_embedding.permute(0, 2, 1)
        matching_vectors = []

        for utterance_embedding in history_embedding:
            matrix1 = torch.matmul(utterance_embedding, response_embedding)

            if self.use_weight_dropout:
                utterance_gru_embedding, _ = self.weight_dropout_utt(utterance_embedding)
            else:
                utterance_gru_embedding, _ = self.utterance_gru(utterance_embedding)

            matrix2 = torch.einsum('aij,jk->aik', utterance_gru_embedding, self.A_matrix)
            matrix2 = torch.matmul(matrix2, response_gru_embedding)

            matrix = torch.stack([matrix1, matrix2], dim=1)
            # matrix:(batch_size,channel,seq_len,embedding_size)
            conv_layer = self.conv2d(matrix)

            if self.use_batch_norm:
                conv_layer = self.batch_norm(conv_layer)

            if self.use_lock_dropout:
                conv_layer = self.lock_dropout(conv_layer, dropout=self.lock_drop_p)

            # add activate function
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.maxPool(conv_layer)
            # flatten
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = self.linear(pooling_layer)
            # add activate function
            matching_vector = torch.tanh(matching_vector)
            matching_vectors.append(matching_vector)

        if self.use_weight_dropout:
            _, last_hidden = self.weight_dropout_fin(torch.stack(matching_vectors, dim=1))
        else:
            _, last_hidden = self.final_gru(torch.stack(matching_vectors, dim=1))

        last_hidden = torch.squeeze(last_hidden)

        if self.use_lock_dropout:
            last_hidden = self.lock_dropout(last_hidden, dropout=self.lock_drop_p)

        logits = self.final_linear(last_hidden)
        y_pred = logits
        return y_pred
