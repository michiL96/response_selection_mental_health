import torch
import torch.nn as nn
from torch.nn import init
from utilities import load_word_embeddings


class EncoderRNN(nn.Module):
    """
    Implementation of the Encoder for the Siamese Neural Network
    """
    def __init__(self, input_size, hidden_size, vocabulary, n_layers=1, dropout=0,
                 bidirectional=True, rnn_type='gru', pretrained_word_embedding=None, pad_token: int = 0):
        super(EncoderRNN, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.vocab_size = vocabulary.num_words
        self.input_size = input_size
        self.hidden_size = hidden_size // self.num_directions
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.voc = vocabulary
        self.pad_token = pad_token

        self.embedding = nn.Embedding(self.vocab_size, input_size, padding_idx=pad_token, sparse=False)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout),
                              bidirectional=bidirectional, batch_first=True).cuda()
        else:
            self.rnn = nn.LSTM(input_size, self.hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout),
                               bidirectional=bidirectional, batch_first=True).cuda()

        self.init_weights()
        if pretrained_word_embedding is not None:
            self.init_weights_embedding(pretrained_word_embedding)

    def init_weights(self):
        init.orthogonal_(self.rnn.weight_ih_l0)
        init.uniform_(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

    def init_weights_embedding(self, pretrained_word_embedding):
        word_embeddings = load_word_embeddings(self.voc.word2index, pretrained_word_embedding)
        embedding_weights = torch.FloatTensor(self.voc.num_words, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k,v in word_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[self.pad_token] = torch.FloatTensor([self.pad_token]*self.input_size)
        del self.embedding.weight
        self.embedding.weight = nn.Parameter(embedding_weights)
        self.embedding.weight.requires_grad = False

    def forward(self, input_seq):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Forward pass
        outputs, hidden = self.rnn(embedded)
        # Return output and final hidden state
        return outputs, hidden


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder

        h_size = self.encoder.hidden_size * self.encoder.num_directions
        M = torch.FloatTensor(h_size, h_size).cuda()
        init.normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, responses):
        context_os, context_hs = self.encoder(contexts)
        response_os, response_hs = self.encoder(responses)

        if self.encoder.rnn_type == 'lstm':
            context_hs = context_hs[0]
            response_hs = response_hs[0]

        results = []
        response_encodings = []

        h_size = self.encoder.hidden_size * self.encoder.num_directions
        for i in range(len(context_hs[0])):
            context_h = context_os[i][-1].view(1, h_size)
            response_h = response_os[i][-1].view(h_size, 1)
            ans = torch.mm(torch.mm(context_h, self.M), response_h)[0][0]

            results.append(torch.sigmoid(ans))
            response_encodings.append(response_h)

        results = torch.stack(results)

        return results, response_encodings
