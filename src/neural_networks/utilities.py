import torch


def load_word_embeddings(word_vocabulary: dict, filename: str = '../word_embeddings/twitter128.tsv'):
    lines = open(filename).readlines()
    embeddings = {}
    for line in lines:
        word = ''.join(line.split('\t')[0])
        embedding = list(map(float, line.split('\t')[1:]))
        if word in word_vocabulary:
            embeddings[word_vocabulary[word]] = embedding
    return embeddings


def text_to_word_sequence(utterance: str, nlp):
    return [tok.text for tok in nlp(utterance)]


def get_optimizer(model, optimizer_name: str, learning_rate: float):
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        return None
