import os
import yaml
import torch
import spacy
import random
from execute_smn import execute_smn
from execute_siamese_nn import execute_siamese_nn

config_filepath = os.path.join('..', 'neural_networks/config.yaml')
with open(config_filepath) as file:
    tmp = yaml.safe_load_all(file)
    for t in tmp:
        config = t

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

random.seed(config['General']['seed'])
torch.manual_seed(config['General']['seed'])
torch.cuda.manual_seed_all(config['General']['seed'])

nlp = spacy.load('it_core_news_lg')

if config['General']['architecture_type'] == 'siamese_nn':
    execute_siamese_nn(config, nlp)
elif config['General']['architecture_type'] == 'smn':
    execute_smn(config, nlp)
else:
    print("The architecture specified is not valid")
