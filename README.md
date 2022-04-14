# Response Selection Models for Personal Healthcare Agents in the Mental Health Domain

Michela Lorandi master thesis in 2021 @ UniTn

## Requirements

In order to properly execute the program, you need to install the python libraries in the 
`requirements.txt` file in the root of the repository.

## Response Selection Models for Personal Healthcare Agents in the Mental Health Domain

In this master's thesis, the work is focused on developing a dialogue system to hold a Follow-Up dialogue in a 
context of a Personal Healthcare Agent (PHA) in the mental health domain. Follow-Up dialogues provide the
therapist the ability to monitor the progress of the patient and provide required care and support in a timely
manner. In this context, I worked on developing the module of the dialogue system responsible for the
generation of natural language for the PHA. For this purpose, I developed two **Information Retrieval** models as
*TermFrequency InverseDocumentFrequency* (TF-IDF) and *Best Matching 25* (BM25), as well as two **Neural
Network** architectures as *Siamese Recurrent Neural Networks* (RNN) and *Sequential Matching Network* (SMN).

### Structure of the repository

The structure of the repository is as follows:

- data
- src
  - information_retrieval_models
  - neural_networks
    - word_embeddings

The `data` directory contains the files needed for the training, validation and test of the models. The dataset used in 
this thesis is not publicly available, but some synthetic samples are provided. 

The `src` folder contains all the scripts needed for the correct execution of the models. This folder is divided in 2 
sub-folders: `information_retrieval_models` and `neural_networks`.   
`information_retrieval_models` contains the scripts for the execution of both TF-IDF and BM25. In this folder, you can 
find the `config.yaml` file in which all the parameters needed for both models are specified.   
`neural_networks` contains the scripts for the execution of the Siamese RNN and SMN, and it contains the `config,yaml` file 
used for the configuration of all the parameters needed for the correct execution of the neural architectures. In this folder,
you can find the `word_embeddings` folder which contains the pretrained word embeddings that can be used in both Neural 
Networks.


### How to execute

Before executing the project, make sure to install all the packages specified in the `requirements.txt` file.

In order to execute an *Information Retrieval Model*, you can modify the configuration file at 
`./src/information_retrieval_models/config.yaml` and then run the following command:   
`cd ./src/information_retrieval_models/ && python3 execute.py`

For the execution of a *Neural Network*, you can modify the parameters contained in the configuration file at 
`./src/neural_networks/config.yaml` and then run the following command:
`cd ./src/neural_networks/ && python3 execute.py`
