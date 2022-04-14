import torch
from utilities import get_optimizer
from early_stopping import EarlyStopping
from siamese_nn import EncoderRNN, DualEncoder
from siamese_load_datasets import load_training_set, load_validation_set, load_test_set
from siamese_train_valid_test import SiameseTrainValidTest, get_batch, recall_at_k, core_convertor


def execute_siamese_nn(config: dict, nlp):
    training_filepath = "../../Data/FollowUpConv_Sislab_Idego_Selection_task_[Context,Response,Flag]_Train.csv"
    vocabulary, training_data = load_training_set(training_filepath, nlp, pad_token=config['General']['pad_token'],
                                                  oov_token=config['General']['oov_token'],
                                                  max_sentence_length=config['General']['max_sentence_length'])
    # Trim voc and pairs
    vocabulary.trim(config['General']['min_word_count'])

    valid_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_A.csv"
    validation_data = load_validation_set(valid_filepath, nlp)
    valid_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_B.csv"
    validation_data += load_validation_set(valid_filepath, nlp)
    print("Validation set - Read {!s} sentence pairs".format(len(validation_data)))

    all_early_stopping = EarlyStopping(patience=3)

    encoder_model = EncoderRNN(config['General']['word_embedding_size'], config['SiameseNN']['hidden_size'], vocabulary,
                               config['SiameseNN']['num_layers'], config['SiameseNN']['dropout'],
                               config['SiameseNN']['bidirectional'], config['General']['rnn_cell_type'])
    encoder_model.cuda()

    model = DualEncoder(encoder_model)
    model.cuda()
    print(model)

    loss_function = torch.nn.BCELoss()    # Binary CrossENtropy
    loss_function.cuda()

    optimizer = get_optimizer(model, config['General']['optimizer_type'], config['General']['learning_rate'])

    train_valid_test = SiameseTrainValidTest(vocabulary, config['General']['pad_token'], config['General']['oov_token'],
                                             config['General']['max_sentence_length'])

    print("Training started")
    best_model = model
    for epoch in range(config['General']['num_epochs']):
        print(f'Epoch {epoch}')
        if all_early_stopping.stop_training:
            break

        train_loss = train_valid_test.training_phase(model, training_data, config['General']['batch_size'],
                                                     loss_function, optimizer, config['General']['gradient_clipping'])
        print(f'\tTrain loss: {train_loss}')

        print('\t\tEvaluating on Development set')
        validation_loss = train_valid_test.validation_phase(model, validation_data, config['General']['batch_size'],
                                                            loss_function)
        print(f'\t\t\tDevelopment Loss: {round(validation_loss, 4)}')

        all_early_stopping.on_epoch_end(epoch=(epoch + 1), current_value=round(validation_loss, 5))
        if all_early_stopping.wait == 0:
            best_model = model
            torch.save(best_model.state_dict(), config['General']['model_filepath'])

    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_C.csv"
    test_data = load_test_set(test_filepath, nlp)
    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_D.csv"
    test_data += load_test_set(test_filepath, nlp)
    print("Test set - Read {!s} sentence pairs".format(len(test_data)))
    counts = train_valid_test.test_phase(best_model, test_data, pool_size=2)
    recall_at_k(counts, len(test_data), 1, pool_size=2)

    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_C.csv"
    test_data = load_test_set(test_filepath, nlp)
    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_D.csv"
    test_data += load_test_set(test_filepath, nlp)
    counts = train_valid_test.test_phase(best_model, test_data, pool_size=10)
    recall_at_k(counts, len(test_data), 1, pool_size=10)
    recall_at_k(counts, len(test_data), 3, pool_size=10)
    recall_at_k(counts, len(test_data), 5, pool_size=10)

    test_filepath = "../../Data/poolSize50/FollowUpConv_Sislab_Idego_Selection_task_poolSize50[Context,Response,Flag]_C+D.csv"
    test_data = load_test_set(test_filepath, nlp)
    counts = train_valid_test.test_phase(best_model, test_data, pool_size=50)
    recall_at_k(counts, len(test_data), 1, pool_size=50)
    recall_at_k(counts, len(test_data), 3, pool_size=50)
    recall_at_k(counts, len(test_data), 5, pool_size=50)
