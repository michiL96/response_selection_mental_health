import torch
from utilities import get_optimizer
from early_stopping import EarlyStopping
from sequential_matching_network import SMN
from smn_metrics import compute_recall_at_k
from smn_train_valid_test import SMNTrainValidTest
from smn_load_datasets import load_training_set, load_validation_test_set


def execute_smn(config: dict, nlp):
    training_filepath = "../../Data/FollowUpConv_Sislab_Idego_Selection_task_[Context,Response,Flag]_Train.csv"
    training_set, vocabulary = load_training_set(training_filepath, nlp, config['General']['pad_token'], config['General']['oov_token'], config['General']['max_sentence_length'])
    
    validation_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_A.csv"
    validation_set = load_validation_test_set(validation_filepath, nlp, config['General']['max_sentence_length'])
    validation_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_B.csv"
    validation_set += load_validation_test_set(validation_filepath, nlp, config['General']['max_sentence_length'])
    
    # build model
    model = SMN(config, vocabulary, config['General']['pad_token'], config['General']['oov_token'])
    print(model)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config['General']['optimizer_name'], config['General']['learning_rate'])
    
    model.cuda()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {pytorch_total_params}')
    
    all_early_stopping = EarlyStopping(patience=3)
    
    print('Starting training')
    print('-'*50)
    train_valid_test = SMNTrainValidTest(vocabulary, config['General']['pad_token'], config['General']['oov_token'], config['General']['max_sentence_length'])
    
    num_epochs = 50
    best_model = model
    for epoch in range(num_epochs):
        if all_early_stopping.stop_training:
            break
    
        train_loss = train_valid_test.training_phase(model, training_set, loss_function, optimizer, config)
        print(f'Epoch {epoch+1}: ')
        print(f'\tTrain loss={round(sum(train_loss)/len(train_loss), 4)}')
    
        print('\t\tEvaluating on Validation set')
        dev_loss = train_valid_test.validation_phase(model, validation_set, loss_function, config)
        print(f'\t\t\tDevelopment loss={round(sum(dev_loss)/len(dev_loss), 4)}')
    
        all_early_stopping.on_epoch_end(epoch=(epoch + 1), current_value=round(sum(dev_loss)/len(dev_loss), 5))
        if all_early_stopping.wait == 0:
            best_model = model
            torch.save(best_model.state_dict(), config['General']['model_filepath'])
    
    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_C.csv"
    test_set = load_validation_test_set(test_filepath, nlp, config['General']['max_sentence_length'])
    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_D.csv"
    test_set += load_validation_test_set(test_filepath, nlp, config['General']['max_sentence_length'])
    all_candidate_scores, test_labels, _ = train_valid_test.test_phase(best_model, test_set, loss_function)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=2, k=1)
    
    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_C.csv"
    test_set = load_validation_test_set(test_filepath, nlp, config['General']['max_sentence_length'])
    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_D.csv"
    test_set += load_validation_test_set(test_filepath, nlp, config['General']['max_sentence_length'])
    all_candidate_scores, test_labels, _ = train_valid_test.test_phase(best_model, test_set, loss_function)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=10, k=1)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=10, k=2)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=10, k=5)
    
    test_filepath = "../../Data/poolSize50/FollowUpConv_Sislab_Idego_Selection_task_poolSize50[Context,Response,Flag]_C+D.csv"
    test_set = load_validation_test_set(test_filepath, nlp, config['General']['max_sentence_length'])
    all_candidate_scores, test_labels, _ = train_valid_test.test_phase(best_model, test_set, loss_function)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=50, k=1)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=50, k=2)
    compute_recall_at_k(all_candidate_scores, test_labels, pool_size=50, k=5)
