import os
import yaml
from train_test import train_predictors, load_test_set, make_prediction, recall_at_k

config_filepath = os.path.join('.', 'config.yaml')
with open(config_filepath) as file:
    tmp = yaml.safe_load_all(file)
    for t in tmp:
        config = t

training_filepath = '../../Data/FollowUpConv_Sislab_Idego_Selection_task_[Context,Response,Flag]_Train.csv'
model = train_predictors(training_filepath, config['General']['model_type'])

if model is not None:
    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_C.csv"
    test_set = load_test_set(test_filepath)
    test_filepath = "../../Data/poolSize2/FollowUpConv_Sislab_Idego_Selection_task_poolSize2[Context,Response,Flag]_D.csv"
    test_set += load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    print('1 in 2 => R@1 = ' + str(r1))

    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_C.csv"
    test_set = load_test_set(test_filepath)
    test_filepath = "../../Data/poolSize10/FollowUpConv_Sislab_Idego_Selection_task_poolSize10[Context,Response,Flag]_D.csv"
    test_set += load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    r2 = round(recall_at_k(y_predicted, y, 2), 4)
    r3 = round(recall_at_k(y_predicted, y, 3), 4)
    r5 = round(recall_at_k(y_predicted, y, 5), 4)
    print('1 in 10 => R@1 = ' + str(r1) + '; R@2 = ' + str(r2) + '; R@3 = ' + str(r3) + '; R@5 = ' + str(r5))

    test_filepath = "../../Data/poolSize50/FollowUpConv_Sislab_Idego_Selection_task_poolSize50[Context,Response,Flag]_C+D.csv"
    test_set = load_test_set(test_filepath)
    y, y_predicted = make_prediction(test_set, model)
    r1 = round(recall_at_k(y_predicted, y, 1), 4)
    r2 = round(recall_at_k(y_predicted, y, 2), 4)
    r3 = round(recall_at_k(y_predicted, y, 3), 4)
    r5 = round(recall_at_k(y_predicted, y, 5), 4)
    print('1 in 50 => R@1 = ' + str(r1) + '; R@2 = ' + str(r2) + '; R@3 = ' + str(r3) + '; R@5 = ' + str(r5))
