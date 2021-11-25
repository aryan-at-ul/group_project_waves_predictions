from colorama import Fore, Back, Style
from colorama import init
from termcolor import colored
import os
import sys
from models import *
import matplotlib
import tensorflow as tf
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
### call from main and then route request for train and test

typ_for_models = ['waveheight','waveperiod']

all_models = {"model_1":{"type":"sequential",
                         "horizon":1,
                         "window_size":7,
                         "activation":"relu",
                         "epoch":100,
                         "function_to_call":"model1_trainer(train_windows, test_windows, train_labels,test_labels,model_name,hyper_paramas)"},
              "model_2":{"type":"sequential conv2",
                         "horizon":1,
                         "window_size":7,
                         "activation":"relu",
                         "epoch":100,
                         "function_to_call":"model2_trainer(train_windows, test_windows, train_labels,test_labels,model_name,hyper_paramas)"

              }
              }

WINDOW_SIZE = 7
HORIZON = 1
def get_train_test_splits(timesteps,timedata,extra):

    full_windows, full_labels = make_windows(timedata, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    return train_windows, test_windows, train_labels, test_labels


def do_train_test_if_model_does_not_exit(df,station_name):
    result_to_write = []
    for typ in typ_for_models:
        for model in all_models.keys():
            result_to_write  = []
            choice = -1
            model_name = f"{model}_{station_name}_{typ}"
            timesteps = df.index.to_numpy()
            timedata = df[typ].to_numpy()
            extra = {}
            train_windows, test_windows, train_labels, test_labels  = get_train_test_splits(timesteps,timedata,extra)
            print(model_name)
            if os.path.exists('/'.join([os.getcwd(), 'model_experiments',model_name])):
                prompt = "A trained model already exit , would you like to train again or use existing model,\n1 : train again \n0 : test existing\n"
                choice = input(prompt)
            if int(choice) == 1 or choice == -1:
                print(colored(f"starting to train and test for {station_name}, model : {model} for {typ}", 'blue', 'on_white'))
                model_params = all_models.get(model)
                hyper_paramas = model_params.get("")
                fn_to_call = model_params["function_to_call"]

                model = eval(fn_to_call)
                print("model evaluation:", model.evaluate(test_windows,test_labels))
                model_preds = make_preds(model,input_data = test_windows)
                model_results = evaluate_preds(y_true=tf.squeeze(test_labels),y_pred = model_preds)
                print(colored(f"model performace {model_name} [newly trained] : {model_results}",'red','on_yellow'))
            else:
                print(colored(f"you will be prompted with existing model paramteres for : {station_name}, model : {model} for {typ}", 'blue', 'on_white'))
                model_path = get_model_path(model_name)
                model = tf.keras.models.load_model(model_path)
                print("test window",test_windows)
                model_preds = make_preds(model,input_data = test_windows)
                print("Predictions",model_preds)
                print("Ylabels",test_labels)
                model_results = evaluate_preds(y_true=tf.squeeze(test_labels),y_pred = model_preds)
                result_to_write.append(model_results)
                print(colored(f"model performace : {model_results}",'red','on_yellow'))

            write_results_to_csv(result_to_write,station_name,typ,model_name)
