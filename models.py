import tensorflow as tf
from tensorflow.keras import layers
from utils import *
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.random.set_seed(42) # cheers for the community always use seed as 42

HORIZON = 1
WINDOW = 7


def model1_trainer(train_windows, test_windows, train_labels,test_labels,model_name,hyper_paramas):
    # here typ will be doing it for waveheight or waveperiod
    model_1 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear")
    ], name=f"{model_name}") # model name 1

    # Compile model
    model_1.compile(loss="mae",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["mae"])
    # Fit model , some thing is wrong with this part overfitting
    model_1.fit(x=train_windows, # train windows of 7 timesteps of waveperiod 1-7 days and then 7th day the target
            y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day) a bit like naive
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_1.name)])
    return model_1



def model2_trainer(train_windows, test_windows, train_labels,test_labels,model_name,hyper_paramas):
    """:
    training 2nd type model here
    """

    model_2 = tf.keras.Sequential([
                layers.Dense(128, activation="relu"),
                layers.Dense(HORIZON)
            ], name=f"{model_name}")
    model_2.compile(loss="mae",
            optimizer=tf.keras.optimizers.Adam())

    model_2.fit(train_windows,
            train_labels,
            batch_size=128,
            epochs=100,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_2.name)])

    return model_2


