import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

def write_results_to_csv(result_list,station_name,typ,model):
    """
    write df created from result set to a csv file
    """
    result = []
    for one in result_list:
        result.append([station_name,one['rmse'],typ,model])
    df = pd.DataFrame(result,
                  columns = ['station_name' , 'rmse','on_col','model_name'])

    df.to_csv('result_compare.csv', mode='a', header=False,index = False)

def mean_absolute_scaled_error(y_true, y_pred):
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # implement for season if one knows the season
  return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    #all values should be as small as possible
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}

def make_preds(model, input_data):
    #print(input_data[0])
    forecast = model.predict(input_data)
    #print(forecast)
    return tf.squeeze(forecast)


def get_model_path(model_name):

    return os.path.join(os.getcwd(), 'model_experiments',model_name)

def evaluate_preds2(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)


    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None,figname = None):

    # Plot the series do not pass df
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("time")
    plt.ylabel("y values")
    if label:
        plt.legend(fontsize=14) # make label bigger
    if figname:
        plt.savefig(f'{figname}.png')
    plt.grid(True)



def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                                verbose=0,
                                                save_best_only=True)


def get_labelled_windows(x, horizon=1):
    #splitting data into two , past and future
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size=7, horizon=1):
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels



def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def data_preparation_for_nbeats(df,WINDOW_SIZE = 7):
    waves_nbeats = df.copy()
    for i in range(WINDOW_SIZE):
        waves_nbeats[f"waveperiod+{i+1}"] = waves_nbeats["waveperiod"].shift(periods=i+1)
    waves_nbeats = waves_nbeats.filter(like='waveperiod')
    waves_nbeats.dropna().head()

    # above df has been prepared below we clean it to prepare test and train split

    X = waves_nbeats.dropna().drop("waveperiod", axis=1)
    y = waves_nbeats.dropna()["waveperiod"]

    # Make train and test sets
    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    print(f"size of x_train{len(X_train)}, size of y_train {len(y_train)} , size of x_test {X_test} and size of y_test {y_test}")

    # on  feaures and labels

    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

    # 2. Combine features & labels
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    # 3. Batch and prefetch for optimal performance
    BATCH_SIZE = 1024 # taken from Appendix D in N-BEATS paper
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset




