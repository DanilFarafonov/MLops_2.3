import pandas as pd
import keras


def preprocess(df_x_train, df_y_train, df_x_test, df_y_test):

    x_train = df_x_train.to_numpy()
    y_train = df_y_train.to_numpy()
    x_test = df_x_test.to_numpy()
    y_test = df_y_test.to_numpy()

    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    df_x_train = pd.DataFrame(x_train)
    df_y_train = pd.DataFrame(y_train)
    df_x_test = pd.DataFrame(x_test)
    df_y_test = pd.DataFrame(y_test)

    return df_x_train, df_y_train, df_x_test, df_y_test
