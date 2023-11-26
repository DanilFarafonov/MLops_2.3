import keras
import pandas as pd


def load():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    df_x_train = pd.DataFrame(x_train.reshape(60000, 784))
    df_y_train = pd.DataFrame(y_train)
    df_x_test = pd.DataFrame(x_test.reshape(10000, 784))
    df_y_test = pd.DataFrame(y_test)

    return df_x_train, df_y_train, df_x_test, df_y_test
