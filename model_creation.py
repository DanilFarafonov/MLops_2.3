from keras.models import Sequential
from keras.layers import Dense


def create_model(x_train, y_train):

    model = Sequential()
    model.add(Dense(800, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size=200,
              epochs=50,
              verbose=1)

    return model
