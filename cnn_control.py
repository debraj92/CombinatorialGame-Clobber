import time

from a2.boolean_negamax_tt import PlayClobber
from a2.clobber_1d import Clobber_1d
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN

import numpy as np
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, BatchNormalization, ReLU, MaxPool1D, Dense, Dropout, AveragePooling1D, GlobalAveragePooling1D
from keras.models import Sequential
import keras.utils
from matplotlib import pyplot as plt
import tensorflow as tf


class cnn:
    training_samples_positive = set()
    training_samples_negative = set()

    elements = [".", "B", "W"]

    games_for_training = set()

    MAX_LENGTH = 40

    train_data = None

    label = None

    EPOCHS = 50

    sample_size = 0  # number of samples in train set
    time_steps = 0  # number of features in train set
    input_dimension = 2  # each feature is represented by 1 number
    #[batch_size, time_steps, input_dimension] ---> [sample_size, time_steps, input_dimension]

    train_data_reshaped = None

    def generateGameCombinations(self, boardString, length, maxlength):

        if length == maxlength:
            self.games_for_training.add(boardString)
            return

        for i in self.elements:
            self.generateGameCombinations(boardString + i, length + 1, maxlength)

    def convertGameStringToInputVector(self, gameBoard):
        x = np.empty(shape=[0, 2])
        for i in range(self.MAX_LENGTH):
            if i < len(gameBoard):
                if gameBoard[i] == "B":
                    x = np.append(x, [[0, 1]], axis=0)
                elif gameBoard[i] == "W":
                    x = np.append(x, [[1, 0]], axis=0)
                else:
                    x = np.append(x, [[0, 0]], axis=0)
            else:
                x = np.append(x, [[0, 0]], axis=0)

        return x

    def createTrainingData(self, first_player=BLACK):
        train_data = []
        label = []
        total = len(self.games_for_training)
        print("creating training data, total games", total)
        count = 0
        for game in self.games_for_training:
            print(game)
            clobber = Clobber_1d(game, first_player)
            play = PlayClobber()
            start_time = time.time()
            outcome, _, _ = play.negamaxClobberGamePlay(clobber, start_time)
            print("Done")
            if outcome == PROVEN_WIN:
                label.append(1)
                train_data.append(self.convertGameStringToInputVector(game))
            else:
                label.append(0)
                train_data.append(self.convertGameStringToInputVector(game))

            count += 1
            complete = int(count/total * 100)

            if complete % 10 == 0:
                print("Completed %", complete)

        self.train_data = np.array(train_data)
        self.label = np.array(label)
        self.label = tf.keras.utils.to_categorical(self.label, num_classes=2)
        print(self.label.shape)
        return

    def reshapeInput(self):
        self.sample_size = self.train_data.shape[0]
        self.time_steps = self.train_data.shape[1]
        self.train_data = np.reshape(self.train_data, (self.sample_size, self.time_steps, self.input_dimension))

    def cnnModel(self):
        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=7, strides=1, activation='relu',
                         input_shape=(self.time_steps, self.input_dimension)))  # 1
        model.add(Dropout(0.25))
        model.add(MaxPool1D(pool_size=2, strides=2))
        model.add(Conv1D(filters=8, kernel_size=5, activation='relu', strides=1))  # 2
        model.add(MaxPool1D(pool_size=2, strides=2))
        model.add(Conv1D(filters=4, kernel_size=3, activation='relu', strides=1))  # 3
        model.add(MaxPool1D(pool_size=2, strides=2))
        model.add(Conv1D(filters=2, kernel_size=1, activation='relu', strides=1))  # 3
        model.add(MaxPool1D(pool_size=2, strides=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu', name="Dense_1"))
        model.add(Dropout(0.25))
        model.add(Dense(self.input_dimension, activation='softmax', name="Dense_2"))
        model.compile('adam', loss='mse', metrics=['accuracy'])
        return model


model = cnn()
model.generateGameCombinations("", 0, 11)
print("Games of size 11 generated")
model.createTrainingData()
print(model.train_data.shape)  # number of games X number of features
print(model.label.shape)  # number of games
model.reshapeInput()
model_conv1D = model.cnnModel()

history = model_conv1D.fit(model.train_data, model.label, epochs=model.EPOCHS,
                    validation_split=0.2, verbose=1)

model_conv1D.save("clobber-black-cnn.h5")

plt.figure(1)
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
plt.show()

print("Training Complete for black")
# REPEAT SAME CODE FOR WHITE

model = cnn()
model.createTrainingData(WHITE)
print(model.train_data.shape)  # number of games X number of features
print(model.label.shape)  # number of games
model.reshapeInput()
model_conv1D = model.cnnModel()

history = model_conv1D.fit(model.train_data, model.label, epochs=model.EPOCHS,
                    validation_split=0.2, verbose=1)
#plot_history(history)
model_conv1D.save("clobber-white-cnn.h5")

plt.figure(1)
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
plt.show()
print("Training Complete for white")