import time

from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN

import numpy as np
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, BatchNormalization, ReLU, MaxPool1D, Dense, Dropout, \
    AveragePooling1D, GlobalAveragePooling1D
from keras.models import Sequential, load_model
import keras.utils
from matplotlib import pyplot as plt
import tensorflow as tf
import shutil
import random
from numpy import random

'''
Reuse checkpoints:
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Load the previously saved weights
model.load_weights(latest)
'''


class cnn:

    move_ordering = {"rl": False, "cnn": True, "default": False, "none": False}
	
    elements = [".", "B", "W"]

    games_for_training = set()

    MAX_LENGTH = 40

    train_data = None

    label = None

    EPOCHS = 40

    sample_size = 0  # number of samples in train set
    time_steps = 0  # number of features in train set
    input_dimension = 2  # each feature is represented by 1 number
    # [batch_size, time_steps, input_dimension] ---> [sample_size, time_steps, input_dimension]

    train_data_reshaped = None

    def createRandomBoard(self, size):
        elements = ["B", "W", "."]
        board = ""
        for i in range(size):
            board += elements[np.random.randint(low=0, high=3)]

        return board

    def generateGameCombinations(self, boardString, length, maxlength):

        if length == maxlength:
            self.games_for_training.add(boardString)
            return

        for i in self.elements:
            self.generateGameCombinations(boardString + i, length + 1, maxlength)

    def convertGameBoardToVector(self, gameBoard):
        x = np.empty(shape=[0, 2])
        for i in range(len(gameBoard)):
            if gameBoard[i] == "B":
                x = np.append(x, [[0, 1]], axis=0)
            elif gameBoard[i] == "W":
                x = np.append(x, [[1, 0]], axis=0)
            else:
                x = np.append(x, [[0, 0]], axis=0)

        return x

    def createSparseGame(self, pieceCount):
        dotsCount = self.MAX_LENGTH - pieceCount
        dots_indices = set(np.random.choice(self.MAX_LENGTH, dotsCount, replace=False))
        game = "........................................"
        for i in range(self.MAX_LENGTH):
            if i not in dots_indices:
                toss = random.randint(2)
                if toss == 0:
                    game = game[:i] + "B" + game[i + 1:]
                else:
                    game = game[:i] + "W" + game[i + 1:]

        self.games_for_training.add(game)

    def createInputFeatureVector(self, gameBoard):
        x = self.convertGameBoardToVector(gameBoard)
        padding = random.randint(15)
        dots_prefix = np.full((padding, 2), [[0, 0]], dtype=np.float32)
        x = np.concatenate((dots_prefix, x), axis=0)
        empty_positions_to_add = self.MAX_LENGTH - len(x)
        dots_suffix = np.full((empty_positions_to_add, 2), [[0, 0]], dtype=np.float32)
        x = np.concatenate((x, dots_suffix), axis=0)
        return x

    def createTrainingData(self, first_player=BLACK, sparseGame=False):
        train_data = []
        label = []
        total = len(self.games_for_training)
        print("creating training data, total games", total)
        count = 0
        for game in self.games_for_training:
            print(game)
            clobber = Clobber_1d(game, first_player)
            play = PlayClobber(self.move_ordering)
            start_time = time.time()
            outcome, _, _ = play.negamaxClobberGamePlay(clobber, start_time)
            print("Done")
            if outcome == PROVEN_WIN:
                label.append(1)
                if sparseGame:
                    train_data.append(self.convertGameBoardToVector(game))
                else:
                    train_data.append(self.createInputFeatureVector(game))
            else:
                label.append(0)
                if sparseGame:
                    train_data.append(self.convertGameBoardToVector(game))
                else:
                    train_data.append(self.createInputFeatureVector(game))

            count += 1
            complete = int(count / total * 100)

            if complete % 10 == 0:
                print("Completed %", complete)

        self.train_data = np.array(train_data)
        self.label = np.array(label)
        self.label = tf.keras.utils.to_categorical(self.label, num_classes=2)
        print(self.label.shape)
        return

    def serialize_train_data(self, fname_train, fname_label):
        with open(fname_train, 'wb') as f:
            np.save(f, self.train_data)

        with open(fname_label, 'wb') as f:
            np.save(f, self.label)

    def read_train_data(self, fname_train, fname_label):
        with open(fname_train, 'rb') as f:
            self.train_data = np.load(f)
            print(self.train_data)

        with open(fname_label, 'rb') as f:
            self.label = np.load(f)
            print(self.label)

    def reshapeInput(self):
        self.sample_size = self.train_data.shape[0]
        self.time_steps = self.train_data.shape[1]
        print("Sample size", self.sample_size)
        print("time steps", self.time_steps)
        self.train_data = np.reshape(self.train_data, (self.sample_size, self.time_steps, self.input_dimension))

    def cnnModel(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=9, strides=1, activation='relu',
                         input_shape=(self.time_steps, self.input_dimension)))  # 1
        model.add(Dropout(0.2))
        model.add(MaxPool1D(pool_size=2, strides=1))
        model.add(Conv1D(filters=32, kernel_size=7, strides=1, activation='relu',
                         input_shape=(self.time_steps, self.input_dimension)))  # 2
        model.add(Dropout(0.1))
        model.add(MaxPool1D(pool_size=2, strides=1))
        model.add(Conv1D(filters=16, kernel_size=5, activation='relu', strides=1))  # 3
        model.add(MaxPool1D(pool_size=2, strides=1))
        model.add(Conv1D(filters=8, kernel_size=3, activation='relu', strides=1))  # 4
        model.add(MaxPool1D(pool_size=2, strides=1))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', name="Dense_1"))
        model.add(Dense(self.input_dimension, activation='softmax', name="Dense_2"))
        model.compile('adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
        return model

    def reloadModel(self, type_player=BLACK):

        if type_player == BLACK:
            return load_model('./clobber-black-cnn.h5')

        return load_model('./clobber-white-cnn.h5')


class cnn_trainer:
    RANDOM_COMBINATION = 1
    ALL_COMBINATION = 2

    def createModelFromScratch(self, board_size, combinations=ALL_COMBINATION, number_of_samples=None,
                               skip_b_train=False):
        model = cnn()
        if combinations == self.ALL_COMBINATION:
            model.generateGameCombinations("", 0, board_size)
            print("Games of size 11 generated")
        else:
            for i in range(number_of_samples):
                board = model.createRandomBoard(board_size)
                model.games_for_training.add(board)

            print("All samples generated")
        if not skip_b_train:
            model.createTrainingData()

            print(model.train_data.shape)  # number of games X number of features
            print(model.label.shape)  # number of games
            model.reshapeInput()

            # model.serialize_train_data('train_data_samples-black.npy', 'labels-black.npy')
            # model.read_train_data('train_data_samples-black.npy', 'labels-black.npy')

            model_conv1D = model.cnnModel()

            history = model_conv1D.fit(model.train_data, model.label,
                                       epochs=model.EPOCHS,
                                       validation_split=0.2, verbose=1)

            model_conv1D.save("clobber-black-cnn.h5")

            plt.figure(1)
            plt.subplot(211)
            plt.plot(history.history['auc'])
            plt.plot(history.history['val_auc'])
            plt.title('model auc')
            plt.ylabel('auc')
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
            # plt.show()
            plt.savefig('performance-black.png')

            plt.clf()

            print("Training Complete for black")

        # REPEAT SAME CODE FOR WHITE

        model_temp = model

        model = cnn()
        model.games_for_training = model_temp.games_for_training

        model.createTrainingData(WHITE)
        print(model.train_data.shape)  # number of games X number of features
        print(model.label.shape)  # number of games
        model.reshapeInput()

        # model.serialize_train_data('train_data_samples-white.npy', 'labels-white.npy')
        # model.read_train_data('train_data_samples-black.npy', 'labels-black.npy')

        model_conv1D = model.cnnModel()

        history = model_conv1D.fit(model.train_data, model.label,
                                   epochs=model.EPOCHS,
                                   validation_split=0.2, verbose=1)
        # plot_history(history)
        model_conv1D.save("clobber-white-cnn.h5")

        plt.figure(1)
        plt.subplot(211)
        plt.plot(history.history['auc_1'])
        plt.plot(history.history['val_auc_1'])
        plt.title('model auc')
        plt.ylabel('auc')
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
        # plt.show()
        plt.savefig('performance-white.png')
        print("Training Complete for white")

    def retrainModelWithSample(self, number_of_samples, board_size, sparseGame=False):
        model = cnn()

        for i in range(number_of_samples):
            if not sparseGame:
                board = model.createRandomBoard(board_size)
                model.games_for_training.add(board)
            else:
                model.createSparseGame(board_size)

        # black
        model.createTrainingData(BLACK, sparseGame)
        model.reshapeInput()

        model.serialize_train_data('train_data_samples-black.npy', 'labels-black.npy')
        #model.read_train_data('train_data_samples-black.npy', 'labels-black.npy')

        model_conv1D = model.reloadModel()

        history = model_conv1D.fit(model.train_data, model.label,
                                   epochs=model.EPOCHS,
                                   validation_split=0.2, verbose=1)

        model_conv1D.save("clobber-black-cnn-retrained.h5")

        plt.figure(1)
        plt.subplot(211)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('model auc')
        plt.ylabel('auc')
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
        # plt.show()
        plt.savefig('performance-black-cnn-retrained.png')
        plt.clf()
        print("Training Complete for black")

        # white
        model_temp = model

        model = cnn()
        model.games_for_training = model_temp.games_for_training

        model.createTrainingData(WHITE, sparseGame)
        model.reshapeInput()

        model.serialize_train_data('train_data_samples-white.npy', 'labels-white.npy')
        #model.read_train_data('train_data_samples-black.npy', 'labels-black.npy')

        model_conv1D = model.reloadModel(WHITE)

        history = model_conv1D.fit(model.train_data, model.label,
                                   epochs=model.EPOCHS,
                                   validation_split=0.2, verbose=1)

        model_conv1D.save("clobber-white-cnn-retrained.h5")

        plt.figure(1)
        plt.subplot(211)
        plt.plot(history.history['auc_1'])
        plt.plot(history.history['val_auc_1'])
        plt.title('model auc')
        plt.ylabel('auc')
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
        # plt.show()
        plt.savefig('performance-white-cnn-retrained.png')
        print("Training Complete for white")

    def finalizeRetraining(self):
        shutil.copy("clobber-white-cnn-retrained.h5", "clobber-white-cnn.h5")
        shutil.copy("clobber-black-cnn-retrained.h5", "clobber-black-cnn.h5")

    def generateTFLite(self):
        model = cnn()
        model_conv1D = model.reloadModel()
        converter = tf.lite.TFLiteConverter.from_keras_model(model_conv1D)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()
        with open('./clobber-black-cnn.tflite', 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)

        model_conv1D = model.reloadModel(WHITE)
        converter = tf.lite.TFLiteConverter.from_keras_model(model_conv1D)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()
        with open('./clobber-white-cnn.tflite', 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)


t = cnn_trainer()

t.createModelFromScratch(17, t.RANDOM_COMBINATION, 800000, False)

#t.retrainModelWithSample(500000, 26)

#t.retrainModelWithSample(600000, 20, True)

#t.finalizeRetraining()

t.generateTFLite()
