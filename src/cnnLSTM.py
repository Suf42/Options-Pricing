import tensorflow as tf
import pandas as pd
import numpy as np
import time



class cnnLSTM(tf.keras.Model):

    def __init__(self, num_features : int, prev_time_step : int):

        super().__init__()
        self.num_features = num_features
        self.prev_time_step = prev_time_step
        self.cnn = self.cnn()
        self.lstm = self.lstm()



    def cnn(self) -> tf.keras.Model:

        inputs = tf.keras.Input(
            shape = (self.prev_time_step, self.num_features, ), 
            name = 'input_layer_cnn' 
        )

        conv1d = tf.keras.layers.Conv1D(
            filters = 32, 
            kernel_size = 10, 
            padding = 'causal', 
            dilation_rate = 4, 
            activation = 'elu', 
            name = 'conv1d_1' 
        )(inputs)

        conv1d = tf.keras.layers.Conv1D(
            filters = 64, 
            kernel_size = 10, 
            padding = 'causal', 
            dilation_rate = 4, 
            activation = 'elu', 
            name = 'conv1d_2' 
        )(conv1d)

        cnn = tf.keras.Model(
            inputs = inputs, 
            outputs = conv1d, 
            name = 'cnn' 
        )

        return cnn



    def lstm(self) -> tf.keras.Model:

        shape = self.cnn.get_layer('conv1d_2').output.shape[1:]

        inputs = tf.keras.Input(
            shape = shape, 
            name = 'input_layer_lstm' 
        )

        lstm = tf.keras.layers.LSTM(
            units = 64, 
            activation = 'tanh', 
            return_sequences = False, 
            name = 'lstm' 
        )(inputs)

        dense = tf.keras.layers.Dense(
            units = 200, 
            activation = 'linear', 
            name = 'dense_1' 
        )(lstm)

        leaky_re_lu = tf.keras.layers.LeakyReLU(
            name = 'leaky_re_lu' 
        )(dense)

        dense = tf.keras.layers.Dense(
            units = 200, 
            activation = 'elu', 
            name = 'dense_2' 
        )(leaky_re_lu)

        dense = tf.keras.layers.Dense(
            units = 200, 
            activation = 'relu', 
            name = 'dense_3' 
        )(dense)

        dense = tf.keras.layers.Dense(
            units = 200, 
            activation = 'elu', 
            name = 'dense_4' 
        )(dense)

        dense = tf.keras.layers.Dense(
            units = 1, 
            activation = 'gelu', 
            name = 'dense_5' 
        )(dense)

        lstm = tf.keras.Model(
            inputs = inputs, 
            outputs = dense, 
            name = 'lstm' 
        )

        return lstm



    def train(self, learning_rate : float, epochs : int, batch_size : int, X_train : np.array, y_train : np.array) -> None:

        self.cnnLSTM = tf.keras.Sequential(
            [self.cnn, self.lstm], 
            name = 'cnn-lstm' 
        )

        ## Print model summary

        print()
        print('------------------------------------------------------------------ Model Summary ------------------------------------------------------------------')
        print()

        print(self.cnn.summary())
        print()

        print(self.lstm.summary())
        print()

        print(self.cnnLSTM.summary())

        print()
        print('------------------------------------------------------------------ Model Summary ------------------------------------------------------------------')
        print()

        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        self.cnnLSTM.compile(optimizer = optimizer, loss = loss)

        ## Print training summary

        print()
        print('---------------------------------------------------------------- Training Summary -----------------------------------------------------------------')
        print()

        start = time.time()
        history = self.cnnLSTM.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        end = time.time()
        print()

        print(f'Minimum training loss: {np.min(history.history["loss"]):.4f}.')
        print(f'Time elapsed: {end - start:.2f} seconds.')

        print()
        print('---------------------------------------------------------------- Training Summary -----------------------------------------------------------------')
        print()



    def eval(self, X_test : np.array, y_test : np.array) -> np.array:

        ## Print testing summary

        print()
        print('----------------------------------------------------------------- Testing Summary -----------------------------------------------------------------')
        print()

        results = self.cnnLSTM.evaluate(X_test, y_test)
        print()

        print(f'Testing loss: {results:.4f}')
        print()

        predictions = self.cnnLSTM.predict(X_test)
        print()

        print(f'Five arbitrary predictions: {predictions[0:5].flatten()}')
        print(f'Actual labels:              {y_test[0:5].flatten()}')

        print()
        print('----------------------------------------------------------------- Testing Summary -----------------------------------------------------------------')
        print()

        return predictions