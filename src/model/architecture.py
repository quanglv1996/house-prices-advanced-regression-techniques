import sys

sys.path.append('../')
from keras.layers import Flatten, Activation
from keras.layers import Dropout, Input, Conv1D, MaxPooling1D, Reshape
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
#from tcn import TCN


class Architecture():

    def __init__(self, input_shape):
        self.input_shape = input_shape


    # Dense -> BatchNorm ->Activation
    def dense_batchnorm_activation(self, input, dense_units, activation='relu'):
        x = Dense(dense_units)(input)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x


    '''For LSTM'''

    # LSTM block
    def block_lstm(self, input, units, dropout=0.3, bidirectional=True, return_sequences=True):
        if bidirectional == True:
            x = Bidirectional(LSTM(units, return_sequences=return_sequences, dropout=dropout, activation='tanh'))(input)
        else:
            x = LSTM(units, return_sequences=return_sequences, dropout=dropout, activation='tanh')(input)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        return x

    def block_gru(self, input, units, bidirectional=True, return_sequences=True, dropout=0.3):
        if bidirectional == True:
            x = Bidirectional(GRU(units, return_sequences=return_sequences, dropout=dropout, activation='tanh'))(input)
        else:
            x = GRU(units, return_sequences=return_sequences, dropout=dropout, activation='tanh')(input)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        return x

    # LSTM model
    def lstm_model(self, lstm_units, dense_units, bidirectional=False):
        # Input shape
        input_model = Input(self.input_shape)
        x = BatchNormalization()(input_model)

        # Block 1
        x = self.block_lstm(input=x, units=lstm_units, bidirectional=bidirectional, return_sequences=True)

        # Block 2
        x = self.block_lstm(input=x, units=lstm_units, bidirectional=bidirectional, return_sequences=False)

        # Dense 1
        x = self.dense_batchnorm_activation(input=x, dense_units=dense_units)
        x = Dropout(0.3)(x)

        # Dense 2
        x = self.dense_batchnorm_activation(input=x, dense_units=int(dense_units / 2))
        x = Dropout(0.3)(x)

        # Dense 3
        x = self.dense_batchnorm_activation(input=x, dense_units=int(dense_units / 4))

        # Output
        output = Dense(1, activation='sigmoid')(x)

        # Compare input and output
        model = Model(inputs=input_model, outputs=output)
        model.summary()

        return model

    # GRU model
    def gru_model(self, gru_units, dense_units, bidirectional=False):
        # Input shape
        input_model = Input(self.input_shape)
        x = BatchNormalization()(input_model)

        # Block 1
        x = self.block_gru(input=x, units=gru_units, bidirectional=bidirectional, return_sequences=True)

        # Block 2
        x = self.block_gru(input=x, units=gru_units, bidirectional=bidirectional, return_sequences=False)

        # Dense 1
        x = self.dense_batchnorm_activation(input=x, dense_units=dense_units)
        x = Dropout(0.3)(x)

        # Dense 2
        x = self.dense_batchnorm_activation(input=x, dense_units=int(dense_units / 2))
        x = Dropout(0.3)(x)

        # Dense 3
        x = self.dense_batchnorm_activation(input=x, dense_units=int(dense_units / 4))
        x = Dropout(0.3)(x)

        # Output
        output = Dense(1, activation='sigmoid')(x)

        # Compare input and output
        model = Model(inputs=input_model, outputs=output)
        model.summary()

        return model



