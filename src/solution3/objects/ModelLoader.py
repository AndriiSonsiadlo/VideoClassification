"""
A collection of models we'll use to attempt to classify videos.
"""
import sys
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Dropout, Flatten, GRU, Bidirectional, SimpleRNN
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.decomposition import PCA

class ModelLoader:
    def __init__(self, model_name, nb_classes, seq_length=40,
                 saved_model=None, input_features_length=2048):
        """
        `model` = one of:
            lstm
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        self.nb_classes = nb_classes

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if saved_model is not None:
            print("Loading model %s" % saved_model)
            self.model = load_model(saved_model)
        elif model_name == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, input_features_length)
            self.model = self.lstm()
        elif model_name == 'reversed_lstm':
            print("Loading reversed LSTM model.")
            self.input_shape = (seq_length, input_features_length)
            self.model = self.reversed_lstm()
        elif model_name == 'gru':
            print("Loading simple GRU.")
            self.input_shape = (seq_length, input_features_length)
            self.model = self.gru()
        elif model_name == 'rnn':
            print("Loading simple RNN.")
            self.input_shape = (seq_length, input_features_length)
            self.model = self.rnn()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(learning_rate=5e-4, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""

        model = Sequential()
        # pca = PCA(n_components=0.9)
        # model.add(pca)
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.6))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def reversed_lstm(self):

        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5, go_backwards=True))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.63))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def gru(self):
        model = Sequential()
        model.add(GRU(2048, input_shape=self.input_shape, return_sequences=False, dropout=0.5, recurrent_dropout=0.4))
        model.add(Bidirectional(GRU(512, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.6))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def rnn(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(SimpleRNN(1024, return_sequences=True, dropout=0.5, recurrent_dropout=0.4))
        model.add(SimpleRNN(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.4))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.6))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
