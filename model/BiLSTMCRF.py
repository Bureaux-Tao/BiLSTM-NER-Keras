import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import regularizers, optimizers
from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, TimeDistributed
from keras_contrib.layers import CRF


class BILSTMCRF():
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.2,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate
    
    def creat_model(self):
        inputs = Input(shape = (None,))
        x = Embedding(
            input_dim = self.vocab_size,
            output_dim = self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(
            units = self.rnn_units,
            return_sequences = True))(x)
        x = Dropout(self.drop_rate)(x)
        x = Bidirectional(LSTM(
            units = int(self.rnn_units / 2),
            return_sequences = True))(x)
        x = Dropout(self.drop_rate)(x)
        x = TimeDistributed(Dense(
            self.n_class))(x)
        x = Dropout(self.drop_rate)(x)
        self.crf = CRF(self.n_class)
        x = self.crf(x)
        self.model = Model(inputs = inputs, outputs = x)
        self.model.summary()
        self.compile()
        return self.model
    
    def compile(self):
        self.model.compile("adam",
                           loss = self.crf.loss_function,
                           metrics = [self.crf.accuracy])