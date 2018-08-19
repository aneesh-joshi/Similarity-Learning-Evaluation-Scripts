from keras.layers import Embedding, Dense, Input, Concatenate
from keras.models import Model


class BaselineModel:
    '''A simple Baseline model which uses Dense/Fully Connected Neural Networks

    Parameters
    ----------
    vector_size : int
        The size of the word embeddings to represent words
    num_predictions : int
        The number of classes into which a prediction has to be made.
        For example, if there are 3 classes : neutral, entailment, contriadiction, num_predictions = 3
    optimizer : str
        The keras optimizer to be used while training the network.
    model_type : {'regression', 'multilayer'}

    '''
    def __init__(self, vector_size, num_predictions, optimizer='adam', model_type='regression'):
        self.vector_size = vector_size
        self.num_predictions = num_predictions
        self.optimizer = optimizer
        self.model = None
        self.model_type = model_type

    def train(self, X, y, n_epochs=3):
        '''Trains or retrains the model

        Parameters
        ----------
        X : numpy array
            The train data input features
        y : numpy array
            The train labels
        '''
        if self.model is None:
            self.model = self._get_model()
            self.model.compile(self.optimizer, 'categorical_crossentropy', metrics=['acc'])

        self.model.fit(X, y, epochs=n_epochs)

    def _get_model(self):
        '''Gets the keras model of the needed `model_type` '''
        input_vec1 = Input(shape=(self.vector_size,), name='x1')
        input_vec2 = Input(shape=(self.vector_size,), name='x2')
        concat_vec = Concatenate()([input_vec1, input_vec2])

        if self.model_type == 'regression':
            fc = Dense(self.num_predictions, activation='softmax')(concat_vec)
            model = Model([input_vec1, input_vec2], fc)
            model.summary()
            return model
        elif self.model_type == 'multilayer':
            fc1 = Dense(100, activation='relu')(concat_vec)
            fc2 = Dense(64, activation='relu')(fc1)
            fc3 = Dense(self.num_predictions, activation='softmax')(fc2)
            model = Model([input_vec1, input_vec2], fc3)
            model.summary()
            return model
        else:
            raise ValueError('Unknown model type %s' % str(self.model_type))
