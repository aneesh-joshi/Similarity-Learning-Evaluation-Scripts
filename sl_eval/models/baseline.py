from keras.layers import Embedding, Dense, Input, Concatenate
from keras.models import Model


class BaselineModel:
    def __init__(self, vector_size, num_predictions, optimizer='adam'):
        self.vector_size = vector_size
        self.num_predictions = num_predictions
        self.optimizer = optimizer
        self.model = None

    def train(self, X, y, n_epochs=3):
        if self.model is None:
            self.model = self._get_model()
            self.model.compile(self.optimizer, 'categorical_crossentropy', metrics=['acc'])

        self.model.fit(X, y, epochs=n_epochs)

    def _get_model(self):
        input_vec1 = Input(shape=(self.vector_size,), name='x1')
        input_vec2 = Input(shape=(self.vector_size,), name='x2')
        concat_vec = Concatenate()([input_vec1, input_vec2])

        fc1 = Dense(100, activation='relu')(concat_vec)
        fc2 = Dense(64, activation='relu')(fc1)
        fc3 = Dense(self.num_predictions, activation='softmax')(fc2)

        model = Model([input_vec1, input_vec2], fc3)
        model.summary()
        return model
