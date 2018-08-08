"""
Similarity functions take a pair of tensors with the same shape, and compute a similarity function
on the vectors in the last dimension.  For example, the tensors might both have shape
`(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
tensor of shape `(batch_size, sentence_length)`.

The similarity function could be as simple as a dot product, or it could be a more complex,
parameterized function.  The SimilarityFunction class exposes an API for a Layer that wants to
allow for multiple similarity functions, such as for initializing and returning weights.

If you want to compute a similarity between tensors of different sizes, you need to first tile them
in the appropriate dimensions to make them the same before you can use these functions.  The
Attention and MatrixAttention layers do this.
"""

from typing import List

from keras import backend as K
from overrides import overrides

from typing import List

from keras import activations, initializers
import tensorflow as tf
def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only
    accepts scalar value conditions, rather than boolean tensors which are treated in an
    elementwise function.  This doesn't match with Theano's implementation of switch, but using
    tensorflow's where, we can exactly retrieve this functionality.
    """

    cond_shape = cond.get_shape()
    input_shape = then_tensor.get_shape()
    if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
        # This happens when the last dim in the input is an embedding dimension. Keras usually does not
        # mask the values along that dimension. Theano broadcasts the value passed along this dimension,
        # but TF does not. Using K.dot() since cond can be a tensor.
        cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
    return tf.where(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)

class SimilarityFunction:
    def __init__(self, name: str, initialization: str='glorot_uniform', activation: str='linear'):
        self.name = name
        self.init = initializers.get(initialization)
        self.activation = activations.get(activation)

    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        """
        Called in a `Layer.build()` method that uses this SimilarityFunction, here we both
        initialize whatever weights are necessary for this similarity function, and return them so
        they can be included in `Layer.trainable_weights`.


        Parameters
        ----------
        tensor_1_dim : int
            The last dimension (typically ``embedding_dim``) of the first input tensor.  We need
            this so we can initialize weights appropriately.
        tensor_2_dim : int
            The last dimension (typically ``embedding_dim``) of the second input tensor.  We need
            this so we can initialize weights appropriately.
        """
        raise NotImplementedError

    def compute_similarity(self, tensor_1, tensor_2):
        """
        Takes two tensors of the same shape, such as (batch_size, length_1, length_2,
        embedding_dim).  Computes a (possibly parameterized) similarity on the final dimension and
        returns a tensor with one less dimension, such as (batch_size, length_1, length_2).
        """
        raise NotImplementedError

class CosineSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors.  It has
    no parameters.
    """
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

    @overrides
    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        if tensor_1_dim != tensor_2_dim:
            raise ValueError("Tensor dims must match for cosine product similarity, but "
                                     "were {} and {}".format(tensor_1_dim, tensor_2_dim))
        return []

    @overrides
    def compute_similarity(self, tensor_1, tensor_2):
        return K.sum(K.l2_normalize(tensor_1, axis=-1) * K.l2_normalize(tensor_2, axis=-1),
                     axis=-1)

def masked_softmax(vector, mask):
    """
    `K.softmax(vector)` does not work if some elements of `vector` should be masked.  This performs
    a softmax on just the non-masked portions of `vector` (passing None in for the mask is also
    acceptable; you'll just get a regular softmax).

    We assume that both `vector` and `mask` (if given) have shape (batch_size, vector_dim).

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorial cross-entropy loss.
    """
    # We calculate masked softmax in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    if mask is not None:
        # Here we get normalized log probabilities for
        # enhanced numerical stability.
        mask = K.cast(mask, "float32")
        input_masked = mask * vector
        shifted = mask * (input_masked - K.max(input_masked, axis=1,
                                               keepdims=True))
        # We add epsilon to avoid numerical instability when
        # the sum in the log yields 0.
        normalization_constant = K.log(K.sum(mask * K.exp(shifted), axis=1,
                                             keepdims=True) + K.epsilon())
        normalized_log_probabilities = mask * (shifted - normalization_constant)
        unmasked_probabilities = K.exp(normalized_log_probabilities)
        return switch(mask, unmasked_probabilities, K.zeros_like(unmasked_probabilities))
    else:
        # There is no mask, so we use the provided ``K.softmax`` function.
        return K.softmax(vector)
