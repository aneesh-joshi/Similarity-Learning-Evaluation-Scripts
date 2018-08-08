from keras.engine.topology import Layer
import keras.backend as K
from keras.layers import Input
import numpy as np

"""Script where all the custom keras layers are kept."""


class TopKLayer(Layer):
    """Layer to get top k values from the interaction matrix in drmm_tks model"""
    def __init__(self, output_dim, topk, **kwargs):
        """

        Parameters
        ----------
        output_dim : tuple of int
            The dimension of the tensor after going through this layer.
        topk : int
            The k topmost values to be returned.
        """
        self.output_dim = output_dim
        self.topk = topk
        super(TopKLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TopKLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.nn.top_k(x, k=self.topk, sorted=True)[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.topk)

    def get_config(self):
        config = {
            'topk': self.topk,
            'output_dim': self.output_dim
        }
        base_config = super(TopKLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicMaxPooling(Layer):
    def __init__(self, psize1, psize2, **kwargs):
        self.psize1 = psize1
        self.psize2 = psize2
        super(DynamicMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_one = input_shape[0]
        self.msize1 = input_shape_one[1]
        self.msize2 = input_shape_one[2]
        super(DynamicMaxPooling, self).build(input_shape)  

    def get_config(self):
        config = {
            'psize1': self.psize1,
            'psize2': self.psize2
        }
        base_config = super(DynamicMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def call(self, data):
        x, self.dpool_index = data
        x_expand = K.tf.gather_nd(x, self.dpool_index)
        stride1 = self.msize1 / self.psize1
        stride2 = self.msize2 / self.psize2
        
        suggestion1 = self.msize1 / stride1
        suggestion2 = self.msize2 / stride2

        if suggestion1 != self.psize1 or suggestion2 != self.psize2:
            print("DynamicMaxPooling Layer can not "
                  "generate ({} x {}) output feature map,"
                  "please use ({} x {} instead.)".format(self.psize1, self.psize2, 
                                                       suggestion1, suggestion2))
            exit()

        x_pool = K.tf.nn.max_pool(x_expand, 
                    [1, self.msize1 / self.psize1, self.msize2 / self.psize2, 1], 
                    [1, self.msize1 / self.psize1, self.msize2 / self.psize2, 1], 
                    "VALID")
        return x_pool

    def compute_output_shape(self, input_shape):
        input_shape_one = input_shape[0]
        return (None, self.psize1, self.psize2, input_shape_one[3])

    @staticmethod
    def dynamic_pooling_index(len1, len2, max_len1, max_len2,
                              compress_ratio1 = 1, compress_ratio2 = 1):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            '''
            TODO: Here is the check of sentences length to be positive.
            To make sure that the lenght of the input sentences are positive. 
            if len1_one == 0:
                print("[Error:DynamicPooling] len1 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            if len2_one == 0:
                print("[Error:DynamicPooling] len2 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            '''
            if len1_one == 0:
                stride1 = max_len1
            else:
                stride1 = 1.0 * max_len1 / len1_one

            if len2_one == 0:
                stride2 = max_len2
            else:
                stride2 = 1.0 * max_len2 / len2_one

            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx,
                                      mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        dpool_bias1 = dpool_bias2 = 0
        if max_len1 % compress_ratio1 != 0:
            dpool_bias1 = 1
        if max_len2 % compress_ratio2 != 0:
            dpool_bias2 = 1
        cur_max_len1 = max_len1 // compress_ratio1 + dpool_bias1
        cur_max_len2 = max_len2 // compress_ratio2 + dpool_bias2
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i] // compress_ratio1, 
                         len2[i] // compress_ratio2, cur_max_len1, cur_max_len2))
        return np.array(index)

from keras.layers import Highway as KerasHighway
from keras.layers import Layer

class Highway(KerasHighway):
    """
    Keras' `Highway` layer does not support masking, but it easily could, just by returning the
    mask.  This `Layer` makes this possible.
    """
    def __init__(self, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.supports_masking = True


class MaskedLayer(Layer):
    """
    Keras 2.0 allowed for arbitrary differences in arguments to the ``call`` method of ``Layers``.
    As part of this, they removed the default ``mask=None`` argument, which means that if you want
    to implement ``call`` with a mask, you need to disable a pylint warning.  Instead of disabling
    it in every single layer in our codebase, which could lead to uncaught errors, we'll have a
    single place where we disable it, and have other layers inherit from this class.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):  # pylint: disable=arguments-differ
        raise NotImplementedError


from copy import deepcopy
from typing import Any, Dict

from keras import backend as K
from overrides import overrides
from .tensor_ops import CosineSimilarity

class MatrixAttention(MaskedLayer):
    '''
    This ``Layer`` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  We don't worry about zeroing out any masked values, because we propagate a correct
    mask.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    This is largely similar to using ``TimeDistributed(Attention)``, except the result is
    unnormalized, and we return a mask, so you can do a masked normalization with the result.  You
    should use this instead of ``TimeDistributed(Attention)`` if you want to compute multiple
    normalizations of the attention matrix.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``, with mask ``(batch_size, num_rows_1)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``, with mask ``(batch_size, num_rows_2)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``, with mask of same shape

    Parameters
    ----------
    similarity_function_params: Dict[str, Any], default={}
        These parameters get passed to a similarity function (see
        :mod:`deep_qa.tensors.similarity_functions` for more info on what's acceptable).  The
        default similarity function with no parameters is a simple dot product.
    '''
    def __init__(self, **kwargs):
        super(MatrixAttention, self).__init__(**kwargs)
        # self.similarity_function_params = deepcopy(similarity_function)
        # if similarity_function is None:
        #     similarity_function = {}
        # similarity_function['name'] = self.name + '_similarity_function'
        self.similarity_function = CosineSimilarity(name='kcosine_sim')

    @overrides
    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]
        self.trainable_weights = self.similarity_function.initialize_weights(tensor_1_dim, tensor_2_dim)
        super(MatrixAttention, self).build(input_shape)


    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    @overrides
    def call(self, inputs, mask=None):
        matrix_1, matrix_2 = inputs
        num_rows_1 = K.shape(matrix_1)[1]
        num_rows_2 = K.shape(matrix_2)[1]
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0)
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0)
        tiled_matrix_1 = K.tile(K.expand_dims(matrix_1, axis=2), tile_dims_1)
        tiled_matrix_2 = K.tile(K.expand_dims(matrix_2, axis=1), tile_dims_2)
        return self.similarity_function.compute_similarity(tiled_matrix_1, tiled_matrix_2)

    @overrides
    def get_config(self):
        base_config = super(MatrixAttention, self).get_config()
        config = {'similarity_function': self.similarity_function_params}
        config.update(base_config)
        return config

def last_dim_flatten(input_tensor):
    '''
    Takes a tensor and returns a matrix while preserving only the last dimension from the input.
    '''
    input_ndim = K.ndim(input_tensor)
    shuffle_pattern = (input_ndim - 1,) + tuple(range(input_ndim - 1))
    dim_shuffled_input = K.permute_dimensions(input_tensor, shuffle_pattern)
    return K.transpose(K.batch_flatten(dim_shuffled_input))

from .tensor_ops import masked_softmax

class MaskedSoftmax(MaskedLayer):
    '''
    This Layer performs a masked softmax.  This could just be a `Lambda` layer that calls our
    `tensors.masked_softmax` function, except that `Lambda` layers do not properly handle masked
    input.

    The expected input to this layer is a tensor of shape `(batch_size, num_options)`, with a mask
    of the same shape.  We also accept an input tensor of shape `(batch_size, num_options, 1)`,
    which we will squeeze to be `(batch_size, num_options)` (though the mask must still be
    `(batch_size, num_options)`).

    While we give the expected input as having two modes, we also accept higher-order tensors.  In
    those cases, we'll first perform a `last_dim_flatten` on both the input and the mask, so that
    we always do the softmax over a single dimension (the last one).

    We give no output mask, as we expect this to only be used at the end of the model, to get a
    final probability distribution over class labels (and it's a softmax, so you'll have zeros in
    the tensor itself; do you really still need a mask?).  If you need this to propagate the mask
    for whatever reason, it would be pretty easy to change it to optionally do so - submit a PR.
    '''
    def __init__(self, **kwargs):
        super(MaskedSoftmax, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def compute_output_shape(self, input_shape):
        if input_shape[-1] == 1:
            return input_shape[:-1]
        else:
            return input_shape

    @overrides
    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        if input_shape[-1] == 1:
            inputs = K.squeeze(inputs, axis=-1)
            input_shape = input_shape[:-1]
        if len(input_shape) > 2:
            original_inputs = inputs
            inputs = last_dim_flatten(inputs)
            if mask is not None:
                mask = last_dim_flatten(mask)
        # Now we have both inputs and mask with shape (?, num_options), and can do a softmax.
        softmax_result = masked_softmax(inputs, mask)
        if len(input_shape) > 2:
            original_shape = K.shape(original_inputs)
            input_shape = K.concatenate([[-1], original_shape[1:]], 0)
            softmax_result = K.reshape(softmax_result, input_shape)
        return softmax_result


class WeightedSum(MaskedLayer):
    """
    This ``Layer`` takes a matrix of vectors and a vector of row weights, and returns a weighted
    sum of the vectors.  You might use this to get some aggregate sentence representation after
    computing an attention over the sentence, for example.

    Inputs:

    - matrix: ``(batch_size, num_rows, embedding_dim)``, with mask ``(batch_size, num_rows)``
    - vector: ``(batch_size, num_rows)``, mask is ignored

    Outputs:

    - A weighted sum of the rows in the matrix, with shape ``(batch_size, embedding_dim)``, with
      mask=``None``.

    Parameters
    ----------
    use_masking: bool, default=True
        If true, we will apply the input mask to the matrix before doing the weighted sum.  If
        you've computed your vector weights with masking, so that masked entries are 0, this is
        unnecessary, and you can set this parameter to False to avoid an expensive computation.

    Notes
    -----
    You probably should have used a mask when you computed your attention weights, so any row
    that's masked in the matrix `should` already be 0 in the attention vector.  But just in case
    you didn't, we'll handle a mask on the matrix here too.  If you know that you did masking right
    on the attention, you can optionally remove the mask computation here, which will save you a
    bit of time and memory.

    While the above spec shows inputs with 3 and 2 modes, we also allow inputs of any order; we
    always sum over the second-to-last dimension of the "matrix", weighted by the last dimension of
    the "vector".  Higher-order tensors get complicated for matching things, though, so there is a
    hard constraint: all dimensions in the "matrix" before the final embedding must be matched in
    the "vector".

    For example, say I have a "matrix" with dimensions (batch_size, num_queries, num_words,
    embedding_dim), representing some kind of embedding or encoding of several multi-word queries.
    My attention "vector" must then have at least those dimensions, and could have more.  So I
    could have an attention over words per query, with shape (batch_size, num_queries, num_words),
    or I could have an attention over query words for every document in some list, with shape
    (batch_size, num_documents, num_queries, num_words).  Both of these cases are fine.  In the
    first case, the returned tensor will have shape (batch_size, num_queries, embedding_dim), and
    in the second case, it will have shape (batch_size, num_documents, num_queries, embedding_dim).
    But you `can't` have an attention "vector" that does not include all of the queries, so shape
    (batch_size, num_words) is not allowed - you haven't specified how to handle that dimension in
    the "matrix", so we can't do anything with this input.
    """
    def __init__(self, use_masking: bool=True, **kwargs):
        self.use_masking = use_masking
        super(WeightedSum, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We don't need to worry about a mask after we've summed over the rows of the matrix.
        # You might actually still need a mask if you used a higher-order tensor, but probably the
        # right place to handle that is with careful use of TimeDistributed.  Or submit a PR.
        return None

    @overrides
    def compute_output_shape(self, input_shapes):
        matrix_shape, attention_shape = input_shapes
        return attention_shape[:-1] + matrix_shape[-1:]

    @overrides
    def call(self, inputs, mask=None):
        # pylint: disable=redefined-variable-type
        matrix, attention_vector = inputs
        num_attention_dims = K.ndim(attention_vector)
        num_matrix_dims = K.ndim(matrix) - 1
        for _ in range(num_attention_dims - num_matrix_dims):
            matrix = K.expand_dims(matrix, axis=1)
        if mask is None:
            matrix_mask = None
        else:
            matrix_mask = mask[0]
        if self.use_masking and matrix_mask is not None:
            for _ in range(num_attention_dims - num_matrix_dims):
                matrix_mask = K.expand_dims(matrix_mask, axis=1)
            matrix = K.cast(K.expand_dims(matrix_mask), 'float32') * matrix
        return K.sum(K.expand_dims(attention_vector, axis=-1) * matrix, -2)

    @overrides
    def get_config(self):
        base_config = super(WeightedSum, self).get_config()
        config = {'use_masking': self.use_masking}
        config.update(base_config)
        return config


class RepeatLike(MaskedLayer):
    """
    This ``Layer`` is like :class:`~.repeat.Repeat`, but gets the number of repetitions to use from
    a second input tensor.  This allows doing a number of repetitions that is unknown at graph
    compilation time, and is necessary when the ``repetitions`` argument to ``Repeat`` would be
    ``None``.

    If the mask is not ``None``, we must be able to call ``K.expand_dims`` using the same axis
    parameter as we do for the input.

    Input:
        - A tensor of arbitrary shape, which we will expand and tile.
        - A second tensor whose shape along one dimension we will copy

    Output:
        - The input tensor repeated along one of the dimensions.

    Parameters
    ----------
    axis: int
        We will add a dimension to the input tensor at this axis.
    copy_from_axis: int
        We will copy the dimension from the second tensor at this axis.
    """
    def __init__(self, axis: int, copy_from_axis: int, **kwargs):
        self.axis = axis
        self.copy_from_axis = copy_from_axis
        super(RepeatLike, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        return None

    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape[0][:self.axis] + (input_shape[1][self.copy_from_axis],) + input_shape[0][self.axis:]

    @overrides
    def call(self, inputs, mask=None):
        return self.__repeat_tensor(inputs[0], inputs[1])

    def __repeat_tensor(self, to_repeat, to_copy):
        expanded = K.expand_dims(to_repeat, self.axis)
        ones = [1] * K.ndim(expanded)
        num_repetitions = K.shape(to_copy)[self.copy_from_axis]
        tile_shape = K.concatenate([ones[:self.axis], [num_repetitions], ones[self.axis+1:]], 0)
        return K.tile(expanded, tile_shape)

    @overrides
    def get_config(self):
        base_config = super(RepeatLike, self).get_config()
        config = {'axis': self.axis, 'copy_from_axis': self.copy_from_axis}
        config.update(base_config)
        return config


VERY_LARGE_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_NEGATIVE_NUMBER = -VERY_LARGE_NUMBER

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


def very_negative_like(tensor):
    return K.ones_like(tensor) * VERY_NEGATIVE_NUMBER

class Max(MaskedLayer):
    """
    This ``Layer`` performs a max over some dimension.  Keras has a similar layer called
    ``GlobalMaxPooling1D``, but it is not as configurable as this one, and it does not support
    masking.

    If the mask is not ``None``, it must be the same shape as the input.

    Input:
        - A tensor of arbitrary shape (having at least 3 dimensions).

    Output:
        - A tensor with one less dimension, where we have taken a max over one of the dimensions.
    """
    def __init__(self, axis: int=-1, **kwargs):
        self.axis = axis
        super(Max, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.any(mask, axis=self.axis)

    @overrides
    def compute_output_shape(self, input_shape):
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)
        return input_shape[:axis] + input_shape[axis+1:]

    @overrides
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = switch(mask, inputs, very_negative_like(inputs))
        return K.max(inputs, axis=self.axis)

    @overrides
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Max, self).get_config()
        config.update(base_config)
        return config



from typing import List, Tuple

class ComplexConcat(MaskedLayer):
    """
    This ``Layer`` does ``K.concatenate()`` on a collection of tensors, but
    allows for more complex operations than ``Merge(mode='concat')``.
    Specifically, you can perform an arbitrary number of elementwise linear
    combinations of the vectors, and concatenate all of the results.  If you do
    not need to do this, you should use the regular ``Merge`` layer instead of
    this ``ComplexConcat``.

    Because the inputs all have the same shape, we assume that the masks are
    also the same, and just return the first mask.

    Input:
        - A list of tensors.  The tensors that you combine **must** have the
          same shape, so that we can do elementwise operations on them, and
          all tensors must have the same number of dimensions, and match on
          all dimensions except the concatenation axis.

    Output:
        - A tensor with some combination of the input tensors concatenated
          along a specific dimension.

    Parameters
    ----------
    axis : int
        The axis to use for ``K.concatenate``.

    combination: List of str
        A comma-separated list of combinations to perform on the input tensors.
        These are either tensor indices (1-indexed), or an arithmetic
        operation between two tensor indices (valid operations: ``*``, ``+``,
        ``-``, ``/``).  For example, these are all valid combination
        parameters: ``"1,2"``, ``"1,2*3"``, ``"1-2,2-1"``, ``"1,1*1"``,
        and ``"1,2,1*2"``.
    """
    def __init__(self, combination: str, axis: int=-1, **kwargs):
        self.axis = axis
        self.combination = combination
        self.combinations = self.combination.split(",")
        self.num_combinations = len(self.combinations)
        super(ComplexConcat, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return None

    @overrides
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("ComplexConcat input must be a list")
        output_shape = list(input_shape[0])
        output_shape[self.axis] = 0
        for combination in self.combinations:
            output_shape[self.axis] += self._get_combination_length(combination, input_shape)
        return tuple(output_shape)

    @overrides
    def call(self, x, mask=None):
        combined_tensor = self._get_combination(self.combinations[0], x)
        for combination in self.combinations[1:]:
            to_concatenate = self._get_combination(combination, x)
            combined_tensor = K.concatenate([combined_tensor, to_concatenate], axis=self.axis)
        return combined_tensor

    def _get_combination(self, combination: str, tensors: List['Tensor']):
        if combination.isdigit():
            return tensors[int(combination) - 1]  # indices in the combination string are 1-indexed
        else:
            if len(combination) != 3:
                raise ValueError("Invalid combination: " + combination)
            first_tensor = self._get_combination(combination[0], tensors)
            second_tensor = self._get_combination(combination[2], tensors)
            if K.int_shape(first_tensor) != K.int_shape(second_tensor):
                shapes_message = "Shapes were: {} and {}".format(K.int_shape(first_tensor),
                                                                 K.int_shape(second_tensor))
                raise ValueError("Cannot combine two tensors with different shapes!  " +
                                         shapes_message)
            operation = combination[1]
            if operation == '*':
                return first_tensor * second_tensor
            elif operation == '/':
                return first_tensor / second_tensor
            elif operation == '+':
                return first_tensor + second_tensor
            elif operation == '-':
                return first_tensor - second_tensor
            else:
                raise ValueError("Invalid operation: " + operation)

    def _get_combination_length(self, combination: str, input_shapes: List[Tuple[int]]):
        if combination.isdigit():
            # indices in the combination string are 1-indexed
            return input_shapes[int(combination) - 1][self.axis]
        else:
            if len(combination) != 3:
                raise ValueError("Invalid combination: " + combination)
            first_length = self._get_combination_length(combination[0], input_shapes)
            second_length = self._get_combination_length(combination[2], input_shapes)
            if first_length != second_length:
                raise ValueError("Cannot combine two tensors with different shapes!")
            return first_length

    @overrides
    def get_config(self):
        config = {"combination": self.combination,
                  "axis": self.axis,
                 }
        base_config = super(ComplexConcat, self).get_config()
        config.update(base_config)
        return config
