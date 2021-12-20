import keras
import tensorflow as tf
import keras.backend as K
from keras.activations import tanh, softmax
from keras.layers import LSTM, Dense, Layer, Lambda


class PointerAttention(Layer):
    '''
    https://www.tensorflow.org/text/tutorials/nmt_with_attention
    '''
    def __init__(self, units, **kwargs):
        super(PointerAttention, self).__init__(**kwargs)
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)
        self.supports_masking = True

    def _extract_context_vector(self, value, weights):
        idx = K.expand_dims( K.argmax(weights) )
        r = K.expand_dims( K.arange(idx.shape[0], dtype=idx.dtype) )
        indices = K.concatenate( [r, idx] )
        return tf.gather_nd(value, indices)
        
    def call(self, value, query, mask=None):
        w1_key = self.W1(value)
        w2_query = self.W2(query)
        w2_query = K.repeat(w2_query, w1_key.shape[1])

        u = self.V( tanh(w1_key + w2_query) )
        u = K.squeeze(u, axis=2)

        if mask is not None:
            mask_values = K.cast(mask[0], u.dtype)
            u += (1-mask_values) * K.constant(-1e20) # -np.infty
                
        a = softmax(u, axis=1)
        context = self._extract_context_vector(value, a)
        return context, a


class PointerDecoder(Layer):
    '''
    The cell abstraction, together with the generic keras.layers.RNN class, make it very easy to implement custom RNN architectures for your research.
    https://www.tensorflow.org/guide/keras/rnn

    https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    https://hyperscience.com/tech-blog/power-of-pointer-networks/
    '''
    def __init__(self, units, output_size, **kwargs):
        super(PointerDecoder, self).__init__(**kwargs)
        self.units = units
        self.output_size = output_size
        self.lstm = LSTM(self.units, return_sequences=True, return_state=True)
        self.attention = self.attention = PointerAttention(units)
        self.supports_masking = True

    def call(self, enc_outputs, initial_state=None, mask=None, *args, **kwargs):
        probs_outputs = []
        # Use the last state of the encoder as the first inputs and use its states as initial states
        inputs = enc_outputs[:, -1:]
        states = initial_state
        for _ in range(self.output_size):
            # Run the decoder on one timestep
            outputs, state_h, state_c = self.lstm(inputs, initial_state=states)

            # Query with the hidden state and store the current probs
            context, probs = self.attention(enc_outputs, state_h, mask)
            probs = K.expand_dims(probs, axis=1)
            probs_outputs.append(probs)

            # Reinject the pointed context as inputs for the next timestep and update the state
            inputs = K.expand_dims(context, axis=1)
            states = [state_h, state_c]

        # Concatenate all probs
        concat = Lambda(lambda x: K.concatenate(x, axis=1))
        outputs = concat(probs_outputs)
        # outputs = K.concatenate(probs_outputs, axis=1)
        return outputs

