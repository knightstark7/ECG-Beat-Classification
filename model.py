import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, LayerNormalization, BatchNormalization, GlobalAveragePooling1D, Dense, Add, Flatten
from tensorflow.keras.models import Model
import numpy as np

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, dropout_rate, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = tf.keras.layers.Dense(dff, activation='relu')

    def call(self, inputs, training=True):
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.layer_norm(inputs + attention_output)

        ffn_output = self.dense(attention_output)
        ffn_output = self.dropout(ffn_output, training=training)
        encoder_output = self.layer_norm(attention_output + ffn_output)

        return encoder_output

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

# Channel Attention Layer
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = tf.keras.layers.Dense(self.filters // self.ratio,
                                                      activation='relu', kernel_initializer='he_normal',
                                                      use_bias=True,
                                                      bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(self.filters,
                                                      kernel_initializer='he_normal',
                                                      use_bias=True,
                                                      bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling1D()(inputs)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        avg_pool = tf.keras.layers.Reshape((1, self.filters))(avg_pool)
        max_pool = tf.keras.layers.Reshape((1, self.filters))(max_pool)

        attention = tf.keras.layers.Add()([avg_pool, max_pool])
        attention = tf.keras.layers.Activation('sigmoid')(attention)

        attention = tf.keras.layers.Reshape((1, self.filters))(attention)

        return tf.keras.layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config

def build_model(max_sequence_length, num_channels, d_model, num_heads, dff, dropout_rate):
    input_sequence = Input(shape=(max_sequence_length, num_channels))

    x = Conv1D(filters=16, kernel_size=21, strides=1, padding='same', activation='relu')(input_sequence)
    x = BatchNormalization()(x)
    x = ChannelAttention(16, 8)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Conv1D(filters=32, kernel_size=23, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = ChannelAttention(32, 8)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Conv1D(filters=64, kernel_size=25, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = ChannelAttention(64, 8)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Conv1D(filters=128, kernel_size=27, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = ChannelAttention(128, 8)(x)

    # Calculate the new sequence length after convolutional and pooling layers
    new_sequence_length = max_sequence_length
    for _ in range(3):  # 3 pooling layers with pool_size=2
        new_sequence_length = (new_sequence_length + 1) // 2

    def positional_encoding(seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return tf.constant(P, dtype=tf.float32)

    positional_encoding = positional_encoding(seq_len=new_sequence_length, d=d_model)
    positional_encoding = tf.expand_dims(positional_encoding, axis=0)  # Shape (1, new_sequence_length, d_model)
    x = Add()([x, positional_encoding])

    query = x
    value = x

    transformer_encoder = TransformerEncoder(num_heads=num_heads, d_model=d_model, dff=dff, dropout_rate=dropout_rate)
    encoder_output = transformer_encoder(query)

    x = Flatten()(encoder_output)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    output = Dense(units=5, activation='softmax')(x)

    model = Model(inputs=input_sequence, outputs=output)

    return model
