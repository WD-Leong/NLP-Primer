import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization, Embedding, DepthwiseConv2D)

def scaled_dot_product_attention(
    q, k, v, mask=None, neg_infty=-1.0e9):
    # Head dimension. #
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    lq = tf.shape(q)[2]
    lk = tf.shape(k)[2]
    
    # Multiplicative Attention. #
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale multiplicative attention mechanism. #
    attn_logits = matmul_qk * tf.math.rsqrt(dk)
    
    # Add the mask to the attention mechanism. #
    if mask is not None:
        attn_mask = (mask * neg_infty)
    else:
        attn_mask = tf.zeros([lq, lk])
    attn_logits += attn_mask
    
    attn_weights = tf.nn.softmax(attn_logits, axis=-1)
    attn_outputs = tf.matmul(attn_weights, v)
    return attn_outputs, attn_weights

# RMS Normalisation Layer. #
class RMSNormalisation(tf.keras.layers.Layer):
    def __init__(self, eps=1.0e-6):
        super(RMSNormalisation, self).__init__()
        self.epsilon = eps
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            "kernel", shape=(input_shape[-1]), 
            initializer="ones", trainable=True)

    def call(self, x):
        eps = tf.constant(self.epsilon)
        x_var = tf.reduce_mean(
            tf.square(x), axis=[-1], keepdims=True)
        x_out = tf.multiply(
            x * tf.math.rsqrt(x_var + eps), self.scale)
        return x_out

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, depth_ker=3):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        self.depth_ker = (1, depth_ker)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)

        self.depth_cnn_q = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
        self.depth_cnn_k = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
        self.depth_cnn_v = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def call(self, v, k, q, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Depthwise CNN Layer. Pad the input #
        # to the left to ensure causality.   #
        batch_size = tf.shape(q)[0]
        zero_shape = [
            batch_size, self.depth_ker[1]-1, self.d_model]
        x_zero_pad = tf.zeros(
            zero_shape, dtype=tf.float32, name="zero_pad")
        
        depth_input_q = tf.expand_dims(
            tf.concat([x_zero_pad, q], axis=1), axis=1)
        depth_input_k = tf.expand_dims(
            tf.concat([x_zero_pad, k], axis=1), axis=1)
        depth_input_v = tf.expand_dims(
            tf.concat([x_zero_pad, v], axis=1), axis=1)
        
        q_in = self.split_heads(tf.squeeze(
            self.depth_cnn_q(depth_input_q), axis=1))
        k_in = self.split_heads(tf.squeeze(
            self.depth_cnn_k(depth_input_k), axis=1))
        v_in = self.split_heads(tf.squeeze(
            self.depth_cnn_v(depth_input_v), axis=1))
        
        attn_tuple = scaled_dot_product_attention(
            q_in, k_in, v_in, mask=mask, neg_infty=-1.0e9)
        
        attn_wgt = attn_tuple[1]
        attn_out = self.wc(
            self.combine_heads(attn_tuple[0]))
        return attn_out, attn_wgt
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        # Use Square ReLU. #
        return self.ffwd_2(tf.square(self.ffwd_1(x)))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        center=False, depth_ker=3, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.d_model = d_model
        self.depth_ker = (1, depth_ker)
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, depth_ker=depth_ker)
        
        self.center = center
        if center:
            self.lnorm_1 = LayerNormalization(epsilon=1.0e-6)
            self.lnorm_2 = LayerNormalization(epsilon=1.0e-6)
        else:
            self.lnorm_1 = RMSNormalisation(eps=1.0e-6)
            self.lnorm_2 = RMSNormalisation(eps=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, training=True, mask=None):
        batch_size = tf.shape(x_enc)[0]
        
        x_embed = x_enc + x_pos
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=mask)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_tuple[0], training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        # Feed Forward Layer. #
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, 
        d_ffwd, vocab_size, max_seq_length, 
        center=False, depth_ker=3, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.center = center
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.depth_ker = depth_ker
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        self.dec_embed = Embedding(vocab_size, d_model)
        self.pos_embed = [Embedding(
            max_seq_length, d_model) for _ in range(n_layers)]
        
        # Decoder Layers. #
        self.dec_layers = [DecoderLayer(
            d_model, n_heads, d_ffwd, 
            center=center, depth_ker=depth_ker, 
            rate1=rate1, rate2=rate2) for _ in range(n_layers)]
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        input_mask = tf.linalg.band_part(
            tf.ones([seq_length, seq_length]), -1, 0)
        input_mask = 1.0 - input_mask
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, 
                training=training, mask=input_mask)
            layer_input  = layer_output
        return layer_output

class PrimerDecoder(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, d_ffwd, 
        vocab_size, max_seq_length, center=False, 
        depth_ker=3, rate1=0.1, rate2=0.1):
        super(PrimerDecoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.center = center
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.depth_ker = depth_ker
        self.vocab_size = vocab_size
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, d_model, n_heads, d_ffwd, 
            vocab_size, max_seq_length, center=center, 
            depth_ker=depth_ker, rate1=rate1, rate2=rate2)
        self.p_decoder = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=True):
        dec_outputs = self.gpt_model(
            x, training=training)
        dec_logits  = self.p_decoder(dec_outputs)
        return dec_logits
    
    def infer(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
        

