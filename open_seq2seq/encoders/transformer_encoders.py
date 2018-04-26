# Copyright (c) 2018 NVIDIA Corporation
"""
Encoders based on Transformers arch from https://arxiv.org/abs/1706.03762
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import math
from .encoder import Encoder
from open_seq2seq.parts.attention import multi_head_attention_fn
from open_seq2seq.parts.common import ffn, \
                                      embed_and_maybe_add_position_signal, \
                                      normalize


class TransformerEncoder(Encoder):
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'ffn_inner_dim': int,
      'd_model': int,
      'attention_heads': int,
      'src_vocab_size': int,
      'encoder_layers': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'encoder_drop_prob': float,
      "encoder_norm_type": str,
    })

  def __init__(self, params,
               model,
               name="transformer_encoder",
               mode='train'):
    """
    Initializes uni-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * src_vocab_size - data vocabulary size
      * d_model - size of embedding to use
      * time_major (optional)
      * mode - train or infer
      ... add any cell-specific parameters here as well
    """
    super(TransformerEncoder, self).__init__(
      params, model, name=name, mode=mode,
    )

    self._drop_prob = self.params.get("encoder_drop_prob", 0.0)
    self._norm_type = self.params.get("encoder_norm_type", 'layer_norm')
    if self._mode != 'train':
      self._drop_prob = 0.0

  def _encode(self, input_dict):
    ffn_inner_dim = self.params["ffn_inner_dim"]
    d_model = self.params['d_model']
    attention_heads = self.params["attention_heads"]
    enc_emb_w = tf.get_variable(name="EncoderEmbeddingMatrix",
                                shape=[
                                  self.params['src_vocab_size'],
                                  self.params['d_model']],
                                dtype=self.params['dtype'])
    if self._mode == 'train':
      training= True
      drop_prob = self._drop_prob
    else:
      training = False
      drop_prob = 0.0
    keep_prob = 1.0 - drop_prob

    x, bias = embed_and_maybe_add_position_signal(
      inputs=input_dict['src_sequence'],
      emb_W=enc_emb_w,
      num_timescales=int(d_model/2),
      heads=attention_heads,
      drop_prob=drop_prob,
      scale=1.0)#math.sqrt(d_model))

    for block_ind in range(self.params['encoder_layers']):
      with tf.variable_scope("EncoderBlock_{}".format(block_ind)):
        # self-attention
        with tf.variable_scope("SelfAttention"):
          x_norm = normalize(x, training=training,
                             norm_type=self._norm_type) # preprocess
          y = multi_head_attention_fn(Q=x_norm, K=x_norm, V=x_norm,
                                      d_model=d_model,
                                      h=attention_heads,
                                      additional_bias=bias)
          x = tf.nn.dropout(x=y, keep_prob=keep_prob) + x # postprocess
        with tf.variable_scope("FFN_And_Drop"):
          x_norm = normalize(x, training=training,
                             norm_type=self._norm_type)  # preprocess
          y = ffn(inputs=x_norm, inner_dim=ffn_inner_dim,
                  resulting_dim=d_model, drop_prob=drop_prob)
          x = tf.nn.dropout(x=y, keep_prob=keep_prob) + x

    outputs = normalize(x, training=training, norm_type=self._norm_type)
    return {'outputs': outputs,
            'state': None,
            'src_lengths': input_dict['src_length'],
            'enc_emb_w': enc_emb_w,
            'encoder_input': input_dict['src_sequence']}
