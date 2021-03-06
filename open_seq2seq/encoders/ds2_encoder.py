# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from .encoder import Encoder


def conv2d_bn_actv(name, inputs, filters, kernel_size, activation_fn, strides,
                   padding, regularizer, training, data_format, bn_momentum,
                   bn_epsilon):
  conv = tf.layers.conv2d(
    name="{}".format(name),
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    kernel_regularizer=regularizer,
    use_bias=False,
    data_format=data_format,
  )
  bn = tf.layers.batch_normalization(
    name="{}/bn".format(name),
    inputs=conv,
    gamma_regularizer=regularizer,
    training=training,
    axis=-1 if data_format == 'channels_last' else 1,
    momentum=bn_momentum,
    epsilon=bn_epsilon,
  )
  output = activation_fn(bn)
  return output


def rnn_cell(rnn_cell_dim, layer_type, dropout_keep_prob=1.0):
  if layer_type == "layernorm_lstm":
    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
      num_units=rnn_cell_dim, dropout_keep_prob=dropout_keep_prob)
  else:
    if layer_type == "lstm":
      cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
    elif layer_type == "gru":
      cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
    elif layer_type == "cudnn_gru":
      cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_cell_dim)
    elif layer_type == "cudnn_lstm":
      cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_cell_dim)
    else:
      raise ValueError("Error: not supported rnn type:{}".format(layer_type))

    cell = tf.nn.rnn_cell.DropoutWrapper(
      cell, output_keep_prob=dropout_keep_prob)
  return cell


def row_conv(name, input_layer, batch, channels, width, activation_fn,
             regularizer, training, data_format, bn_momentum, bn_epsilon):
  if width < 2:
    return input_layer

  if data_format == 'channels_last':
    x = tf.reshape(input_layer, [batch, -1, 1, channels])
  else:
    input_layer = tf.transpose(input_layer, [0, 2, 1])  # B C T
    x = tf.reshape(input_layer, [batch, channels, -1, 1])
  cast_back = False
  if x.dtype.base_dtype == tf.float16:
    x = tf.cast(x, tf.float32)
    cast_back = True
  filters = tf.get_variable(
    name+'/w',
    shape=[width, 1, channels, 1],
    regularizer=regularizer,
    dtype=tf.float32,
  )
  strides = [1, 1, 1, 1]
  y = tf.nn.depthwise_conv2d(
    name=name + '/conv',
    input=x,
    filter=filters,
    strides=strides,
    padding='SAME',
    data_format='NHWC' if data_format == 'channels_last' else 'NCHW',
  )
  bn = tf.layers.batch_normalization(
    name="{}/bn".format(name),
    inputs=y,
    gamma_regularizer=regularizer,
    training=training,
    axis=-1 if data_format == 'channels_last' else 1,
    momentum=bn_momentum,
    epsilon=bn_epsilon,
  )
  output = activation_fn(bn)
  if data_format == 'channels_first':
    output = tf.transpose(output, [0, 2, 3, 1])
  output = tf.reshape(output, [batch, -1, channels])
  if cast_back:
    output = tf.cast(output, tf.float16)
  return output


class DeepSpeech2Encoder(Encoder):
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'dropout_keep_prob': float,
      'conv_layers': list,
      'activation_fn': None,  # any valid callable
      'num_rnn_layers': int,
      'row_conv': bool,
      'n_hidden': int,
      'use_cudnn_rnn': bool,
      'rnn_cell_dim': int,
      'rnn_type': ['layernorm_lstm', 'lstm', 'gru', 'cudnn_gru', 'cudnn_lstm'],
      'rnn_unidirectional': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'row_conv_width': int,
      'data_format': ['channels_first', 'channels_last'],
      'bn_momentum': float,
      'bn_epsilon': float,
    })

  def __init__(self, params=None, name="ds2_encoder", mode='train'):
    super(DeepSpeech2Encoder, self).__init__(params, name, mode)

  def _encode(self, input_dict):
    source_sequence = input_dict['src_inputs']

    seq_length = input_dict['src_lengths']
    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    bn_momentum = self.params.get('bn_momentum', 0.99)
    bn_epsilon = self.params.get('bn_epsilon', 1e-3)

    input_layer = tf.expand_dims(source_sequence, dim=-1)
    batch_size = input_layer.get_shape().as_list()[0]
    if data_format == 'channels_last':
      top_layer = input_layer
    else:
      top_layer = tf.transpose(input_layer, [0, 3, 1, 2])
    
    # ----- Convolutional layers -----------------------------------------------
    conv_layers = self.params['conv_layers']

    for idx_conv in range(len(conv_layers)):
      ch_out = conv_layers[idx_conv]['num_channels']
      kernel_size = conv_layers[idx_conv]['kernel_size']  # [time, freq]
      strides = conv_layers[idx_conv]['stride']
      padding = conv_layers[idx_conv]['padding']

      if padding == "VALID":
        seq_length = (seq_length - kernel_size[0] + strides[0]) // strides[0]
      else:
        seq_length = (seq_length + strides[0] - 1) // strides[0]

      top_layer = conv2d_bn_actv(
        name="conv{}".format(idx_conv + 1),
        inputs=top_layer,
        filters=ch_out,
        kernel_size=kernel_size,
        activation_fn=self.params['activation_fn'],
        strides=strides,
        padding=padding,
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
      )
    if data_format == 'channels_first':
      top_layer = tf.transpose(top_layer, [0, 2, 3, 1])

    # reshape to [B, T, FxC]
    f = top_layer.get_shape().as_list()[2]
    c = top_layer.get_shape().as_list()[3]
    fc = f * c
    top_layer = tf.reshape(top_layer, [batch_size, -1, fc])

    # ----- RNN ---------------------------------------------------------------
    num_rnn_layers = self.params['num_rnn_layers']
    if num_rnn_layers > 0:
      rnn_cell_dim = self.params['rnn_cell_dim']
      rnn_type = self.params['rnn_type']
      if self.params['use_cudnn_rnn']:
        # reshape to [B, T, C] --> [T, B, C]
        rnn_input = tf.transpose(top_layer, [1, 0, 2])
        if self.params['rnn_unidirectional']:
          direction = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
        else:
          direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

        if rnn_type == "cudnn_gru" or rnn_type == "gru":
          rnn_block = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=num_rnn_layers,
            num_units=rnn_cell_dim,
            direction=direction,
            dropout=1.0 - dropout_keep_prob,
            dtype=rnn_input.dtype,
            name="cudnn_gru",
          )
        elif rnn_type == "cudnn_lstm" or rnn_type == "lstm":
          rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_rnn_layers,
            num_units=rnn_cell_dim,
            direction=direction,
            dropout=1.0 - dropout_keep_prob,
            dtype=rnn_input.dtype,
            name="cudnn_lstm",
          )
        else:
          raise ValueError(
            "{} is not a valid rnn_type for cudnn_rnn layers".format(rnn_type)
          )
        top_layer, state = rnn_block(rnn_input)
        top_layer = tf.transpose(top_layer, [1, 0, 2])
      else:
        rnn_input = top_layer
        multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
          [rnn_cell(rnn_cell_dim=rnn_cell_dim, layer_type=rnn_type,
                    dropout_keep_prob=dropout_keep_prob)
           for _ in range(num_rnn_layers)]
        )
        if self.params['rnn_unidirectional']:
          top_layer, state = tf.nn.dynamic_rnn(
            cell=multirnn_cell_fw,
            inputs=rnn_input,
            sequence_length=seq_length,
            dtype=rnn_input.dtype,
            time_major=False,
          )
        else:
          multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [rnn_cell(rnn_cell_dim=rnn_cell_dim, layer_type=rnn_type,
                      dropout_keep_prob=dropout_keep_prob)
             for _ in range(num_rnn_layers)]
          )
          top_layer, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multirnn_cell_fw, cell_bw=multirnn_cell_bw,
            inputs=rnn_input,
            sequence_length=seq_length,
            dtype=rnn_input.dtype,
            time_major=False
          )
          # concat 2 tensors [B, T, n_cell_dim] --> [B, T, 2*n_cell_dim]
          top_layer = tf.concat(top_layer, 2)
    # -- end of rnn------------------------------------------------------------

    if self.params['row_conv']:
      channels = top_layer.get_shape().as_list()[-1]
      top_layer = row_conv(
        name="row_conv",
        input_layer=top_layer,
        batch=batch_size,
        channels=channels,
        activation_fn=self.params['activation_fn'],
        width=self.params['row_conv_width'],
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
      )

    # Reshape [B, T, C] --> [T*B, C]
    c = top_layer.get_shape().as_list()[-1]
    top_layer = tf.reshape(top_layer, [-1, c])

    # --- hidden layer with clipped ReLU activation and dropout-----------------
    top_layer = tf.layers.dense(
      inputs=top_layer,
      units=self.params['n_hidden'],
      kernel_regularizer=regularizer,
      activation=self.params['activation_fn'],
      name='fully_connected',
    )
    outputs = tf.nn.dropout(x=top_layer, keep_prob=dropout_keep_prob)

    # reshape from  [T*B,A] --> [T, B, A].
    # Output shape: [n_steps, batch_size, n_hidden]
    outputs = tf.reshape(
      outputs,
      [batch_size, -1, self.params['n_hidden']],
    )

    return {
      'encoder_outputs': outputs,
      'src_lengths': seq_length,
    }
