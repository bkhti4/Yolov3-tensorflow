# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 07:13:35 2020

@author: bakhtiar
"""
# create a YOLO-fastest Keras model and save it to file
import struct
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model

layer_test = 0

def _conv_block(inp, convs, skip=False, dropout=False, dropout_pb=0.15, send_skip=False):
  global layer_test
  x = inp
  count = 0
  for conv in convs:
    #print('layer:', layer_test)
    #layer_test += 1
    if count == (len(convs) - 3) and skip:
      skip_connection = x
    if count == 1 and send_skip:
      send_inter = x
    count += 1
    #if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
    if conv['groups']:
      gps = conv['n_groups']
    else:
      gps = 1
    x = Conv2D(conv['filter'],
				   conv['kernel'],
				   strides=conv['stride'],
           groups=gps,
				   padding='same' if conv['pad'] else 'valid', # peculiar padding as darknet prefer left and top
				   name='conv_' + str(conv['layer_idx']),
				   use_bias=False)(x)
    if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
  if dropout: x = Dropout(rate=dropout_pb)(x)
  if send_skip:
    if skip:
      return add([skip_connection, x]), send_inter
    else:
      return x
  else:
    if skip:
      return add([skip_connection, x])
    else:
      return x
  
def make_yolo_fastest_model():
  input_image = Input(shape=(None, None, 3))
	# Layer  0 => 4
  #skip_81 = input_image
  x = _conv_block(input_image, [{'filter': 8, 'kernel': 3, 'stride': 2, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 0},
                                {'filter':  8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 1},
							                  {'filter': 8, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 8, 'layer_idx': 2},
                                {'filter': 4, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 3}])
  #skip_81 = x
  # Layer 5 => 9
  x = _conv_block(x, [{'filter':  8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 4},
							  {'filter': 8, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 8, 'layer_idx': 5},
                {'filter': 4, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 6}],
                 skip=True, dropout=True, dropout_pb=0.15)

  # Layer 10 => 13
  x = _conv_block(x, [{'filter':  24, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 10},
							  {'filter': 24, 'kernel': 3, 'stride': 2, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 24, 'layer_idx': 11},
                {'filter': 8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 12}])

  # Layer 14 => 23
  for i in range(2):
    x = _conv_block(x, [{'filter':  32, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 14+(i*5)},
							  {'filter': 32, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 32, 'layer_idx': 15+(i*5)},
                {'filter': 8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 16+(i*5)}],
                 skip=True, dropout=True, dropout_pb=0.15)

  # Layer 24 => 27
  x = _conv_block(x, [{'filter':  32, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 24},
							  {'filter': 32, 'kernel': 3, 'stride': 2, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 32, 'layer_idx': 25},
                {'filter': 8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 26}])

  # Layer 28 => 37
  for i in range(2):
    x = _conv_block(x, [{'filter':  48, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 28+(i*5)},
							  {'filter': 48, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 48, 'layer_idx': 29+(i*5)},
                {'filter': 8, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 30+(i*5)}],
                 skip=True, dropout=True, dropout_pb=0.15)

  # Layer 38 => 41
  x = _conv_block(x, [{'filter':  48, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 38},
							  {'filter': 48, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 48, 'layer_idx': 39},
                {'filter': 16, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 40}])

  # Layer 42 => 61
  for i in range(4):
    x = _conv_block(x, [{'filter':  96, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 42+(i*5)},
							  {'filter': 96, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 96, 'layer_idx': 43+(i*5)},
                {'filter': 16, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 44+(i*5)}],
                 skip=True, dropout=True, dropout_pb=0.15)

  # Layer 62 => 65
  x = _conv_block(x, [{'filter':  96, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 62},
                      {'filter': 96, 'kernel': 3, 'stride': 2, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 96, 'layer_idx': 63},
                      {'filter': 24, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 64}])

  # Layer 66 => 85
  for i in range(4):
    x = _conv_block(x, [{'filter':  136, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 66+(i*5)},
							  {'filter': 136, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 136, 'layer_idx': 67+(i*5)},
                {'filter': 24, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 68+(i*5)}],
                 skip=True, dropout=True, dropout_pb=0.15) 

  # Layer 86 => 89 
  x = _conv_block(x, [{'filter':  136, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 86}])
  skip_81 = x
  x = _conv_block(x, [{'filter': 136, 'kernel': 3, 'stride': 2, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 136, 'layer_idx': 87},
                      {'filter': 48, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 88}])

  # Layer 90 => 114
  for i in range(5):
    x = _conv_block(x, [{'filter':  224, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 90+(i*5)},
							  {'filter': 224, 'kernel': 3, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 224, 'layer_idx': 91+(i*5)},
                {'filter': 48, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': False, 'groups': False, 'layer_idx': 92+(i*5)}],
                 skip=True, dropout=True, dropout_pb=0.15) 

  # Layer 115
  x = _conv_block(x, [{'filter': 96, 'kernel': 1, 'stride': 1, 'pad': False, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 115}]) 

	# Layer 116 => 121
  yolo_121 = _conv_block(x, [{'filter': 96, 'kernel': 5, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 96, 'layer_idx': 116},
                  {'filter': 128, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': True,  'leaky': False, 'groups': False, 'layer_idx': 117},
							    {'filter': 128, 'kernel': 5, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 128, 'layer_idx': 118},
                  {'filter': 128, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': True,  'leaky': False,  'groups': False, 'layer_idx': 119},
							    {'filter': 255, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': False, 'leaky': False, 'groups': False, 'layer_idx': 120}])

	# Layer 122 => 123
  x = UpSampling2D(2)(x)
  x = concatenate([x, skip_81])

	# Layer 124 => 130
  yolo_130 = _conv_block(x, [{'filter': 96, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': False, 'layer_idx': 124},
							    {'filter': 96, 'kernel': 5, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 96, 'layer_idx': 125},
                  {'filter': 96, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': True,  'leaky': False, 'groups': False, 'layer_idx': 126},
							    {'filter': 96, 'kernel': 5, 'stride': 1, 'pad': True, 'bnorm': True, 'leaky': True, 'groups': True, 'n_groups': 96, 'layer_idx': 127},
                  {'filter': 96, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': True,  'leaky': False,  'groups': False, 'layer_idx': 128},
							    {'filter': 255, 'kernel': 1, 'stride': 1, 'pad': True, 'bnorm': False, 'leaky': False, 'groups': False, 'layer_idx': 129}])

  model = Model(input_image, [yolo_121, yolo_130])
  return model


class WeightReader:
	def __init__(self, weight_file):
		with open(weight_file, 'rb') as w_f:
			major,	= struct.unpack('i', w_f.read(4))
			minor,	= struct.unpack('i', w_f.read(4))
			revision, = struct.unpack('i', w_f.read(4))
			if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
				w_f.read(8)
			else:
				w_f.read(4)
			transpose = (major > 1000) or (minor > 1000)
			binary = w_f.read()
		self.offset = 0
		self.all_weights = np.frombuffer(binary, dtype='float32')

	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]

	def load_weights_yolo_fastest(self, model):
		for i in range(130):
			try:
				conv_layer = model.get_layer('conv_' + str(i))
				print("loading weights of convolution #" + str(i))
				if i not in [120, 129]:
					norm_layer = model.get_layer('bnorm_' + str(i))
					size = np.prod(norm_layer.get_weights()[0].shape)
					beta  = self.read_bytes(size) # bias
					gamma = self.read_bytes(size) # scale
					mean  = self.read_bytes(size) # mean
					var   = self.read_bytes(size) # variance
					weights = norm_layer.set_weights([gamma, beta, mean, var])
				if len(conv_layer.get_weights()) > 1:
					bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel, bias])
				else:
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel])
			except ValueError:
				print("no convolution #" + str(i))

	def reset(self):
		self.offset = 0

# define the model
model = make_yolo_fastest_model()
# load the model weights
weight_reader = WeightReader('yolo-fastest.weights')
# set the model weights into the model
weight_reader.load_weights_yolo_fastest(model)
# save the model to file
model.save('model_yolo_fastest.h5')