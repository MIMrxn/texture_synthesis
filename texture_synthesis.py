"""
    Texture synthesis based off of Gatys 
    "Texture Synthesis Using Convolutional Neural Networks"
    arXiv:1505.07376v3 [cs.CV] 6 Nov 2015
"""

import keras

from keras.models import Model, Sequential
from keras.applications import vgg19, resnet50
from keras.layers import Conv2D
from keras import backend as K

import numpy as np

#img_width = 256
#img_height = 256
img_size = 256
batch_size = 1

# base vgg19 model
vgg19_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
vgg19_model.summary()

# get the layers of the vgg19 model
vgg19_layers = [layer for layer in vgg19_model.layers]
gatys_layers = []

''' reconstruct gatys model using the base vgg19 model
    change max pooling layers to average pooling layers
    model structure (layer, name, output(batch_size, img_size, img_size, filters)):
        
        layer 1: block1_conv1 (img_size, img_size, 64)
        layer 2: block1_conv2 (img_size, img_size, 64)
        layer 3: block1_pool (img_size/2, img_size/2, 64)
        
        layer 4: block2_conv1 (img_size/2, img_size/2, 128)
        layer 5: block2_conv2 (img_size/2, img_size/2, 128)
        layer 6: block2_pool (img_size/4, img_size/4, 128)
        
        layer 7: block3_conv1 (img_size/4, img_size/4, 256)
        layer 8: block3_conv2 (img_size/4, img_size/4, 256)
        layer 9: block3_conv3 (img_size/4, img_size/4, 256)
        layer 10: block3_conv4 (img_size/4, img_size/4, 256)
        layer 11: block3_pool (img_size/8, img_size/8, 256)
        
        layer 12: block4_conv1 (img_size/8, img_size/8, 512)
        layer 13: block4_conv2 (img_size/8, img_size/8, 512)
        layer 14: block4_conv3 (img_size/8, img_size/8, 512)
        layer 15: block4_conv4 (img_size/8, img_size/8, 512)
        layer 16: block4_pool (img_size/16, img_size/16, 512)
        
        layer 17: block5_conv1 (img_size/16, img_size/16, 512)
        layer 18: block5_conv2 (img_size/16, img_size/16, 512)
        layer 19: block5_conv3 (img_size/16, img_size/16, 512)
        layer 20: block5_conv4 (img_size/16, img_size/16, 512)
        layer 21: block5_pool (img_size/32, img_size/32, 512)
'''
for i in range(len(vgg19_layers)):
    layer_config = vgg19_layers[i].get_config()
    if isinstance(vgg19_layers[i], keras.layers.Conv2D):
        if i==1:
            layer_config['input_shape'] = (img_size, img_size, 3)
        layer_config['padding'] = 'same'
        gatys_layers.append(keras.layers.Conv2D.from_config(layer_config))
    elif isinstance(vgg19_layers[i], keras.layers.MaxPooling2D):
        layer_config['padding'] = 'same'
        gatys_layers.append(keras.layers.AveragePooling2D.from_config(layer_config))
    

gatys_model = keras.models.Sequential(layers = gatys_layers)
gatys_model.summary()

## TODO: ADD RESCALED WEIGHTS INTO GATYS MODEL

# get the output of each layer to use to calculate the gram matrix
# output has shape (batch_size, img_size, img_size, num_filters)
# activations_dict = dict([(layer.name, layer.output) for layer in gatys_model.layers])
activation_shapes = [layer.output_shape for layer in gatys_model.layers]

# calculates gram matrix of the feature maps in a layer
def compute_gram_matrix(activation):
    # ensure number of axes of tensor is 4
    assert K.ndim(activation) == 4
    # get activation output shape
    activation_shape = activation.output_shape
    # N: number of filters
    # M: size of each feature map
    N = activation_shape[3]
    M = activation_shape[1] * activation_shape[2]
    # format size to: (batch_size, num_filters, img_size, img_size)
    if K.image_data_format() == 'channels_last':
        activation = K.permute_dimensions(activation, (0, 3, 1, 2))
    # F: feature maps matrix of shape (N, M)
    F = K.reshape(activation, (N, M))
    # G: gram matrix of size (N, N)
    G = K.dot(F, K.transpose(F))
    return G

# calculate mean squared distance between original image's gram matrix 
# and generated image's gram matrix
def compute_E(w, G_img, G_gen, activation):
    # ensure number of axes of tensor is 4
    assert K.ndim(activation) == 4
    # get activation output shape
    activation_shape = activation.output_shape
    # N: number of filters
    # M: size of each feature map
    N = activation_shape[3]
    M = activation_shape[1] * activation_shape[2]
    s = 1.0 / (4.0 * K.square(N) * K.square(M))
    E = w * s * K.sum(K.square(G_img - G_gen))
    return E

# compute derivative of E
def compute_derivative_E(w, E, F_gen, G_img, G_gen, activation):
    # ensure number of axes of tensor is 4
    assert K.ndim(activation) == 4
    # get activation output shape
    activation_shape = activation.output_shape
    # N: number of filters
    # M: size of each feature map
    N = activation_shape[3]
    M = activation_shape[1] * activation_shape[2]
    s = 1.0 / (K.square(N) * K.square(M))
    gradient = w * s * K.dot(K.transpose(F_gen), G_img - G_gen)
    return gradient

# keras and tesnorflow do not offer L-BFGS
# possible solutions exist here:
# https://github.com/keras-team/keras/issues/5085
# https://github.com/tensorflow/tensorflow/issues/446
# try existing optimizers
# learning rate and clipping are shared with all optimizers
sgd_opt = keras.optimizers.SGD()
adam_opt = keras.optimizers.Adam()
rmsprop_opt = keras.optimizers.rmsprop()

## TODO preprocess images
def preprocess_img(img, img_size):
    ## TODO; crop center of each image
    
    # add batch dimension
    img = np.expand_dims(img, axis=0)
    return img