# -*- coding: utf-8 -*-

#%pylab inline
import pylab
import numpy as np
from numpy import prod,sum
import glob
import sys
import os
from collections import OrderedDict
import caffe
import time
import pprint
base_dir = os.getcwd()
sys.path.append(base_dir)
from DeepImageSynthesis import *
#VGGweights = os.path.join(base_dir, 'Models/ResNet-50-model.caffemodel')
#VGGmodel = os.path.join(base_dir, 'Models/ResNet-50-deploy.prototxt')
VGGweights = os.path.join(base_dir, 'Models/vgg_normalised.caffemodel')
VGGmodel = os.path.join(base_dir, 'Models/VGG_ave_pool_deploy.prototxt')
imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)
im_dir = os.path.join(base_dir, 'Images/')
gpu = 0
caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
caffe.set_device(gpu)

runDataSetImages = False


#load source image
def processImg( imgDir, imgName):
    source_img_name = glob.glob1(imgDir, imgName)[0]
    source_img_org = caffe.io.load_image(imgDir + source_img_name)
    im_size = 256.
    [source_img, net] = load_image(imgDir + source_img_name, im_size, #Uses Misc, to load image
                                VGGmodel, VGGweights, imagenet_mean, 
                                show_img=True)
    im_size = np.asarray(source_img.shape[-2:])
    
    #l-bfgs parameters optimisation
    maxiter = 2000
    m = 20
    
    #define layers to include in the texture model and weights w_l
    tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
    tex_weights = [1e9,1e9,1e9,1e9,1e9]
    
    #pass image through the network and save the constraints on each layer
    constraints = OrderedDict()
    net.forward(data = source_img)
    for l,layer in enumerate(tex_layers):
        constraints[layer] = constraint([LossFunctions.gram_mse_loss],
                                        [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                         'weight': tex_weights[l]}])
    
    #get optimisation bounds
    bounds = get_bounds([source_img],im_size)
    # generate new texture
    result = ImageSyn(net, constraints, bounds=bounds,
                      callback=lambda x: show_progress(x,net), 
                      minimize_options={'maxiter': maxiter,
                                        'maxcor': m,
                                        'disp':True,
                                        'ftol': 0, 'gtol': 0})
    #match histogram of new texture with that of the source texture and show both images
    new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1]
    new_texture = histogram_matching(new_texture, source_img_org)
    pylab.imshow(new_texture)
    pylab.savefig( 'results/%s' % (imgName) )
    pylab.figure()
    pylab.imshow(source_img_org)
    
# BASED ON https://www.robots.ox.ac.uk/~vgg/data/dtd/ dataset image folder
root = 'DbImages'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

if( runDataSetImages ):
    for i,fol in enumerate(dirlist):
        if(i==3):
            break
        imgRoot = root + '/' + fol + '/'
        imglist = [ item for item in os.listdir(imgRoot) if os.path.isfile(os.path.join(imgRoot, item)) ]
        for img in imglist:
            processImg(imgRoot,img)
else:
    processImg(im_dir,'pebbles.jpg')
