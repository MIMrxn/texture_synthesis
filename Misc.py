import numpy as np
import scipy
import caffe
import matplotlib.pyplot as plt
from IPython.display import display,clear_output
from PIL import Image
import os
import pickle

class constraint(object):
    '''
    Object that contains the constraints on a particular layer for the image synthesis.
    '''

    def __init__(self, loss_functions, parameter_lists):
        self.loss_functions = loss_functions
        self.parameter_lists = parameter_lists
  
def get_indices(net, constraints):
    '''
    Helper function to pick the indices of the layers included in the loss function from all layers of the network.
    
    :param net: caffe.Classifier object defining the network
    :param contraints: dictionary where each key is a layer and the corresponding entry is a constraint object
    :return: list of layers in the network and list of indices of the loss layers in descending order
    '''

    indices = [ndx for ndx,layer in enumerate(net.blobs.keys()) if layer in constraints.keys()]
    return net.blobs.keys(),indices[::-1]

def show_progress(x, net, title=None, handle=False):
    '''
    Helper function to show intermediate results during the gradient descent.

    :param x: vectorised image on which the gradient descent is performed
    :param net: caffe.Classifier object defining the network
    :param title: optional title of figuer
    :param handle: obtional return of figure handle
    :return: figure handle (optional)
    '''

    disp_image = (x.reshape(*net.blobs['data'].data.shape)[0].transpose(1,2,0)[:,:,::-1]-x.min())/(x.max()-x.min())
    clear_output()
    plt.imshow(disp_image)
    if title != None:
        ax = plt.gca()
        ax.set_title(title)
    f = plt.gcf()
    display()
    plt.show()    
    if handle:
        return f
   
def get_bounds(images, im_size):
    '''
    Helper function to get optimisation bounds from source image.

    :param images: a list of images 
    :param im_size: image size (height, width) for the generated image
    :return: list of bounds on each pixel for the optimisation
    '''

    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds 

def test_gradient(function, parameters, eps=1e-6):
    '''
    Simple gradient test for any loss function defined on layer output

    :param function: function to be tested, must return function value and gradient
    :param parameters: input arguments to function passed as keyword arguments
    :param eps: step size for numerical gradient evaluation 
    :return: numerical gradient and gradient from function
    '''

    i,j,k,l = [np.random.randint(s) for s in parameters['activations'].shape]
    f1,_ = function(**parameters)
    parameters['activations'][i,j,k,l] += eps
    f2,g = function(**parameters)
    
    return [(f2-f1)/eps,g[i,j,k,l]]

def gram_matrix(activations):
    '''
    Gives the gram matrix for feature map activations in caffe format with batchsize 1. Normalises by spatial dimensions.

    :param activations: feature map activations to compute gram matrix from
    :return: normalised gram matrix
    '''

    N = activations.shape[1]
    F = activations.reshape(N,-1)
    M = F.shape[1]
    G = np.dot(F,F.T) / M
    return G
    
def disp_img(img):
    '''
    Returns rescaled image for display with imshow
    '''
    disp_img = (img - img.min())/(img.max()-img.min())
    return disp_img  

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    
    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return matched_image

def load_image(file_name, im_size, net_model, net_weights, mean, show_img=False):
    '''
    Loads and preprocesses image into caffe format by constructing and using the appropriate network.

    :param file_name: file name of the image to be loaded
    :param im_size: size of the image after preprocessing if float that the original image is rescaled to contain im_size**2 pixels
    :param net_model: file name of the prototxt file defining the network model
    :param net_weights: file name of caffemodel file defining the network weights
    :param mean: mean values for each color channel (bgr) which are subtracted during preprocessing
    :param show_img: if True shows the loaded image before preprocessing
    :return: preprocessed image and caffe.Classifier object defining the network
    '''

    img = caffe.io.load_image(file_name)
    img = crop_image(img, im_size)
    if show_img:
        plt.imshow(img)
    if isinstance(im_size,float):
        im_scale = np.sqrt(im_size**2 /np.prod(np.asarray(img.shape[:2])))
        im_size = im_scale * np.asarray(img.shape[:2])
    batchSize = 1
    with open(net_model,'r+') as f:
        data = f.readlines() 
    data[2] = "input_dim: %i\n" %(batchSize)
    data[4] = "input_dim: %i\n" %(im_size[0])
    data[5] = "input_dim: %i\n" %(im_size[1])
    with open(net_model,'r+') as f:
        f.writelines(data)
    net_mean =  np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
    #load pretrained network
    net = caffe.Classifier( 
    net_model, net_weights,
    mean = net_mean,
    channel_swap=(2,1,0),
    input_scale=255,)
    img_pp = net.transformer.preprocess('data',img)[None,:]
    return[img_pp, net]

def summarize(net):
    '''
    Summarizes the layer information of the network
    :param net: caffe.Classifier object defining the network
    :return: list of 2-tuples with layer info (name, output_shape)
    '''
    #summary = [(name, output[0].data.shape, np.prod(output[0].data.shape)) for name, output in net.params.items()]
    summary = [(name, output.data.shape) for name, output in net.blobs.items()]
    return summary

def crop_image(img, im_size):
    '''
    Center crops an image
    :param img: image to crop
    :param im_size: crop size
    :return: center cropped image of size im_size x im_size
    '''
    crop_size = int(im_size)
    h, w, c = img.shape[:3]
    np_img = np.zeros([h, w, c], dtype=np.uint8)
    np_img = img

    w0 = w//2-(crop_size//2)
    h0 = h//2-(crop_size//2)
    cropped_img = np_img[h0:h0+crop_size, w0:w0+crop_size]
    return cropped_img

def calculate_params(net, constrained_layers):
    '''
    Calculates the total parameters to match
    :param net: caffe.Classifier object defining the network
    :param constrained_layers: layers we are constraining and matching parameters from
    :return: total parameters to match
    '''
    summary = summarize(net)
    #Grab each layer with its feature map sizes
    N_list = [(name, output[1]) for name, output in summary for layer in constrained_layers if layer==name]
    total_params = np.sum([output[1]*output[1]/2 for name, output in summary for layer in constrained_layers if layer==name])
    return total_params

def begin_metrics(net, img, img_size, tex_layers, tex_weights, maxiter):
    '''
    Creates a metrics file and populates it with starting metrics
    :param net: caffe.Classifier object defining the network
    :param img: source image we are collecting metrics for
    :param img_size: source image size
    :param tex_layers: layers we are constraining and matching parameters from
    :param tex_weights: weights for constrained layers
    :param maxiter: max iterations
    '''
    base_dir = os.getcwd()
    metrics_dir = os.path.join(base_dir, 'metrics/')
    img_name = os.path.splitext(img)[0]
    metrics_file = os.path.join(metrics_dir, img_name+'_metrics.pkl')

    f = open(metrics_file, 'wb')
    f.seek(0)
    f.truncate()
    img_info = [img, img_size]
    summary = summarize(net)
    total_params = calculate_params(net, tex_layers)
    pickle.dump(img_info, f)
    pickle.dump(summary, f)
    pickle.dump(tex_layers, f)
    pickle.dump(tex_weights, f)
    pickle.dump(total_params, f)
    pickle.dump(maxiter, f)
    f.close()

def get_img_metrics(img):
    '''
    Read in metrics data and save it to a dictionary
    :param img: source image to obtain metrics from
    '''
    base_dir = os.getcwd()
    metrics_dir = os.path.join(base_dir, 'metrics/')
    img_name = os.path.splitext(img)[0]
    metrics_file = os.path.join(metrics_dir, img_name+'_metrics.pkl')
    metrics_dict = {}
    with open(metrics_file, 'rb') as f:
        img_info = pickle.load(f)
        summary = pickle.load(f)
        tex_layers = pickle.load(f)
        tex_weights = pickle.load(f)
        total_params = pickle.load(f)
        maxiter = pickle.load(f)
        f_vals = pickle.load(f)
        metrics_dict = {'img_info': img_info, 
                    'summary': summary,
                    'tex_layers': tex_layers,
                    'tex_weigts': tex_weights,
                    'total_params': total_params,
                    'maxiter': maxiter,
                    'f_vals': f_vals}
    return metrics_dict