import torch
import torch.nn as nn
import torch.autograd
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2 


def hard_update(fromm, to):
    for fp, tp in zip(fromm.parameters(), to.parameters()):
        fp.data.copy_(tp.data)


def soft_update(fromm, to, tau):
    for fp, tp in zip(fromm.parameters(), to.parameters()):
        fp.data.copy_((1.0-tau)*fp.data + tau*tp.data)


'''
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
'''

def layer_init(layer, w_scale=1.0):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        if 'bias' in name:
            #layer._parameters[name].data.fill_(0.0)
            layer._parameters[name].data.uniform_(-0.08,0.08)
            #nn.init.constant_(layer._parameters[name].data, 0)
        else:
            '''
            nn.init.orthogonal_(layer._parameters[name].data)
            layer._parameters[name].data.mul_(w_scale)
            '''
            
            # Xavier init:
            nn.init.xavier_normal_(layer._parameters[name].data, gain=w_scale)
            
            '''
            
            # Uniform init:
            layer._parameters[name].data.uniform_(-0.1,0.1)
            layer._parameters[name].data.mul_(w_scale)
            '''
            
            '''
            if len(layer._parameters[name].size()) > 1:
                #nn.init.kaiming_normal_(layer._parameters[name], mode="fan_out", nonlinearity='leaky_relu')
                nn.init.orthogonal_(layer._parameters[name].data)
                layer._parameters[name].data.mul_(w_scale)
            '''
    return layer

def layer_init_lstm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer

def layer_init_gru(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def PreprocessFunctionConcatenate(x, use_cuda=False):
    x = np.concatenate(x, axis=None)
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)

def PreprocessFunction(x, use_cuda=False, normalization=True):
    if normalization:
        x = x/255.0
    if use_cuda:
        return torch.from_numpy(x).cuda()
    else:
        return torch.from_numpy(x)

def ResizeCNNPreprocessFunction(x, size, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to resize, normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param size: int or tuple, (height,width) size
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    if isinstance(size, int): size = (size,size)
    scaling_operation = T.Compose([T.ToPILImage(),
                                    T.Resize(size=size)])
    if len(x.shape) == 4:
        #batch:
        imgs = []
        for i in range(x.shape[0]):
            img = x[i]
            # handled Stacked images...
            per_image_first_channel_indices = range(0,img.shape[-1]+1,3)
            img = np.concatenate( [ np.array(scaling_operation(img[...,idx_begin:idx_end])) for idx_begin, idx_end in zip(per_image_first_channel_indices,per_image_first_channel_indices[1:])], axis=-1)
            img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
            imgs.append(img)
        x = torch.cat(imgs, dim=0)
    else:
        x = scaling_operation(x)
        x = np.array(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
    # WATCHOUT: it is necessary to cast the tensor into float before doing
    # the division, otherwise the result is yielded as a uint8 (full of zeros...)
    x = x.type(torch.FloatTensor) / 255. if normalize_rgb_values else x
    if use_cuda:
        return x.type(torch.cuda.FloatTensor)
    return x.type(torch.FloatTensor)


def ResizeCNNInterpolationFunction(x, size, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to resize, normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param size: int size (height==width)
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    x = np.array(x).astype(np.float32)
    
    h,w = x.shape[1:3]
    '''
    osize = h
    if h != w:
        osize = max([h, w])
        b = x.shape[0]
        c = x.shape[3]
        x_flat = x.reshape((-1, h, w))
        xs = []
        for idx in range(x_flat.shape[0]):
            xs.append( cv2.resize(x_flat[idx], (osize, osize)).reshape((1, osize, osize)))
        xs = np.concatenate(xs, axis=0)
        x = xs.reshape((b, osize, osize, c))

    scaling_factor = float(size)/osize
    '''
    b = x.shape[0]
    c = x.shape[3]
    x = x.transpose(0, 3, 1, 2)
    # b x c x h x w 
    if size is not None and size != h:
        x_flat = x.reshape((-1, h, w))
        xs = []
        for idx in range(x_flat.shape[0]):
            xs.append( cv2.resize(x_flat[idx], (size, size)).reshape((1, size, size)))
        xs = np.concatenate(xs, axis=0)
        x = xs.reshape((b, c, size, size))

    x = torch.from_numpy(x)
    x = x / 255. if normalize_rgb_values else x
    #x = F.interpolate(x, scale_factor=scaling_factor)
    if use_cuda:
        return x.type(torch.cuda.FloatTensor)
    return x.type(torch.FloatTensor)


def CNNPreprocessFunction(x, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    x = x.transpose((2, 0, 1))
    x = x / 255. if normalize_rgb_values else x
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def random_sample(indices, batch_size):
    '''
    TODO
    :param indices:
    :param batch_size:
    :returns: Generator
    '''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]


# Note, consider alternative way of calculating output size.
# Idea: use model.named_modules() generator to find last module and look at its
# Number of features / output channels (if the last module is a ConvNet)
def output_size_for_model(model, input_shape):
    '''
    Computes the size of the last layer of the :param model:
    which takes an input of shape :param input_shape:.

    :param model: torch.nn.Module model which takes an input of shape :param input_shape:
    :param input_shape: Shape of the input of the
    :returns: size of the flattened output torch.Tensor
    '''
    return model(torch.autograd.Variable(torch.zeros(1, *input_shape))).view(1, -1).size(1)
