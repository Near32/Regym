import os
import numpy as np
import torch
import torch.nn as nn


class BaseNet:
    def __init__(self):
        pass


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
            #layer._parameters[name].data.uniform_(-0.08,0.08)
            nn.init.constant_(layer._parameters[name].data, 0)
        else:
            #nn.init.orthogonal_(layer._parameters[name].data)
            '''
            fanIn = param.size(0)
            fanOut = param.size(1)

            factor = math.sqrt(2.0/(fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            layer._parameters[name].data.copy_(weight)
            '''
            
            '''
            layer._parameters[name].data.uniform_(-0.08,0.08)
            layer._parameters[name].data.mul_(w_scale)
            '''
            if len(layer._parameters[name].size()) > 1:
                #nn.init.kaiming_normal_(layer._parameters[name], mode="fan_out", nonlinearity='leaky_relu')
                nn.init.orthogonal_(layer._parameters[name].data)
                layer._parameters[name].data.mul_(w_scale)
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


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x


def range_tensor(end):
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()
