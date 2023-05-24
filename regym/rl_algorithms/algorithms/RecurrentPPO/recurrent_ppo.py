from typing import Dict, List, Any, Optional, Callable

import copy
from collections import deque 
from functools import partial 
import time 

import ray
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt 

import regym
from regym.rl_algorithms.algorithms.algorithm import Algorithm
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from regym.rl_algorithms.algorithms.RecurrentPPO import recurrent_ppo_loss
from regym.rl_algorithms.replay_buffers import (
    ReplayStorage, 
    PrioritizedReplayStorage, 
    SharedPrioritizedReplayStorage,
)
from regym.rl_algorithms.utils import (
    archi_concat_fn, 
    concat_fn, 
    _concatenate_hdict, 
    _concatenate_list_hdict, 
    _extract_rnn_states_from_seq_indices,
)

from regym.thirdparty.Archi.Archi.model import Model as ArchiModel

import wandb
summary_writer = None 


def standardize(x):
    stable_eps = 1.0e-12
    return (x - x.mean()) / (x.std()+stable_eps)

class RunningMeanStd(object):
    def __init__(
        self, 
        mean=None, 
        std=None, 
        count=0,
        shape=(1,),
    ):
        if mean is None:
            self.mean = mean
        else:
            self.mean = mean*torch.ones(shape)
        if std is None:
            self.std = std
        else:
            self.std = std*torch.ones(shape)
        self.count = count

    def update(self, bdata:List[torch.Tensor]):
        bdata = torch.stack(bdata, dim=0)
        bmean = torch.mean(bdata, dim=0)
        bstd = torch.std(bdata, dim=0)
        bcount = len(bdata)
        
        if self.count == 0:
            self.mean = bmean
            self.std = bstd
            self.count = bcount

            return

        self.update_from_moments(
            bmean=bmean,
            bstd=bstd,
            bcount=bcount,
        )
    
    def update_from_moments(self, bmean, bstd, bcount):
        delta = bmean - self.mean
        tot_count = self.count + bcount
        
        new_mean = self.mean + delta * bcount / tot_count
        
        bvar = torch.square(bstd)
        var = torch.square(self.std)
        m_a = var * self.count 
        m_b = bvar * bcount
        M2 = m_a + m_b + torch.square(delta) * self.count * bcount / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.std = torch.sqrt(new_var)
        self.count = tot_count

        
class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RecurrentPPOAlgorithm(R2D2Algorithm):
    def __init__(
        self, 
        kwargs: Dict[str, Any], 
        model: nn.Module,
        optimizer=None,
        loss_fn: Callable = recurrent_ppo_loss.compute_loss,
        sum_writer=None,
        name='recurrent_ppo_algo',
        single_storage=False,
    ):
        '''
        Refer to original paper for further explanation: https://arxiv.org/pdf/1707.06347.pdf
        horizon: (0, infinity) Number of timesteps that will elapse in between optimization calls.
        discount: (0,1) Reward discount factor
        use_gae: Flag, wether to use Generalized Advantage Estimation (GAE) (instead of return base estimation)
        gae_tau: (0,1) GAE hyperparameter.
        use_cuda: Flag, to specify whether to use CUDA tensors in Pytorch calculations
        entropy_weight: (0,1) Coefficient for (regularatization) entropy based loss
        gradient_clip: float, Clips gradients to reduce the chance of destructive updates
        optimization_epochs: int, Number of epochs per optimization step.
        mini_batch_size: int, Mini batch size to use to calculate losses (Use power of 2 for efficciency)
        ppo_ratio_clip: float, clip boundaries (1 - clip, 1 + clip) used in clipping loss function.
        learning_rate: float, optimizer learning rate.
        adam_eps: (float), Small Epsilon value used for ADAM optimizer. Prevents numerical instability when v^{hat} (Second momentum estimator) is near 0.
        model: (Pytorch nn.Module) Used to represent BOTH policy network and value network
        ''' 
        Algorithm.__init__(self=self, name=name)
        self.single_storage = single_storage
        
        print(kwargs)

        
        self.train_request_count = 0 

        self.kwargs = copy.deepcopy(kwargs)        
        self.use_cuda = kwargs["use_cuda"]
        self.nbr_actor = self.kwargs['nbr_actor']
        
        self.use_HER = self.kwargs['use_HER'] if 'use_HER' in self.kwargs else False
        self.n_step = 1
        # LEGACY:
        '''
        if self.kwargs.get('n_step', None) is not None:
            raise NotImplementedError
        
        if self.n_step > 1:
            self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]

        '''

        self.horizon = kwargs['horizon']
        self.min_capacity = 1 #self.horizon*self.nbr_actor
        self.batch_size = int(kwargs["batch_size"])
        self.nbr_minibatches = int(kwargs["nbr_minibatches"])

        self.GAMMA = float(kwargs["discount"])
        
        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        if optimizer is None:
            parameters = self.model.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = float(kwargs['learning_rate']) 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            self.optimizer = optim.Adam(
                parameters, 
                lr=lr, 
                betas=(0.9,0.999), 
                eps=float(kwargs['adam_eps']),
                weight_decay=float(kwargs.get("adam_weight_decay", 0.0)),
            )
        else: 
            self.optimizer = optimizer

        self.loss_fn = loss_fn
        print(f"WARNING: loss_fn is {self.loss_fn}")
            
        
        # DEPRECATED in order to allow extra_inputs infos 
        # stored in the rnn_states that acts as frame_states...
        #self.recurrent = False
        self.recurrent = True
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.recurrent_nn_submodule_names): self.recurrent = True

        self.keys = [
            's', 'a', 'r', 'succ_s', 'non_terminal', 'info',
            'v', 'q', 'pi', 'log_pi', 'ent', 'greedy_action',
            'adv', 'std_adv', 'ret', 'qa', 'log_pi_a',
            'mean', 'action_logits', 'succ_info',
        ]
        self.circular_keys={}
        self.circular_offsets={}
        
        if self.recurrent:
            self.keys.append('rnn_states')
            #self.circular_keys.update({'next_rnn_states':'rnn_states'})
            #self.circular_offsets.update({'next_rnn_states':1})
          
        self.keys_to_retrieve = [
            's','a','non_terminal','ret','adv','std_adv', 
            'v','log_pi_a','ent',
        ] #copy.deepcopy(self.keys)
        if self.recurrent:  
            self.keys_to_retrieve += ['rnn_states']
            #'next_rnn_states']
        
        self.kremap = {
            's':'states',
            'a':'actions',
            'ret':'returns',
            'adv':'advantages',
            'std_adv':'std_advantages',
            'non_terminal':'non_terminals',
            'log_pi_a':'log_probs_old',

            'v':'v',
            'ent':'ent',
        }
        
        self.storages = None
        self.use_mp = False 
        self.sequence_replay_overlap_length = 0
        self.kwargs['sequence_replay_overlap_length'] = 0
        
        # PREVIOUSLY : when forcing the unroll lenght:
        #self.sequence_replay_unroll_length = self.horizon
        #self.kwargs['sequence_replay_unroll_length'] = self.horizon
        # NOW: letting it be set by user:
        self.sequence_replay_unroll_length = self.kwargs['sequence_replay_unroll_length']

        self.sequence_replay_store_on_terminal = False
        self.sequence_replay_burn_in_ratio = self.kwargs['sequence_replay_burn_in_ratio']
        self.sequence_replay_burn_in_length = int(self.sequence_replay_unroll_length*self.sequence_replay_burn_in_ratio)
        self.kwargs['sequence_replay_burn_in_length'] = self.sequence_replay_burn_in_length
        #self.sequence_replay_store_on_terminal = kwargs["sequence_replay_store_on_terminal"]
        #self.replay_buffer_capacity = kwargs['replay_capacity'] // (self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)
        
        #PREVIOUSLY: only as many as there were actors, and so there would be only one element per storage of length horizon:
        #self.replay_buffer_capacity = self.nbr_actor * (self.horizon // self.sequence_replay_unroll_length)
        # NOW: we still want to separate over the different actors,
        # and also we want the horizon to be separate over the different unroll length : 
        self.replay_buffer_capacity = self.nbr_actor * (self.horizon // self.sequence_replay_unroll_length)
        
        if self.kwargs.get('use_PER', False):
            print("WARNING: RPPO cannot use PER.")
            print("WARNING: Falling back onto normal ReplayStorage.")
        self.kwargs['use_PER'] = False 
        self.use_PER = False
        self.reset_storages()
        
        global summary_writer
        if sum_writer is not None: summary_writer = sum_writer
        self.summary_writer = summary_writer
        if regym.RegymManager is not None:
            from regym import RaySharedVariable
            try:
                self._param_update_counter = ray.get_actor(f"{self.name}.param_update_counter")
                self._param_obs_counter = ray.get_actor(f"{self.name}.param_obs_counter")
            except ValueError:  # Name is not taken.
                self._param_update_counter = RaySharedVariable.options(name=f"{self.name}.param_update_counter").remote(0)
                self._param_obs_counter = RaySharedVariable.options(name=f"{self.name}.param_obs_counter").remote(0)
        else:
            from regym import SharedVariable
            self._param_update_counter = SharedVariable(0)
            self._param_obs_counter = SharedVariable(0)
        
    def reset_storages(self, nbr_actor: int=None):
        R2D2Algorithm.reset_storages(self, nbr_actor=nbr_actor)

        self.storage_buffers = [list() for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]
        return 

    def set_nbr_actor(self, nbr_actor):
        self.nbr_actor = nbr_actor

    def get_models(self):
        return {'model': self.model}

    def set_models(self, models_dict):
        if "model" in models_dict:
            hard_update(self.model, models_dict["model"])
    
    def compute_advantages_and_returns(
        self,
        storage_idx,
        non_episodic=False,
        normalizer=None,
    ):
        rs = []
        vs = []
        non_terminals = []

        for sidx in range(len(self.storages[storage_idx])):
            r = self.storages[storage_idx].r[0][sidx][0]
            v = self.storages[storage_idx].v[0][sidx][0]
            # (temporal_dim=unroll_length x 1)
            v_key = 'v'
            non_terminal = self.storages[storage_idx].non_terminal[0][sidx][0]#.squeeze().tolist()
            
            rs.append(r)
            vs.append(v)
            non_terminals.append(non_terminal)
        
        # (temporal_dim=unroll_length x 1)
        r = torch.cat(rs, dim=0)
        v = torch.cat(vs, dim=0)
        non_terminal = torch.cat(non_terminals, dim=0).squeeze().tolist()
        # (temporal_dim = nbr_storages * unroll_length x 1)
        
        # sidx contains the last segment of temporally-ordered data
        succ_s = self.storages[storage_idx].succ_s[0][sidx][0]
        rnn_states = self.storages[storage_idx].rnn_states[0][sidx]

        out_d = self._compute_advantages_and_returns(
            r=r,
            v=v,
            v_key=v_key,
            non_terminal=non_terminal,
            succ_s=succ_s,
            rnn_states=rnn_states,
            discount=self.kwargs['discount'],
            gae_tau=self.kwargs['gae_tau'],
            non_episodic=non_episodic,
            normalizer=normalizer,
        )
        
        advs = torch.split(
            out_d['advantages'], 
            self.sequence_replay_unroll_length,
        )
        
        std_advs = torch.split(
            standardize(out_d['advantages']),
            self.sequence_replay_unroll_length,
        )

        rets = torch.split(
            out_d['returns'], 
            self.sequence_replay_unroll_length,
        )
        
        for adv, std_adv, ret in zip(advs, std_advs, rets):
            self.storages[storage_idx].add(
                data={
                    'adv':adv.unsqueeze(0), 
                    'std_adv':std_adv.unsqueeze(0), 
                    'ret':ret.unsqueeze(0),
                },
            )
        
        return 

    def _compute_advantages_and_returns(
        self, 
        r,
        v,
        v_key,
        non_terminal,
        succ_s,
        rnn_states,
        discount,
        gae_tau,
        non_episodic=False,
        normalizer=None,
    ):
        torch.set_grad_enabled(False)
        self.model.train(False)

        ret = torch.zeros_like(r)
        adv = torch.zeros_like(r)

        if normalizer is not None:
            r = r / normalizer
        
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
        
        if non_terminal[-1]: 
            next_state = succ_s[-1].cuda() if self.kwargs['use_cuda'] else succ_s[-1]
            next_state = next_state.unsqueeze(0)
            if self.recurrent:
                seq_indices = [-1]
                final_rnn_states = _extract_rnn_states_from_seq_indices(
                    rnn_states, 
                    seq_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )
            final_prediction = next_state_value = self.model(
                next_state, 
                rnn_states=final_rnn_states,
            )
            returns = final_prediction[v_key].cpu().detach()
        else:
            returns = r[-1].reshape((1,1)) #torch.zeros(1,1)
        
        # Adding next state return/value and dummy advantages to the storage on the N+1 spots: 
        # not used during optimization, but necessary to compute the returns and advantages of previous states.
        ret[-1] = returns 
        adv[-1] = torch.zeros(1,1)
        # Adding next state value to the storage for the computation of gae for previous states:
        if isinstance(v, list):
            v.append(returns)
        else:
            assert isinstance(v, torch.Tensor)
            v = torch.cat([v, returns], dim=0)

        gae = 0.0
        for i in reversed(range(len(r))):
            if not self.kwargs['use_gae']:
                if non_episodic:    notdone = 1.0
                else:               notdone = non_terminal[i]
                returns = r[i] + discount * notdone * returns
                advantages = returns - v[i].detach()
            else:
                if non_episodic:    notdone = 1.0
                else:               notdone = non_terminal[i]
                td_error = r[i]  + discount * notdone * v[i + 1].detach() - v[i].detach()
                advantages = gae = td_error + discount * gae_tau * notdone * gae 
                returns = advantages + v[i].detach()
            adv[i] = advantages.detach()
            ret[i] = returns.detach()
        
        return {'returns':ret, 'advantages':adv}

    def train(self, minibatch_size:int=None):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer

        if minibatch_size is None:  minibatch_size = self.batch_size

        # Compute Returns and Advantages:
        start = time.time()
        for idx, storage in enumerate(self.storages): 
            #if len(storage) <= 1: continue
            #storage.placeholder()
            self.compute_advantages_and_returns(storage_idx=idx)
            '''
            if self.use_rnd: 
                self.compute_int_advantages_and_int_returns(storage_idx=idx, non_episodic=self.kwargs['rnd_non_episodic_int_r'])
            '''
        end = time.time()
        wandb.log({'PerUpdate/TimeComplexity/ComputeReturnsAdvantagesFn':  end-start}, commit=False) # self.param_update_counter)
        
        # Update observations running mean and std: 
        '''
        if self.use_rnd: 
            start = time.time()
            for idx, storage in enumerate(self.storages): 
                if len(storage) <= 1: continue
                self.obs_rms.update(storage.s)
            end = time.time()
            wandb.log({'PerUpdate/TimeComplexity/UpdateObsMeanStdFn':  end-start}, commit=False) # self.param_update_counter)
            self.obs_mean = self.obs_rms.mean
            self.obs_std = self.obs_rms.std
            # (1, *obs_shape)
        '''
        
        start = time.time()
        #samples = self.retrieve_values_from_storages()
        # PREVIOUSLY:
        #samples = self.retrieve_values_from_storages(minibatch_size=1)
        # NOW: 
        # in order to make sure to sample the whole dataset we put
        # minibatch_size to the size of the dataset:
        samples = self.retrieve_values_from_storages(minibatch_size=len(self.storages[0]))
        end = time.time()

        wandb.log({'PerUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)

        #if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)

        start = time.time()
        #self.optimize_model(minibatch_size, samples)
        for it in range(self.kwargs['optimization_epochs']):
            self.optimize_model(
                minibatch_size=self.nbr_actor,
                samples=samples,
            )
        end = time.time()
        
        wandb.log({'PerUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        self.reset_storages()
        
        return
    
    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal=False):        
        if self.storages is None:
            self.reset_storages()
        if not(with_replay_buffer): 
            storages = self.storages
            self.storages = None
            
        sum_writer = self.summary_writer
        self.summary_writer = None
        
        param_update_counter = self._param_update_counter
        self._param_update_counter = None 

        if isinstance(self.model, ArchiModel):
            self.model.reset()
        
        param_obs_counter = self._param_obs_counter
        self._param_obs_counter = None 

        cloned_algo = copy.deepcopy(self)
         
        if not(with_replay_buffer): 
            self.storages = storages
            # the following line might increase the size of the clone algo:
            if not minimal:
                cloned_algo.reset_storages()

        self.summary_writer = sum_writer
        
        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

        self._param_obs_counter = param_obs_counter
        cloned_algo._param_obs_counter = param_obs_counter

        # Goes through all variables 'Proxy' (dealing with multiprocessing)
        # contained in this class and removes them from clone
        if not(clone_proxies):
            proxy_key_values = [
                (key, value) 
                for key, value in cloned_algo.__dict__.items() 
                if ('Proxy' in str(type(value)))
            ]
            for key, value in proxy_key_values:
                setattr(cloned_algo, key, None)

        return cloned_algo

    
