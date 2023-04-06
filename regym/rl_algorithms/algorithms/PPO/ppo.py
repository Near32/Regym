from typing import Dict, List 

import copy
import time 
from functools import partial 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import ray

import regym
from ..algorithm import Algorithm
from regym.rl_algorithms.utils import _extract_rnn_states_from_batch_indices
from regym.rl_algorithms.utils import _concatenate_hdict, _concatenate_list_hdict
from ...networks import hard_update, random_sample
from ...replay_buffers import Storage

from regym.rl_algorithms.utils import archi_concat_fn
from regym.thirdparty.Archi.Archi.model import Model as ArchiModel

from . import ppo_loss, rnd_loss, ppo_vae_loss
from . import ppo_actor_loss, ppo_critic_loss

import wandb
summary_writer = None 

import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager
from multiprocessing.managers import NamespaceProxy
import types 

class SProxy(NamespaceProxy):
    _exposed_ = tuple(dir(Storage))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args, kwargs)
            return wrapper
        return result
    def __len__(self):
        callmethod = object.__getattribute__(self, "_callmethod")        
        return callmethod("__len__")

SyncManager.register('Storage', Storage, SProxy)

import logging 


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


class PPOAlgorithm(Algorithm):
    def __init__(
        self, 
        kwargs, 
        model, 
        optimizer=None, 
        target_intr_model=None, 
        predict_intr_model=None, 
        loss_fn=ppo_loss.compute_loss,
        sum_writer=None,
        name="ppo_algo"):
        '''
        TODO specify which values live inside of kwargs
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
        super(PPOAlgorithm, self).__init__(name=name)

        self.train_request_count =0 

        self.kwargs = copy.deepcopy(kwargs)
        self.use_cuda = kwargs['use_cuda']
        self.nbr_actor = self.kwargs['nbr_actor']
        
        self.use_rnd = False
        if target_intr_model is not None and predict_intr_model is not None:
            self.use_rnd = True
            self.target_intr_model = target_intr_model
            self.predict_intr_model = predict_intr_model
            #self.obs_rms = RunningMeanStd(mean=0.0, std=0.0, shape=(self.kwargs['input_shape']))
            self.obs_rms = RunningMeanStd(mean=None, std=None)
            self.obs_mean = self.obs_rms.mean
            self.obs_std = self.obs_rms.std
            self.discounted_int_returns = RewardForwardFilter(gamma=self.kwargs['intrinsic_discount'])
            self.int_reward_rms = RunningMeanStd(mean=0.0, std=1.0)
        
        self.goal_oriented = self.kwargs.get('goal_oriented', False)

        self.weights_decay_lambda = float(self.kwargs.get('weights_decay_lambda', 0.0))
        self.weights_entropy_lambda = float(self.kwargs.get('weights_entropy_lambda', 0.0))
        self.weights_entropy_reg_alpha = float(self.kwargs.get('weights_entropy_reg_alpha', 0.0))

        self.running_counter_extrinsic_reward = 0
        self.ext_reward_mean = 0.0
        self.ext_reward_std = 1.0
            
        self.use_vae = self.kwargs.get('use_vae', False)

        self.model = model
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.use_rnd:
                self.target_intr_model = self.target_intr_model.cuda()
                self.predict_intr_model = self.predict_intr_model.cuda()
        

        if optimizer is None:
            parameters = self.model.parameters()
            # TODO : find out whether the RND weights should be optimized separately ...
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = self.kwargs['learning_rate'] 
            if self.kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            
            if self.use_rnd: 
                #parameters = list(parameters)+list(self.predict_intr_model.parameters())
                self.optimizer_rnd = optim.Adam(
                    self.predict_intr_model.parameters(), 
                    lr=lr,
                    #TODO: find original paper values: betas=(0.9, 0.999),
                    eps=float(self.kwargs['adam_eps']),
                    weight_decay=float(self.kwargs.get('adam_weight_decay', 0.0)),
                )
            
            self.optimizer = optim.Adam(
                parameters, 
                lr=lr,
                #TODO: find original paper values: betas=(0.9, 0.999),
                eps=float(self.kwargs['adam_eps']),
                weight_decay=float(self.kwargs.get('adam_weight_decay', 0.0)),
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

        self.storages = None
        # TODO ; integrate mp usage here if needs be, cf dqn.py
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
    
    def parameters(self):
        parameters = self.model.parameters()
        # TODO : find out whether the RND weights should be optimized separately ...
        if self.use_rnd: 
            parameters = list(parameters)+list(self.predict_intr_model.parameters())
        return parameters

    @property
    def param_update_counter(self):
        if isinstance(self._param_update_counter, ray.actor.ActorHandle):
            return ray.get(self._param_update_counter.get.remote())    
        else:
            return self._param_update_counter.get()

    @param_update_counter.setter
    def param_update_counter(self, val):
        if isinstance(self._param_update_counter, ray.actor.ActorHandle):
            self._param_update_counter.set.remote(val) 
        else:
            self._param_update_counter.set(val)
    
    @property
    def param_obs_counter(self):
        if isinstance(self._param_obs_counter, ray.actor.ActorHandle):
            return ray.get(self._param_obs_counter.get.remote())    
        else:
            return self._param_obs_counter.get()

    @param_obs_counter.setter
    def param_obs_counter(self, val):
        if isinstance(self._param_obs_counter, ray.actor.ActorHandle):
            self._param_obs_counter.set.remote(val) 
        else:
            self._param_obs_counter.set(val)
    
    def set_optimizer(self, optimizer):
        self.optimizer.load_state_dict(optimizer.state_dict())

    def get_optimizer(self):
        return self.optimizer

    def get_models(self):
        return {'model': self.model}

    def set_models(self, models_dict):
        if "model" in models_dict:
            hard_update(self.model, models_dict["model"])
        
    def get_nbr_actor(self):
        nbr_actor = self.nbr_actor
        return nbr_actor

    def set_nbr_actor(self, nbr_actor):
        self.nbr_actor = nbr_actor

    def get_update_count(self):
        return self.param_update_counter

    def get_obs_count(self):
        return self.param_obs_counter

    def reset_storages(self, nbr_actor=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

        if self.storages is not None:
            for storage in self.storages: storage.reset()
            return 

        keys = [
            's', 'a', 'r', 'succ_s', 'non_terminal', 'info',
            'v', 'q', 'pi', 'log_pi', 'ent', 'greedy_action',
            'adv', 'ret', 'qa', 'log_pi_a',
            'mean', 'action_logits', 'succ_info',
        ]
        self.storages = []
        for i in range(self.nbr_actor):
            self.storages.append(Storage())
            if self.recurrent:
                self.storages[-1].add_key('rnn_states')
                self.storages[-1].add_key('next_rnn_states')
            if self.use_rnd:
                self.storages[-1].add_key('int_r')
                self.storages[-1].add_key('int_v')
                self.storages[-1].add_key('int_ret')
                self.storages[-1].add_key('int_adv')
                self.storages[-1].add_key('target_int_f')

    def stored_experiences(self):
        self.train_request_count += 1
        nbr_stored_experiences = sum([len(storage) for storage in self.storages])

        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        wandb.log({'PerTrainingRequest/NbrStoredExperiences':  nbr_stored_experiences}, commit=False) # self.train_request_count)
        
        return nbr_stored_experiences

    def normalize_ext_rewards(self, storage_idx):
        normalized_ext_rewards = []
        for i in range(len(self.storages[storage_idx].r)):
            #normalized_ext_rewards.append(self.storages[storage_idx].r[i] / (self.ext_reward_std+1e-8))
            # Proper normalization to standard gaussian:
            normalized_ext_rewards.append( (self.storages[storage_idx].r[i]-self.ext_reward_mean) / (self.ext_reward_std+1e-8))
        return normalized_ext_rewards

    def normalize_int_rewards(self, storage_idx):
        normalized_int_rewards = []
        for i in range(len(self.storages[storage_idx].int_r)):
            # Scaling alone:
            normalized_int_rewards.append(self.storages[storage_idx].int_r[i] / (self.int_reward_std+1e-8))
        return normalized_int_rewards

    def normalize_int_rewards_by_discounted_int_returns(self, storage_idx):
        normalized_int_rewards = []
        for i in range(len(self.storages[storage_idx].int_r)):
            # Scaling alone:
            normalized_int_rewards.append(self.storages[storage_idx].int_r[i] / (self.int_reward_rms.std+1e-8))
        return normalized_int_rewards

    def compute_advantages_and_returns(self, storage_idx, non_episodic=False):
        torch.set_grad_enabled(False)
        ext_r = self.storages[storage_idx].r
        
        # it is indeed not necessary to
        # normalize the Extrinsic reward:
        #norm_ext_r = self.normalize_ext_rewards(storage_idx)
        
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
        
        if self.storages[storage_idx].non_terminal[-1]: 
            next_state = self.storages[storage_idx].succ_s[-1].cuda() if self.kwargs['use_cuda'] else self.storages[storage_idx].succ_s[-1]
            rnn_states = None
            if self.recurrent:
                rnn_states = self.storages[storage_idx].rnn_states[-1]
            returns = next_state_value = self.model(next_state, rnn_states=rnn_states)['v'].cpu().detach()
        else:
            returns = torch.zeros(1,1)
        # Adding next state return/value and dummy advantages to the storage on the N+1 spots: 
        # not used during optimization, but necessary to compute the returns and advantages of previous states.
        self.storages[storage_idx].ret[-1] = returns 
        self.storages[storage_idx].adv[-1] = torch.zeros(1,1)
        # Adding next state value to the storage for the computation of gae for previous states:
        self.storages[storage_idx].v.append(returns)
        
        #wandb.log({f"Training/FinalStateExtReturn_actor{storage_idx}":returns.item()}, commit=False)

        gae = 0.0
        #for i in reversed(range(len(self.storages[storage_idx])-1)):
        for i in reversed(range(len(self.storages[storage_idx].r))):
            if not self.kwargs['use_gae']:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                returns = ext_r[i] + self.kwargs['discount'] * notdone * returns
                #returns = norm_ext_r[i] + self.kwargs['discount'] * notdone * returns
                advantages = returns - self.storages[storage_idx].v[i].detach()
            else:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                td_error = ext_r[i]  + self.kwargs['discount'] * notdone * self.storages[storage_idx].v[i + 1].detach() - self.storages[storage_idx].v[i].detach()
                #td_error = norm_ext_r[i]  + self.kwargs['discount'] * notdone * self.storages[storage_idx].v[i + 1].detach() - self.storages[storage_idx].v[i].detach()
                advantages = gae = td_error + self.kwargs['discount'] * self.kwargs['gae_tau'] * notdone * gae 
                returns = advantages + self.storages[storage_idx].v[i].detach()
            self.storages[storage_idx].adv[i] = advantages.detach()
            self.storages[storage_idx].ret[i] = returns.detach()
        
        #wandb.log({f"Training/MeanExtReturn_actor{storage_idx}": sum(self.storages[storage_idx].ret).mean().item()/self.kwargs['horizon']}, commit=False)

    def compute_int_advantages_and_int_returns(self, storage_idx, non_episodic=True):
        '''
        Compute intrinsic returns and advantages from normalized intrinsic rewards.
        Indeed, int_r values in storages have been normalized upon computation.
        At computation-time, updates of the running mean and std are performed too.
        '''
        torch.set_grad_enabled(False)
        # TODO: the following is too much apparently,
        #norm_int_r = self.normalize_int_rewards(storage_idx)
        # just dividing by the discounted_int_return STD is enough:
        norm_int_r = self.normalize_int_rewards_by_discounted_int_returns(storage_idx)
        int_advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
        
        #int_returns = self.storages[storage_idx].int_v[-1].detach()
        if self.storages[storage_idx].non_terminal[-1]: 
            next_state = self.storages[storage_idx].succ_s[-1]
            int_returns, _ = self.compute_intrinsic_reward(next_state)
            int_returns.unsqueeze_(0).unsqueeze_(1)
            # Normalization (scaling):
            int_returns = int_returns / (self.int_reward_rms.std+1e-8)
        else:
            int_returns = torch.zeros(1,1)
        # Adding next state return/value and dummy advantages to the storage on the N+1 spots: 
        # not used during optimization, but necessary to compute the returns and advantages of previous states.
        self.storages[storage_idx].int_ret[-1] = int_returns 
        self.storages[storage_idx].int_adv[-1] = torch.zeros(1,1)
        # Adding next intrinsic state value to the storage for the computation of gae for previous states:
        self.storages[storage_idx].int_v.append(int_returns)
        
        gae = 0.0
        #for i in reversed(range(len(self.storages[storage_idx].int_r)-1)):
        for i in reversed(range(len(self.storages[storage_idx].int_r))):
            if not self.kwargs['use_gae']:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                int_returns = norm_int_r[i] + self.kwargs['intrinsic_discount'] * notdone * int_returns
                int_advantages = int_returns - self.storages[storage_idx].int_v[i].detach()
            else:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                td_error = norm_int_r[i]  + self.kwargs['intrinsic_discount'] * notdone * self.storages[storage_idx].int_v[i + 1].detach() - self.storages[storage_idx].int_v[i].detach()
                int_advantages = gae = td_error + self.kwargs['intrinsic_discount'] * self.kwargs['gae_tau'] * notdone * gae 
                int_returns = int_advantages + self.storages[storage_idx].int_v[i].detach()
            self.storages[storage_idx].int_adv[i] = int_advantages.detach()
            self.storages[storage_idx].int_ret[i] = int_returns.detach()

    def train(self):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        if summary_writer is None:
            summary_writer = self.summary_writer
            
        # Normalize Int Return :
        if self.use_rnd:
            start = time.time()
            pstep_pactor_int_reward = []
            for idx, storage in enumerate(self.storages):
                if len(storage) <= 1: continue
                storage.placeholder()
                pstep_pactor_int_reward.append(torch.stack(storage.int_r))
            pstep_pactor_int_reward = torch.stack(pstep_pactor_int_reward, dim=1)
            # (horizon x nbr_actors) 
            pstep_pactor_curiosity_discounted_int_returns = torch.stack(
                [self.discounted_int_returns.update(pstep_int_reward)
                for pstep_int_reward in pstep_pactor_int_reward.data],
                dim=1,
            )
            # (horizon x nbr_actors)
            bmean, bstd, bcount = (
                torch.mean(pstep_pactor_curiosity_discounted_int_returns),
                torch.std(pstep_pactor_curiosity_discounted_int_returns),
                pstep_pactor_curiosity_discounted_int_returns.shape[1],
            )
            
            self.int_reward_rms.update_from_moments(
                bmean=bmean,
                bstd=bstd,
                bcount=bcount,
            )

            end = time.time()
            wandb.log({'PerUpdate/TimeComplexity/UpdateIntRewardRunningMeanStdFn':  end-start}, commit=False) # self.param_update_counter)
        
        # Compute Returns and Advantages:
        start = time.time()
        for idx, storage in enumerate(self.storages): 
            if len(storage) <= 1: continue
            storage.placeholder()
            self.compute_advantages_and_returns(storage_idx=idx)
            if self.use_rnd: 
                self.compute_int_advantages_and_int_returns(storage_idx=idx, non_episodic=self.kwargs['rnd_non_episodic_int_r'])
        end = time.time()
        wandb.log({'PerUpdate/TimeComplexity/ComputeReturnsAdvantagesFn':  end-start}, commit=False) # self.param_update_counter)
        
        # Update observations running mean and std: 
        start = time.time()
        if self.use_rnd: 
            for idx, storage in enumerate(self.storages): 
                if len(storage) <= 1: continue
                self.obs_rms.update(storage.s)
                #for ob in storage.s: self.update_obs_mean_std(ob)
        end = time.time()
        wandb.log({'PerUpdate/TimeComplexity/UpdateObsMeanStdFn':  end-start}, commit=False) # self.param_update_counter)
        self.obs_mean = self.obs_rms.mean
        self.obs_std = self.obs_rms.std
        # (1, *obs_shape)

        # states, actions, next_states, log_probs_old, returns, advantages, std_advantages, \
        # int_returns, int_advantages, std_int_advantages, \
        # target_random_features, rnn_states = self.retrieve_values_from_storages()
        start = time.time()
        samples = self.retrieve_values_from_storages()
        end = time.time()

        wandb.log({'PerUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)

        #if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)

        start = time.time()
        for it in range(self.kwargs['optimization_epochs']):
            self.optimize_model(samples)
        end = time.time()
        
        wandb.log({'PerUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        self.reset_storages()
        #if self.running_counter_intrinsic_reward >= self.kwargs['horizon']*self.nbr_actor: #self.update_period_intrinsic_reward:
        #  self.running_counter_intrinsic_reward = 0


    def retrieve_values_from_storages(self):
        '''
        Each storage stores in their key entries either numpy arrays or hierarchical dictionnaries of numpy arrays.
        This function samples from each storage, concatenate the sampled elements on the batch dimension,
        and maintains the hierarchy of dictionnaries.
        '''
        torch.set_grad_enabled(False)
        keys=['s', 'a', 'log_pi_a', 'ret', 'adv']

        fulls = {}
        
        if self.use_rnd:
            keys += ['succ_s', 'int_ret', 'int_adv', 'target_int_f']
            
        if self.recurrent:
            keys += ['rnn_states'] #, 'next_rnn_states']
        
        """
        # depr : goal update
        if self.goal_oriented:
            keys += ['g']
        """

        for key in keys:    
            fulls[key] = []

        for storage in self.storages:
            # Check that there is something in the storage 
            storage_size = len(storage)
                
            if storage_size <= 1: continue
            sample = storage.cat(keys)
            
            values = {}
            for key, value in zip(keys, sample):
                #value = value.tolist()
                if isinstance(value[0], dict): 
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        #concat_fn=partial(torch.cat, dim=0),   # concatenate on the unrolling dimension (axis=1).
                        concat_fn=archi_concat_fn,
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
                values[key] = value 

            for key, value in values.items():
                fulls[key].append(value)
        
        out_fulls = {}
        for key, value in fulls.items():
            if len(value) >1:
                if isinstance(value[0], dict):
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        concat_fn=partial(torch.cat, dim=0),   # concatenate on the unrolling dimension (axis=1).
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
            else:
                value = value[0]

            out_fulls[key] = value
            if 'adv' in key:
                out_fulls[f'std_{key}'] = self.standardize(value).squeeze()

        return out_fulls

    def standardize(self, x):
        stable_eps = 1.0e-8 #1e-30
        return (x - x.mean()) / (x.std()+stable_eps)

    def compute_intrinsic_reward(self, states):
        if self.obs_rms.mean is not None:
            normalized_states = (states-self.obs_rms.mean) / (self.obs_rms.std+1e-8) 
        else:
            normalized_states = states

        if self.kwargs['rnd_obs_clip'] > 1e-3:
          normalized_states = torch.clamp( normalized_states, -self.kwargs['rnd_obs_clip'], self.kwargs['rnd_obs_clip'])
        if self.kwargs['use_cuda']: normalized_states = normalized_states.cuda()
        
        pred_features = self.predict_intr_model(normalized_states)
        target_features = self.target_intr_model(normalized_states)
        
        # Clamping:
        #pred_features = torch.clamp(pred_features, -1e20, 1e20)
        #target_features = torch.clamp(target_features, -1e20, 1e20)
        
        # Softmax:
        #pred_features = F.softmax(pred_features)
        #softmax_target_features = F.softmax(target_features)
        if torch.isnan(pred_features).long().sum().item() or torch.isnan(target_features).long().sum().item():
            import ipdb; ipdb.set_trace()
        #int_reward = torch.nn.functional.smooth_l1_loss(target_features,pred_features)
        
        #int_reward = torch.nn.functional.mse_loss(target_features,pred_features)
        int_reward = ((target_features-pred_features).pow(2).sum(1) / 2).data
        #int_reward = torch.nn.functional.mse_loss(softmax_target_features,pred_features)
        
        # No clipping on the intrinsic reward in the original paper:
        #int_reward = torch.clamp(int_reward, -1, 1)
        int_reward = int_reward.detach().cpu().squeeze()
        # TODO self.update_int_reward_mean_std(int_reward)

        # Normalization will be done upon usage...
        # Kept intact here for logging purposes...        
        #int_r = int_reward / (self.int_reward_std+1e-8)

        return int_reward, target_features.detach().cpu()

    def update_ext_reward_mean_std(self, unnormalized_er_list):
        for unnormalized_er in unnormalized_er_list:
            rmean = self.ext_reward_mean
            rstd = self.ext_reward_std
            rc = self.running_counter_extrinsic_reward

            self.running_counter_extrinsic_reward += 1
            
            self.ext_reward_mean = (self.ext_reward_mean*rc+unnormalized_er)/self.running_counter_extrinsic_reward
            self.ext_reward_std = np.sqrt( ( np.power(self.ext_reward_std,2)*rc+np.power(unnormalized_er-rmean, 2) ) / self.running_counter_extrinsic_reward )
        
    def update_int_reward_mean_std_batch(self, ir):
        rmean = self.int_reward_mean*torch.ones(1)
        rstd = self.int_reward_std*torch.ones(1)
        rc = self.running_counter_intrinsic_reward
        
        ir = torch.stack(ir)
        bmean = torch.mean(ir)
        bvar = torch.var(ir)
        bcount = len(ir)

        delta = bmean - rmean
        tot_count = rc + bcount
        
        new_mean = rmean + delta * bcount / tot_count
        
        var = torch.square(rstd)
        m_a = var * rc 
        m_b = bvar * bcount
        M2 = m_a + m_b + torch.square(delta) * rc * bcount / tot_count
        new_var = M2 / tot_count
        
        self.int_reward_mean = new_mean
        self.int_reward_std = torch.sqrt(new_var)
        self.running_counter_intrinsic_reward = tot_count
        wandb.log({f"Training/RunningIntRewardCounter": self.running_counter_intrinsic_reward}, commit=False)
        
    def update_int_reward_mean_std(self, unnormalized_ir):
        rmean = self.int_reward_mean
        rstd = self.int_reward_std
        rc = self.running_counter_intrinsic_reward

        self.running_counter_intrinsic_reward += 1
        wandb.log({f"Training/RunningIntRewardCounter": self.running_counter_intrinsic_reward}, commit=False)
        
        self.int_reward_mean = (self.int_reward_mean*rc+unnormalized_ir)/self.running_counter_intrinsic_reward
        self.int_reward_std = np.sqrt( 
                (np.power(self.int_reward_std,2)*rc+np.power(unnormalized_ir-rmean, 2) ) / self.running_counter_intrinsic_reward )
        
        # TODO : the following is moved and the condition is changed
        # for the update to occur after each update of the agent occuring.
        #if self.running_counter_intrinsic_reward >= self.update_period_intrinsic_reward:
        #  self.running_counter_intrinsic_reward = 0

    def update_obs_mean_std(self, unnormalized_obs):
        torch.set_grad_enabled(False)
        rmean = self.obs_mean
        rstd = self.obs_std
        rc = self.running_counter_obs

        self.running_counter_obs += 1
        
        self.obs_mean = (self.obs_mean*rc+unnormalized_obs)/self.running_counter_obs
        self.obs_std = np.sqrt( ( np.power(self.obs_std,2)*rc+np.power(unnormalized_obs-rmean, 2) ) / self.running_counter_obs )
        
        '''
        wandb.log({
            #f"RND/obs_mean/Mean": self.obs_mean.mean().item(),
            #f"RND/obs_mean/Std": self.obs_mean.std().item(),
            #f"RND/obs_std/Mean": self.obs_std.mean().item(),
            #f"RND/obs_std/Std": self.obs_std.std().item(),
            f"RND/running_counter_obs": self.running_counter_obs,
            f"RND/update_period_obs": self.update_period_obs,
            },
            commit = False,
        )
        '''

        # TODO : if self.running_counter_obs >= self.update_period_obs:
        #   self.running_counter_obs = 0

    def optimize_model(self, samples):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        # What is this: create dictionary to store length of each part of the recurrent submodules of the current model
        # nbr_layers_per_rnn = None
        # if self.recurrent:
        #     nbr_layers_per_rnn = {recurrent_submodule_name: len(rnn_states[recurrent_submodule_name]['hidden'])
        #                           for recurrent_submodule_name in rnn_states}

        start = time.time()
        torch.set_grad_enabled(True)
        self.model.train(True)
        if self.use_rnd:    self.predict_intr_model.train(True)
        
        states = samples['s']
        rnn_states = samples['rnn_states']
        actions = samples['a']
        log_probs_old = samples['log_pi_a']
        returns = samples['ret']
        advantages = samples['adv']
        std_advantages = samples['std_adv']

        if self.use_rnd:
            next_states = samples['succ_s']
            int_returns = samples['int_ret']
            int_advantages = samples['int_adv']
            std_int_advantages = samples['std_int_adv']
            target_random_features = samples['target_int_f']

        if self.kwargs['mini_batch_size'] == 'None':
            sampler = [np.arange(samples['s'].size(0))]
        else: 
            sampler = random_sample(np.arange(samples['s'].size(0)), self.kwargs['mini_batch_size'])
            #sampler = random_sample(np.arange(advantages.size(0)), self.kwargs['mini_batch_size'])
        sampler = list(sampler)
        nbr_minibatches = len(sampler)

        #self.optimizer.zero_grad()

        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            
            sampled_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]
            sampled_std_advantages = std_advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else std_advantages[batch_indices]
                
            sampled_states = sampled_states.detach()
            sampled_actions = sampled_actions.detach()
            sampled_log_probs_old = sampled_log_probs_old.detach()
            sampled_returns = sampled_returns.detach()
            sampled_advantages = sampled_advantages.detach()
            sampled_std_advantages = sampled_std_advantages.detach()

            if self.use_rnd:
                sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
                sampled_next_states = sampled_next_states.detach()
                sampled_int_returns = int_returns[batch_indices].cuda() if self.kwargs['use_cuda'] else int_returns[batch_indices]
                sampled_int_advantages = int_advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else int_advantages[batch_indices]
                sampled_std_int_advantages = std_int_advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else std_int_advantages[batch_indices]
                sampled_target_random_features = target_random_features[batch_indices].cuda() if self.kwargs['use_cuda'] else target_random_features[batch_indices]
                
                sampled_int_returns = sampled_int_returns.detach()
                sampled_int_advantages = sampled_int_advantages.detach()
                sampled_std_int_advantages = sampled_std_int_advantages.detach()
                sampled_target_random_features = sampled_target_random_features.detach()
                states_mean = self.obs_mean.cuda() if self.kwargs['use_cuda'] else self.obs_mean
                states_std = self.obs_std.cuda() if self.kwargs['use_cuda'] else self.obs_std

            #self.optimizer.zero_grad()
            if self.use_rnd:
                loss = rnd_loss.compute_loss(sampled_states, 
                                             sampled_actions, 
                                             sampled_next_states,
                                             sampled_log_probs_old,
                                             ext_returns=sampled_returns, 
                                             ext_advantages=sampled_advantages,
                                             std_ext_advantages=sampled_std_advantages,
                                             int_returns=sampled_int_returns, 
                                             int_advantages=sampled_int_advantages, 
                                             std_int_advantages=sampled_std_int_advantages,
                                             target_random_features=sampled_target_random_features,
                                             states_mean=states_mean, 
                                             states_std=states_std,
                                             rnn_states=sampled_rnn_states,
                                             ratio_clip=self.kwargs['ppo_ratio_clip'], 
                                             entropy_weight=self.kwargs['entropy_weight'],
                                             value_weight=self.kwargs['value_weight'],
                                             rnd_weight=self.kwargs['rnd_weight'],
                                             model=self.model,
                                             rnd_obs_clip=self.kwargs['rnd_obs_clip'],
                                             pred_intr_model=self.predict_intr_model,
                                             intrinsic_reward_ratio=self.kwargs['rnd_loss_int_ratio'],
                                             iteration_count=self.param_update_counter,
                                             summary_writer=self.summary_writer )
            elif self.use_vae:
                loss = ppo_vae_loss.compute_loss(sampled_states, 
                                             sampled_actions, 
                                             sampled_log_probs_old,
                                             sampled_returns, 
                                             sampled_advantages, 
                                             sampled_std_advantages,
                                             rnn_states=sampled_rnn_states,
                                             ratio_clip=self.kwargs['ppo_ratio_clip'], 
                                             entropy_weight=self.kwargs['entropy_weight'],
                                             value_weight=self.kwargs['value_weight'],
                                             vae_weight=self.kwargs['vae_weight'],
                                             model=self.model,
                                             iteration_count=self.param_update_counter,
                                             summary_writer=self.summary_writer)
            else:
                loss = ppo_loss.compute_loss(
                    sampled_states, 
                    sampled_actions, 
                    sampled_log_probs_old,
                    sampled_returns, 
                    sampled_advantages, 
                    sampled_std_advantages,
                    rnn_states=sampled_rnn_states,
                    use_std_adv=self.kwargs['standardized_adv'],
                    ratio_clip=self.kwargs['ppo_ratio_clip'], 
                    entropy_weight=self.kwargs['entropy_weight'],
                    value_weight=self.kwargs['value_weight'],
                    model=self.model,
                    iteration_count=self.param_update_counter,
                    summary_writer=self.summary_writer,
                )

            '''
            (loss/nbr_minibatches).backward(retain_graph=False)
            '''
            self.optimizer.zero_grad()
            if self.use_rnd:
                self.optimizer_rnd.zero_grad()

            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()
            if self.use_rnd:
                # IMPORTANT : no gradient clipping for the RND loss ...
                self.optimizer_rnd.step()

            self.param_update_counter += 1 
            '''
            for name, param in self.model.named_parameters():
                if hasattr(param, 'grad') and param.grad is not None:
                    summary_writer.add_histogram(f"Training/{name}", param.grad.cpu(), self.param_update_counter)
            '''
        
        '''
        if self.kwargs['gradient_clip'] > 1e-3:
            nn.utils.clip_grad_norm_(self.parameters(), self.kwargs['gradient_clip'])
        self.optimizer.step()
        '''
        self.model.train(False)
        if self.use_rnd:    self.predict_intr_model.train(False)
        torch.set_grad_enabled(False)

        end = time.time()
        wandb.log({'PerUpdate/TimeComplexity/OptimizationLoss':  end-start}, commit=False) # self.param_update_counter)
        if self.use_rnd:
            wandb.log({
                'Training/IntRewardMean':  self.int_reward_rms.mean.cpu().item(), 
                'Training/IntRewardStd': self.int_reward_rms.std.cpu().item(), 
                },
                commit=False,
            )

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

        param_obs_counter = self._param_obs_counter
        self._param_obs_counter = None 

        if isinstance(self.model, ArchiModel):
            self.model.reset()
        
        cloned_algo = copy.deepcopy(self)
        
        if not(with_replay_buffer): 
            self.storages = storages
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

    def async_actor(self):        
        storages = self.storages
        self.storages = None
        
        sum_writer = self.summary_writer
        self.summary_writer = None
        
        param_update_counter = self._param_update_counter
        self._param_update_counter = None 

        param_obs_counter = self._param_obs_counter
        self._param_obs_counter = None 

        cloned_algo = copy.deepcopy(self)
        
        self.storages = storages
        cloned_algo.storages = storages

        self.summary_writer = sum_writer
        cloned_algo.summary_writer = sum_writer

        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

        self._param_obs_counter = param_obs_counter
        cloned_algo._param_obs_counter = param_obs_counter

        return cloned_algo

    @staticmethod
    def check_mandatory_kwarg_arguments(kwargs: dict):
        '''
        Checks that all mandatory hyperparameters are present
        inside of dictionary :param kwargs:

        :param kwargs: Dictionary of hyperparameters
        '''
        # Future improvement: add a condition to check_kwarg (discount should be between (0:1])
        keywords = ['horizon', 'discount', 'use_gae', 'gae_tau', 'use_cuda',
                    'entropy_weight', 'gradient_clip', 'optimization_epochs',
                    'mini_batch_size', 'ppo_ratio_clip', 'learning_rate', 'adam_eps']

        def check_kwarg_and_condition(keyword, kwargs):
            if keyword not in kwargs:
                raise ValueError(f"Keyword: '{keyword}' not found in kwargs")
        for keyword in keywords: check_kwarg_and_condition(keyword, kwargs)

