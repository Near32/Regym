from typing import Dict, List, Any, Optional, Callable

import copy
from collections import deque 
from functools import partial 

import ray
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt 

import regym
from regym.rl_algorithms.algorithms.algorithm import Algorithm
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from regym.rl_algorithms.algorithms.RecurrentPPO import recurrent_ppo_loss
from regym.rl_algorithms.replay_buffers import ReplayStorage, PrioritizedReplayStorage, SharedPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, concat_fn, _concatenate_hdict, _concatenate_list_hdict

import wandb
sum_writer = None



class RecurrentPPOAlgorithm(R2D2Algorithm):
    def __init__(
        self, 
        kwargs: Dict[str, Any], 
        model: nn.Module,
        optimizer=None,
        loss_fn: Callable = recurrent_ppo_loss.compute_loss,
        sum_writer=None,
        name='recurrent_ppo_algo',
        single_storage=True,
    ):
        
        Algorithm.__init__(self=self, name=name)
        self.single_storage = single_storage

        print(kwargs)

        self.sequence_replay_unroll_length = kwargs['sequence_replay_unroll_length']
        self.sequence_replay_overlap_length = kwargs['sequence_replay_overlap_length']
        self.sequence_replay_burn_in_length = kwargs['sequence_replay_burn_in_length']
        
        self.sequence_replay_store_on_terminal = kwargs["sequence_replay_store_on_terminal"]
        
        self.replay_buffer_capacity = kwargs['replay_capacity'] // (self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)
        
        assert kwargs['n_step'] < kwargs['sequence_replay_unroll_length']-kwargs['sequence_replay_burn_in_length'], \
                "Sequence_replay_unroll_length-sequence_replay_burn_in_length needs to be set to a value greater \
                 than n_step return, in order to be able to compute the bellman target."
        
        self.train_request_count = 0 

        self.kwargs = copy.deepcopy(kwargs)        
        self.use_cuda = kwargs["use_cuda"]
        self.nbr_actor = self.kwargs['nbr_actor']
        
        self.n_step = 1
        if self.kwargs.get('n_step', None) is not None:
            raise NotImplementedError
        
        if self.n_step > 1:
            self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]

        self.use_PER = self.kwargs['use_PER']
        self.use_HER = self.kwargs.get('use_HER', False)

        self.weights_decay_lambda = float(self.kwargs.get('weights_decay_lambda', 0.0))
        self.weights_entropy_lambda = float(self.kwargs.get('weights_entropy_lambda', 0.0))
        self.weights_entropy_reg_alpha = float(self.kwargs.get('weights_entropy_reg_alpha', 0.0))
        
        
        # TODO : self.min_capacity = int(float(kwargs["min_capacity"]))
        self.horizon = kwargs['horizon']
        self.min_capacity = horizon*self.nbr_actor
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
            'adv', 'ret', 'qa', 'log_pi_a',
            'mean', 'action_logits', 'succ_info',
        ]
        if self.recurrent:
            self.keys.append('rnn_states')
            #self.keys.append('next_rnn_states')
          
        # TODO: WARNING: rnn states can be handled that way but it is meaningless since dealing with sequences...
        self.circular_keys={'succ_s':'s'}
        # On the contrary to DQNAlgorithm,
        # since we are dealing with batches of unrolled experiences,
        # succ_s ought to be the sequence of unrolled experiences that comes
        # directly after the current unrolled sequence s:
        self.circular_offsets={'succ_s':1}
        
        # TODO: WARNING: rnn states can be handled that way but it is meaningless since dealing with sequences...
        if self.recurrent:
            self.circular_keys.update({'next_rnn_states':'rnn_states'})
            self.circular_offsets.update({'next_rnn_states':1})

        self.storages = None
        self.use_mp = False 
        if self.use_mp:
            #torch.multiprocessing.freeze_support()
            if not hasattr(regym, 'AlgoManager'):
                torch.multiprocessing.set_start_method("forkserver")#, force=True)
                regym.AlgoManager = mp.Manager()
            #regym.RegymManager = SyncManager()
            #regym.RegymManager.start()

            regym.samples = regym.AlgoManager.dict()
            regym.sampling_config = regym.AlgoManager.dict()
            regym.sampling_config['stay_on'] = True
            regym.sampling_config['keep_sampling'] = False
            regym.sampling_config['minibatch_size'] = 32
            
            self.reset_storages()
            
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            
            regym.sampling_process = mp.Process(
                target=self._mp_sampling,
                kwargs={
                    'storages':self.storages,
                    'sampling_config':regym.sampling_config,
                    'samples':regym.samples,
                },
            )
            regym.sampling_process.start()
        else:
            self.reset_storages()
        
        self.storage_buffers = [list() for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]

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
        
    def get_models(self):
        return {'model': self.model}

    def set_models(self, models_dict):
        if "model" in models_dict:
            hard_update(self.model, models_dict["model"])
    
    def train(self, minibatch_size:int=None):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer

        if minibatch_size is None:  minibatch_size = self.batch_size

        start = time.time()
        if self.use_mp:
            samples = self._retrieve_values_from_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        else:
            samples = self.retrieve_values_from_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        end = time.time()

        wandb.log({'PerUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)


        start = time.time()
        self.optimize_model(minibatch_size, samples)
        end = time.time()
        
        wandb.log({'PerUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
      
