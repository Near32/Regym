from typing import List, Dict
import torch
import torch.nn as nn
from .imagination_core import ImaginationCore
from ...networks import CategoricalActorCriticNet

class I2AModel(nn.Module):
    '''
    Refer to Figure 1 of original paper: https://arxiv.org/pdf/1707.06203.pdf
    for a visual representation of this model. The basic functionality of this
    model is to act as a policy network. It recieves a state / observation from
    the environment and returns (amongs other things) an action to be executed.
    '''
    def __init__(self, actor_critic_head: nn.Module,
                 model_free_network: nn.Module,
                 aggregator,
                 rollout_encoder: nn.Module,
                 imagination_core: ImaginationCore,
                 imagined_rollouts_per_step: int,
                 rollout_length: int,
                 rnn_keys: List[str],
                 latent_encoder: nn.Module,
                 kwargs: Dict[str, object]):
        '''
        :param actor_critic_head: nn.Module: Head of the model which outputs a
                                  distribution over actions and an estimation
                                  of the value of the state / observation. It
                                  receives as input the concatenation of:
                                      - The output of :param aggregator:
                                      - The output of :param model_free_network:
        :param model_free_network: nn.Module: Body of the module which takes
                                   a state / observation of the environment
                                   as input and outputs a feature vector
                                   to be used by the :param actor_critic_head:
        :param aggregator: Aggregator function which processes all rollout encodings
                           created by :param rollout_encoder:. The output of the
                           aggregator will be concatenated with the output of
                           :param model_free_network: to be passed as input
                           to :param actor_critic_head:
        :param rollout_encoder: nn.MModule: Recurrent Neural Net which takes
                                as input the trajectories generated by the
                                :param imagination_core: and individually encodes
                                each one into an `rollout embedding` which will
                                be passed as input to the :param aggregator:
        :param imagination_core: ImaginationCore: object containing a distill policy
                                 and an environment model which in conjunction are
                                 used to generate `imaginary` trajectories by using
                                 the distill policy and environment model to do
                                 forward planning.
        :param imagined_rollouts_per_step: int. Number of rollouts to generate
                                           through the :param imagination_core:
                                           for every forward pass of this model
        :param rollout_length: Number of steps to take on each forward rollout
                               by the :param imagination_core:
        :param rnn_keys: List of names of the submodules of this model which
                         feature recurrent models (LSTMs)
        :param kwargs: 
                      use_cuda: Bool: Whether or not to use CUDA for this model's computation
        '''
        super(I2AModel, self).__init__()

        self.actor_critic_head = actor_critic_head
        self.nbr_actions = None
        if isinstance(actor_critic_head, CategoricalActorCriticNet):
          self.nbr_actions = self.actor_critic_head.action_dim
        self.model_free_network = model_free_network
        self.aggregator = aggregator
        self.rollout_encoder = rollout_encoder
        self.imagination_core = imagination_core
        self.imagined_rollouts_per_step = imagined_rollouts_per_step
        self.rollout_length = rollout_length
        self.latent_encoder = latent_encoder
        self.embedded_state = None 
        self.kwargs = kwargs
        self.use_cuda = kwargs['use_cuda']
        self.rnn_keys = rnn_keys

        self.recurrent = False
        self.rnn_states = None
        if len(self.rnn_keys):
            self.recurrent = True
            self._reset_rnn_states()

        if self.use_cuda: self = self.cuda()

    def forward(self, state: torch.Tensor, action=None, rnn_states=None):
        '''
        :param state: torch.Tensor: preprocessed observation/state as a PyTorch Tensor
                      of dimensions batch_size=1 x input_shape
        :param action: action for which the log likelyhood will be computed.
        :param rnn_states: dictionnary of list of rnn_states if not None.
                           Used by the training algorithm, thus no need to pre/postprocess.
        :returns: the prediction dictionary returned from the
                  :self.actor_critic_head.__forward__(): method
        '''
        if self.kwargs['use_latent_embedding']:
          state = self.latent_encoder(state)
          self.embedded_state = state 

        rollout_embeddings = []
        first_action = None
        batch_size = state.size(0)
        for i in range(self.imagined_rollouts_per_step):
            # 0. Create the first action batch:
            if self.nbr_actions is not None and self.nbr_actions==self.imagined_rollouts_per_step: 
              first_action = i*torch.ones((batch_size)).long()
              if self.use_cuda: first_action = first_action.cuda()
            # 1. Imagine state and reward for self.imagined_rollouts_per_step times
            rollout_states, rollout_actions, rollout_rewards = self.imagination_core.imagine_rollout(state, self.rollout_length, first_action=first_action)
            # dimensions: self.rollout_length x batch x input_shape / action / reward-size
            # 2. encode them with RolloutEncoder and use aggregator to concatenate them together into imagination code
            rollout_embedding = self.rollout_encoder(rollout_states, rollout_rewards)
            # dimensions: batch x rollout_encoder_embedding_size
            firstactions = rollout_actions[0]
            # (batch x 1)
            onehot_firstactions = self._onehot_encode_actions(firstactions)
            # (batch x nbr_actions) 
            rollout_embedding_firstaction = torch.cat([rollout_embedding, onehot_firstactions], dim=1)
            # dimensions: batch x (rollout_encoder_embedding_size+nbr_actions)
            rollout_embeddings.append(rollout_embedding_firstaction.unsqueeze(1))
            
        rollout_embeddings = torch.cat(rollout_embeddings, dim=1)
        # dimensions: batch x self.imagined_rollouts_per_step x (rollout_encoder_embedding_size+nbr_actions)
        imagination_code = self.aggregator(rollout_embeddings)
        if self.use_cuda: imagination_code = imagination_code.cuda()
        # dimensions: batch x self.imagined_rollouts_per_step*(rollout_encoder_embedding_size+nbr_actions)
        # 3. model free pass
        features = self.model_free_network(state)
        
        # dimensions: batch x model_free_feature_dim
        # 4. concatenate model free pass and imagination code
        imagination_code_features = torch.cat([imagination_code, features], dim=1)

        # 5. Final fully connected layer which turns into action.
        if self.recurrent:
            if rnn_states is None:
                # Inference
                self._pre_process_rnn_states()
                rnn_states4achead, correspondance = self._remove_in_keys('achead_', self.rnn_states)
                prediction = self.actor_critic_head(imagination_code_features, rnn_states=rnn_states4achead, action=action)
                prediction = self._regularize_keys(prediction, correspondance)
                self._update_rnn_states(prediction)
            else:
                rnn_states4achead, correspondance = self._remove_in_keys('achead_', rnn_states)
                prediction = self.actor_critic_head(imagination_code_features, rnn_states=rnn_states4achead, action=action)
                prediction = self._regularize_keys(prediction, correspondance)
        else:
            prediction = self.actor_critic_head(imagination_code_features, action=action)

        return prediction

    def _onehot_encode_actions(self, action):
      '''
      :param action: tensor of shape (batch x 1)
      '''
      batch_size = action.size(0)
      action = action.long()
      onehot_actions = torch.zeros( batch_size, self.nbr_actions)
      if self.use_cuda: onehot_actions = onehot_actions.cuda()
      onehot_actions[range(batch_size), action] = 1
      return onehot_actions
      
    def _regularize_keys(self, prediction, correspondance):
        for recurrent_submodule_name in self.rnn_states:
            if self.rnn_states[recurrent_submodule_name] is None: continue
            corr_name = correspondance[recurrent_submodule_name]
            prediction['next_rnn_states'][recurrent_submodule_name] = prediction['next_rnn_states'].pop(corr_name)
            prediction['rnn_states'][recurrent_submodule_name] = prediction['rnn_states'].pop(corr_name)
        return prediction

    def _update_rnn_states(self, prediction):
        for recurrent_submodule_name in self.rnn_states:
            if self.rnn_states[recurrent_submodule_name] is None: continue
            for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                self.rnn_states[recurrent_submodule_name]['hidden'][idx] = prediction['next_rnn_states'][recurrent_submodule_name]['hidden'][idx].cpu()
                self.rnn_states[recurrent_submodule_name]['cell'][idx]   = prediction['next_rnn_states'][recurrent_submodule_name]['cell'][idx].cpu()
    
    def _pre_process_rnn_states(self, done=False):
        if done or self.rnn_states is None: self._reset_rnn_states()
        if self.use_cuda:
            for recurrent_submodule_name in self.rnn_states:
                if self.rnn_states[recurrent_submodule_name] is None: continue
                for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                    self.rnn_states[recurrent_submodule_name]['hidden'][idx] = self.rnn_states[recurrent_submodule_name]['hidden'][idx].cuda()
                    self.rnn_states[recurrent_submodule_name]['cell'][idx]   = self.rnn_states[recurrent_submodule_name]['cell'][idx].cuda()

    def _reset_rnn_states(self):
        self.rnn_states = {k: None for k in self.rnn_keys}
        for k in self.rnn_states:
            if 'phi' in k:
                self.rnn_states[k] = self.actor_critic_head.network.phi_body.get_reset_states(cuda=self.use_cuda)
            if 'critic' in k:
                self.rnn_states[k] = self.actor_critic_head.network.critic_body.get_reset_states(cuda=self.use_cuda)
            if 'actor' in k:
                self.rnn_states[k] = self.actor_critic_head.network.actor_body.get_reset_states(cuda=self.use_cuda)

    def _remove_in_keys(self, part2rm, dictionnary):
        newdict = {}
        corr = {}
        for k in dictionnary:
            if part2rm in k:
                newk = k.replace(part2rm, '')
                newdict[newk] = dictionnary[k]
                corr[k] = newk
            else: corr[k] = k
        return newdict, corr


#class I2AModel(nn.Module):
#  def __init__(self, actor_critic_head,
#               model_free_network,
#               aggregator,
#               rollout_encoder,
#               imagination_core,
#               imagined_rollouts_per_step,
#               rollout_length,
#               kwargs):
#    '''
#    :param imagined_rollouts_per_step: number of rollouts to
#                  imagine at each inference state.
#    :param rollout_length: nbr of steps per rollout.
#    '''
#    super(I2AModel, self).__init__()
#
#    self.actor_critic_head = actor_critic_head
#    self.model_free_network = model_free_network
#    self.aggregator = aggregator
#    self.rollout_encoder = rollout_encoder
#    self.imagination_core = imagination_core
#    self.imagined_rollouts_per_step = imagined_rollouts_per_step
#    self.rollout_length = rollout_length
#    self.kwargs = kwargs
#
#    self.recurrent = False
#    self.rnn_states = None
#    if len(self.rnn_keys):
#        self.recurrent = True
#        self._reset_rnn_states()
#
#    if self.kwargs['use_cuda']: self = self.cuda()
#
#
#
#  def forward(self, state, action=None, rnn_states=None):
#    '''
#    :param state: preprocessed observation/state as a PyTorch Tensor
#                  of dimensions batch_size=1 x input_shape
#    :param action: action for which the log likelyhood will be computed.
#    :param rnn_states: dictionnary of list of rnn_states if not None.
#                       Used by the training algorithm, thus no need to pre/postprocess.
#    '''
#    rollout_embeddings = []
#    for i in range(self.imagined_rollouts_per_step):
#        # 1. Imagine state and reward for self.imagined_rollouts_per_step times
#        rollout_states, rollout_rewards = self.imagination_core.imagine_rollout(state, self.rollout_length)
#        # dimensions: self.rollout_length x batch x input_shape / reward-size
#        # 2. encode them with RolloutEncoder:
#        rollout_embedding = self.rollout_encoder(rollout_states, rollout_rewards)
#        # dimensions: batch x rollout_encoder_embedding_size
#        rollout_embeddings.append(rollout_embedding.unsqueeze(1))
#    rollout_embeddings = torch.cat(rollout_embeddings, dim=1)
#    # dimensions: batch x self.imagined_rollouts_per_step x rollout_encoder_embedding_size
#    # 3. use aggregator to concatenate them together into imagination code:
#    imagination_code = self.aggregator(rollout_embeddings)
#    # dimensions: batch x self.imagined_rollouts_per_step*rollout_encoder_embedding_size
#    # 4. model free pass
#    features = self.model_free_network(state)
#    # dimensions: batch x model_free_feature_dim
#    # 5. concatenate model free pass and imagination code
#    imagination_code_features = torch.cat([imagination_code, features], dim=1)
#    # 6. Final actor critic module which turns the imagination code into action and value.
#    if self.recurrent:
#      if rnn_states is None:
#        self._pre_process_rnn_states()
#        rnn_states4achead, correspondance = _remove_in_keys('achead_', self.rnn_states)
#        prediction = self.actor_critic_head(imagination_code_features, rnn_states=rnn_states4achead, action=action)
#        self._update_rnn_states(prediction, correspondance=correspondance)
#      else:
#        prediction = self.actor_critic_head(imagination_code_features, rnn_states=rnn_states, action=action)
#    else:
#      prediction = self.actor_critic_head(imagination_code_features, action=action)
#
#    return prediction
