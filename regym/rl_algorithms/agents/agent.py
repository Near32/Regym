import torch
import numpy as np 


def named_children(cm):
    for name, m in cm._modules.items():
        if m is not None:
            yield name, m

def look_for_keys_and_apply(cm, keys, prefix='', accum=list(), apply_fn=None, kwargs={}):
    for name, m in named_children(cm):
        look_for_keys_and_apply(m, keys=keys, prefix=prefix+'.'+name, accum=accum, apply_fn=apply_fn, kwargs=kwargs)
        if any( [key in m._get_name() for key in keys]):
            if isinstance(apply_fn, str):   apply_fn = getattr(m, apply_fn, None)
            if apply_fn is not None:    accum[name] = apply_fn(**kwargs)
            


class Agent(object):
    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm
        
        self.handled_experiences = 0
        self.save_path = None
        self.episode_count = 0

        self.training = True
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']
        
        self.goal_oriented = self.algorithm.kwargs['goal_oriented'] if 'goal_oriented' in self.algorithm.kwargs else False
        self.goals = None 
        if 'goal_preprocess' in self.algorithm.kwargs:
            self.goal_preprocessing = self.algorithm.kwargs['goal_preprocess']
        elif self.goal_oriented:
            raise NotImplementedError

        self.nbr_actor = self.algorithm.get_nbr_actor()
        self.previously_done_actors = [False]*self.nbr_actor

        self.recurrent = False
        self.rnn_states = None
        self.rnn_keys, self.rnn_states = Agent._reset_rnn_states(self.algorithm, self.nbr_actor)
        if len(self.rnn_keys):
            self.recurrent = True

    def get_experience_count(self):
        raise NotImplementedError

    def get_update_count(self):
        raise NotImplementedError

    def set_nbr_actor(self, nbr_actor):
        if nbr_actor != self.nbr_actor:
            self.nbr_actor = nbr_actor
            self.reset_actors(init=True)
            self.algorithm.reset_storages(nbr_actor=self.nbr_actor)

    def reset_actors(self, indices=None, init=False):
        '''
        In case of a multi-actor process, this function is called to reset
        the actors' internal values.
        '''
        if indices is None: indices = range(self.nbr_actor)
        
        if init:
            self.previously_done_actors = [False]*self.nbr_actor
        else:
            for idx in indices: self.previously_done_actors[idx] = False

        if self.recurrent:
            _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)

    def update_actors(self, batch_idx):
        '''
        In case of a multi-actor process, this function is called to handle
        the (dynamic) number of environment that are being used.
        More specifically, it regularizes the rnn_states when
        an actor's episode ends.
        It is assumed that update can only go by removing stuffs...
        Indeed, actors all start at the same time, and thus self.reset_actors()
        ought to be called at that time.
        Note: since the number of environment that are running changes, 
        the size of the rnn_states on the batch dimension will change too.
        Therefore, we cannot identify an rnn_state by the actor/environment index.
        Thus, it is a batch index that is requested, that would truly be in touch
        with the current batch dimension.
        :param batch_idx: index of the actor whose episode just finished.
        '''
        if self.recurrent:
            self.remove_from_rnn_states(batch_idx=batch_idx)

        if self.goal_oriented:
            self.remove_from_goals(batch_idx=batch_idx)

    def update_goals(self, goals):
        assert(self.goal_oriented)
        self.goals = goals
        
    def remove_from_goals(self, batch_idx):
        self.goals = np.concatenate(
                    [self.goals[:batch_idx,...], 
                     self.goals[batch_idx+1:,...]],
                     axis=0)
                
    @staticmethod
    def _reset_rnn_states(algorithm, nbr_actor):
        # TODO: account for the indices in rnn states:
        lookedup_keys = ['LSTM', 'GRU']
        rnn_states = {}
        kwargs = {'cuda': algorithm.kwargs['use_cuda'], 'repeat':nbr_actor}
        look_for_keys_and_apply(algorithm.get_models()['model'], keys=lookedup_keys, accum=rnn_states, apply_fn='get_reset_states', kwargs=kwargs)
        rnn_keys = list(rnn_states.keys())
        return rnn_keys, rnn_states
        

    def remove_from_rnn_states(self, batch_idx):
        '''
        Remove a row(=batch) of data from the rnn_states.
        :param batch_idx: index on the batch dimension that specifies which row to remove.
        '''
        for recurrent_submodule_name in self.rnn_states:
            for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                self.rnn_states[recurrent_submodule_name]['hidden'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['hidden'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['hidden'][idx][batch_idx+1:,...]],
                     dim=0)
                self.rnn_states[recurrent_submodule_name]['cell'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['cell'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['cell'][idx][batch_idx+1:,...]],
                     dim=0)
        
    def _pre_process_rnn_states(self):
        if self.rnn_states is None: 
            _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)

        if self.algorithm.kwargs['use_cuda']:
            for recurrent_submodule_name in self.rnn_states:
                for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                    self.rnn_states[recurrent_submodule_name]['hidden'][idx] = self.rnn_states[recurrent_submodule_name]['hidden'][idx].cuda()
                    self.rnn_states[recurrent_submodule_name]['cell'][idx]   = self.rnn_states[recurrent_submodule_name]['cell'][idx].cuda()

    @staticmethod
    def _extract_from_rnn_states(rnn_states_batched: dict, batch_idx: int):
        rnn_states = {k: {'hidden':[], 'cell':[]} for k in rnn_states_batched}
        for recurrent_submodule_name in rnn_states_batched:
            for idx in range(len(rnn_states_batched[recurrent_submodule_name]['hidden'])):
                rnn_states[recurrent_submodule_name]['hidden'].append( rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_idx,...].unsqueeze(0))
                rnn_states[recurrent_submodule_name]['cell'].append( rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_idx,...].unsqueeze(0))
        return rnn_states

    def _post_process(self, prediction):
        if self.recurrent:
            for recurrent_submodule_name in self.rnn_states:
                for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                    self.rnn_states[recurrent_submodule_name]['hidden'][idx] = prediction['next_rnn_states'][recurrent_submodule_name]['hidden'][idx].cpu()
                    self.rnn_states[recurrent_submodule_name]['cell'][idx]   = prediction['next_rnn_states'][recurrent_submodule_name]['cell'][idx].cpu()

            for k, v in prediction.items():
                if isinstance(v, dict):
                    for vk in v:
                        hs, cs = v[vk]['hidden'], v[vk]['cell']
                        for idx in range(len(hs)):
                            hs[idx] = hs[idx].detach().cpu()
                            cs[idx] = cs[idx].detach().cpu()
                        prediction[k][vk] = {'hidden': hs, 'cell': cs}
                else:
                    prediction[k] = v.detach().cpu()
        else:
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}

        return prediction

    @staticmethod
    def _extract_from_prediction(prediction: dict, batch_idx: int):
        out_pred = dict()
        for k, v in prediction.items():
            if isinstance(v, dict):
                continue
            out_pred[k] = v[batch_idx,...].unsqueeze(0)
        return out_pred

    @staticmethod
    def _extract_from_hdict(hdict: dict, batch_idx: int, goal_preprocessing_fn=None):
        out_hdict = dict()
        for k, v in hdict.items():
            if isinstance(v, dict):
                v = Agent._extract_from_hdict(hdict=v, batch_idx=batch_idx, goal_preprocessing_fn=goal_preprocessing_fn)
            else:
                if isinstance(v, torch.Tensor):
                    v = v[batch_idx, ...].unsqueeze(0)
                else:
                    v = np.expand_dims(v[batch_idx, ...], axis=0)
                if goal_preprocessing_fn is not None:
                    v = goal_preprocessing_fn(v)
            out_hdict[k] = v 
        return out_hdict

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        non_terminal = torch.from_numpy(1 - np.array(done)).type(torch.FloatTensor)
        state = self.state_preprocessing(state, use_cuda=False)
        succ_state = self.state_preprocessing(succ_state, use_cuda=False)
        if isinstance(reward, np.ndarray): r = torch.from_numpy(reward).type(torch.FloatTensor)
        else: r = torch.ones(1).type(torch.FloatTensor)*reward
        return state, r, succ_state, non_terminal

    def handle_experience(self, s, a, r, succ_s, done, goal=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goal: numpy tensor of goal of shape batch x goal_shape.
        '''
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def clone(self, training=None):
        raise NotImplementedError

    def save(self):
        assert(self.save_path is not None)
        torch.save(self.clone(), self.save_path)