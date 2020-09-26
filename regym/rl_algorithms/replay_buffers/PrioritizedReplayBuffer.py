import numpy as np
from .experience import EXP
from .ReplayBuffer import ReplayStorage
import regym


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.2, beta=1.0) :
        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.sumPi_alpha = 0.0

    def save(self, path):
        path += '.prb'
        np.savez(path, tree=self.tree, data=self.data,
                 length=np.asarray(self.length), sumPi=np.asarray(self.sumPi_alpha),
                 counter=np.asarray(self.counter), alpha=np.asarray(self.alpha))

    def load(self, path):
        path += '.prb.npz'
        data = np.load(path)
        self.tree = data['tree']
        self.data = data['data']
        self.counter = int(data['counter'])
        self.length = int(data['length'])
        self.sumPi_alpha = float(data['sumPi'])
        self.alpha = float(data['alpha'])

    def reset(self):
        self.__init__(capacity=self.capacity, alpha=self.alpha)

    def add(self, exp, priority):
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        idx = self.counter + self.capacity -1

        self.data[self.counter] = exp

        self.counter += 1
        self.length = min(self.length+1, self.capacity)
        if self.counter >= self.capacity :
            self.counter = 0

        self.sumPi_alpha += priority
        self.update(idx,priority)

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha

    def update(self, idx, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        change = priority - self.tree[idx]

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority

        self._propagate(idx,change)

    def _propagate(self, idx, change) :
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0 :
            self._propagate(parentidx, change)

    def __call__(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        data = self.data[dataidx]
        priority = self.tree[idx]

        return (idx, priority, data)

    def get(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1

        data = self.data[dataidx]
        if not isinstance(data,EXP) :
            raise TypeError

        priority = self.tree[idx]

        return (idx, priority, *data)

    def get_importance_sampling_weight(priority,beta=1.0) :
        return pow( self.capacity * priority , -beta )

    def get_buffer(self) :
        return [ self.data[i] for i in range(self.capacity) if isinstance(self.data[i],EXP) ]


    def _retrieve(self,idx,s) :
         leftidx = 2*idx+1
         rightidx = leftidx+1

         if leftidx >= len(self.tree) :
            return idx

         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])

    def total(self) :
        return self.tree[0]

    def __len__(self) :
        return self.length

    def sample(self, batch_size):
        prioritysum = self.total()
        # Random Experience Sampling with priority
        low = 0.0
        step = (prioritysum-low) / batch_size
        randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size))

        transitions = list()
        priorities = []
        for i in range(batch_size):
            '''
            Sampling from this replayBuffer requires it to be fully populated.
            Otherwise, we might end up trying to sample a leaf ot the binary sumtree
            that does not contain any data, thus throwing a TypeError.
            '''
            try :
                el = self.get(randexp[i])
                priorities.append( el[1] )
                transitions.append(el)
            except TypeError as e :
                continue

        # Importance Sampling Weighting:
        priorities = np.array(priorities, dtype=np.float32)
        importanceSamplingWeights = np.power( len(self) * priorities , -self.beta)

        return transitions, importanceSamplingWeights


class PrioritizedReplayStorage_:
    def __init__(self, capacity, alpha=0.2, beta=1.0, keys=None) :
        if keys is None:    keys = []
        keys = keys + ['s', 'a', 'r', 'succ_s', 'non_terminal',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'qa', 'log_pi_a',
                       'mean', 'action_logits', 'rnn_state']
        self.keys = keys
        self.reset()

        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-2
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)

        self.position = dict()
        self.current_size = dict()

        self.sumPi_alpha = 0.0
        self.max_priority = np.ones(1, dtype=np.float32)

    def add_key(self, key):
        self.keys += [key]
        setattr(self, key, np.zeros(self.capacity+1, dtype=object))
        self.position[key] = 0
        self.current_size[key] = 0

    def reset(self):
        for k in self.keys:
            setattr(self, k, np.zeros(self.capacity+1, dtype=object))
            self.position[k] = 0
            self.current_size[k] = 0

    def total(self) :
        return self.tree[0]

    def cat(self, keys):
        data = [getattr(self, k)[:self.current_size[k]] for k in keys]
        return data

    def get_data_idx(self, idx, keys=None):
        if keys is None:    keys = self.keys
        data = []
        for k in keys:
            v = [ vi for vi in getattr(self, k) if vi is not None]
            data.append(v)
        data = [[v[idx]] for v in data]
        return data

    def __len__(self) :
        return self.length

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha

    def update(self, idx, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        change = priority - self.tree[idx]

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority

        self.max_priority = max(priority, self.max_priority)

        self._propagate(idx,change)

    def _propagate(self, idx, change) :
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0 :
            self._propagate(parentidx, change)

    def add(self, exp, priority=None):
        if priority is None:
            priority = self.max_priority

        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        idx = self.position['s'] + self.capacity -1

        #self.data[self.counter] = exp
        for k, v in exp.items():
            #getattr(self, key)[self.counter] = value
            assert k in self.keys, f"Tried to add value key({k}, {v}), but {k} is not registered."
            getattr(self, k)[self.position[k]] = v
            self.position[k] = int((self.position[k]+1) % self.capacity)
            self.current_size[k] = min(self.capacity, self.current_size[k]+1)

        #self.counter += 1
        self.length = min(self.length+1, self.capacity)
        #if self.counter >= self.capacity :
        #    self.counter = 0

        self.sumPi_alpha += priority
        self.update(idx,priority)

    def _retrieve(self,idx,s) :
         leftidx = 2*idx+1
         rightidx = leftidx+1

         if leftidx >= len(self.tree) :
            return idx

         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])

    def __call__(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        #data = self.data[dataidx]
        data = self.get_data_idx(dataidx)
        priority = self.tree[idx]

        return (idx, priority, data)

    def get(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1

        '''
        data = self.data[dataidx]
        if not isinstance(data,EXP) :
            raise TypeError
        '''
        data = self.get_data_idx(dataidx)

        priority = self.tree[idx]

        return (idx, priority, *data)

    def get_importance_sampling_weight(priority,beta=1.0) :
        return pow( self.capacity * priority , -beta )

    def sample(self, batch_size, keys=None):
        #return np.random.choice(self.memory[:self.current_size], batch_size)
        if keys is None:    keys = self.keys
        min_current_size = self.capacity
        for idx_key in reversed(range(len(keys))):
            key = keys[idx_key]
            if self.current_size[key] == 0:
                continue
            if self.current_size[key] < min_current_size:
                min_current_size = self.current_size[key]

        prioritysum = self.total()
        # Random Experience Sampling with priority
        low = 0.0
        step = (prioritysum-low) / batch_size
        randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size))

        tree_indices = [self._retrieve(0,rexp) for rexp in randexp]
        data_indices = [tidx-self.capacity+1 for tidx in tree_indices]

        priorities = [self.tree[tidx] for tidx in tree_indices]


        indices = np.random.choice(np.arange(min_current_size), batch_size)
        data = [getattr(self, k)[indices] for k in keys]
        return data


class PrioritizedReplayStorage(ReplayStorage):
    def __init__(self,
                 capacity,
                 alpha=0.2,
                 beta=1.0,
                 beta_increase_interval=1e4,
                 eta=0.9,
                 keys=None,
                 circular_keys={'succ_s':'s'},
                 circular_offsets={'succ_s':1}):
        super(PrioritizedReplayStorage, self).__init__(
            capacity=capacity,
            keys=keys,
            circular_keys=circular_keys,
            circular_offsets=circular_offsets
        )

        self.length = regym.RegymManager.Value(int, 0)
        self.alpha = alpha
        self.beta_start = beta
        self.beta_increase_interval = beta_increase_interval
        #assert self.beta_increase_interval <= self.capacity
        self.beta = self.beta_start

        self.eta = eta
        self.epsilon = 1e-4

        """
        self.tree = np.zeros(2 * int(self.capacity) - 1)
        self.max_priority = np.ones(1, dtype=np.float32)
        """
        self.tree = regym.RegymManager.list([ 0 for _ in range(2 * int(self.capacity) - 1)])
        self.max_priority = regym.RegymManager.Value(float, 1.0)
        self.sumPi_alpha = 0.0
        
    def _update_beta(self, iteration=None):
        if iteration is None:   iteration = self.length.value
        self.beta = min(1.0, self.beta_start+iteration*(1.0-self.beta_start)/self.beta_increase_interval)

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.length.value

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha

    def sequence_priority(self, sequence_errors) :
        '''
        :param sequence_errors: torch.Tensor of shape (unroll_dim,)
        '''
        max_error = sequence_errors.max()
        mean_error = sequence_errors.mean()
        return self.eta*max_error+(1-self.eta)*mean_error+self.epsilon

    def update(self, idx, priority):
        if np.isnan(priority).any() or np.isinf(priority).any():
            priority = self.max_priority.value

        change = priority - self.tree[idx]

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority

        self.max_priority.value = max(priority, self.max_priority.value)

        self._propagate(idx,change)

    def _propagate(self, idx, change):
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0 :
            self._propagate(parentidx, change)

    def add(self, exp, priority):
        if priority is None:
            priority = self.max_priority.value

        super(PrioritizedReplayStorage, self).add(data=exp)
        self.length.value = min(self.length.value+1, self.capacity)

        if np.isnan(priority).any() or np.isinf(priority).any() :
            priority = self.max_priority.value

        self.sumPi_alpha += priority

        idx = self.position['s'] + self.capacity -1
        self.update(idx,priority)

        self._update_beta()

    def _retrieve(self,idx,s):
         leftidx = 2*idx+1
         rightidx = leftidx+1

         if leftidx >= len(self.tree):
            return idx

         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])

    def sample(self, batch_size, keys=None):
        if keys is None:    keys = self.keys + self.circular_keys.keys()

        # Random Experience Sampling with priority
        prioritysum = self.total()
        low = 0.0
        step = (prioritysum-low) / batch_size
        randexp = np.arange(low,prioritysum,step)[:batch_size,...]+np.random.uniform(low=0.0,high=step,size=(batch_size))

        self.tree_indices = [self._retrieve(0,rexp) for rexp in randexp]
        priorities = [self.tree[tidx] for tidx in self.tree_indices]

        #Check that priorities are valid:
        valid = False
        while not valid:
            valid = True
            for idx in range(len(priorities)):
                if priorities[idx] == 0:
                    valid = False
                    newrandexp = randexp[idx]-np.random.uniform(low=0.0, high=step, size=(1))
                    self.tree_indices[idx] = self._retrieve(0,newrandexp)
                    priorities[idx] = self.tree[self.tree_indices[idx]]
                    break

        # Importance Sampling Weighting:
        priorities = np.array(priorities, dtype=np.float32)
        self.importanceSamplingWeights = np.power( len(self) * priorities , -self.beta)

        data_indices = np.array([tidx-self.capacity+1 for tidx in self.tree_indices])
        data = self.cat(keys=keys, indices=data_indices)

        return data, self.importanceSamplingWeights



class SplitPrioritizedReplayStorage(PrioritizedReplayStorage):
    def __init__(self,
                 capacity,
                 alpha=0.2,
                 beta=1.0,
                 beta_increase_interval=1e4,
                 keys=None,
                 circular_keys={'succ_s':'s'},
                 circular_offsets={'succ_s':1},
                 test_train_split_interval=10,
                 test_capacity=None):
        if test_capacity is None: test_capacity=capacity
        self.test_capacity = test_capacity
        self.test_train_split_interval = test_train_split_interval
        self.data_count = 0
        self.test_storage = PrioritizedReplayStorage(capacity=self.test_capacity,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     beta_increase_interval=80,
                                                     keys=keys,
                                                     circular_keys=circular_keys,
                                                     circular_offsets=circular_offsets)
        super(SplitPrioritizedReplayStorage, self).__init__(capacity=capacity,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       beta_increase_interval=beta_increase_interval,
                                                       keys=keys,
                                                       circular_keys=circular_keys,
                                                       circular_offsets=circular_offsets)

    def total(self, test=False):
        if test:
            return self.storage.total()
        else:
            return self.tree[0]

    def get_test_storage(self):
        return self.test_storage

    def get_size(self, test=False):
        if test:
            return len(self.test_storage)
        else:
            return self.current_size['s']

    def update(self, idx, priority, test=False):
        if test:
            self.test_storage.update(idx=idx, priority=priority)
        else:
            super(SplitPrioritizedReplayStorage, self).update(idx=idx, priority=priority)
            '''
            if np.isnan(priority) or np.isinf(priority) :
                priority = self.max_priority

            change = priority - self.tree[idx]

            previous_priority = self.tree[idx]
            self.sumPi_alpha -= previous_priority

            self.sumPi_alpha += priority
            self.tree[idx] = priority

            self.max_priority = max(priority, self.max_priority)

            self._propagate(idx,change)
            '''

    def reset(self):
        self.test_storage.reset()
        super(SplitPrioritizedReplayStorage, self).reset()

    def add(self, data, priority):
        self.data_count += 1
        if self.data_count % self.test_train_split_interval == 0:
            self.test_storage.add(exp=data, priority=priority)
        else:
            super(SplitPrioritizedReplayStorage, self).add(exp=data, priority=priority)
            '''
            if priority is None:
                priority = self.max_priority

            super(PrioritizedReplayStorage, self).add(data=exp)
            self.length = min(self.length+1, self.capacity)

            if np.isnan(priority) or np.isinf(priority) :
                priority = self.max_priority
            self.sumPi_alpha += priority

            idx = self.position['s'] + self.capacity -1
            self.update(idx,priority)

            self._update_beta()
            '''

    def sample(self, batch_size, keys=None, test=False):
        if test:
            return self.test_storage.sample(batch_size=batch_size, keys=keys)
        else:
            return super(SplitPrioritizedReplayStorage, self).sample(batch_size=batch_size, keys=keys)
        '''
        if keys is None:    keys = self.keys + self.circular_keys.keys()

        # Random Experience Sampling with priority
        prioritysum = self.total()
        low = 0.0
        step = (prioritysum-low) / batch_size
        randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size))

        self.tree_indices = [self._retrieve(0,rexp) for rexp in randexp]
        priorities = [self.tree[tidx] for tidx in self.tree_indices]

        #Check that priorities are valid:
        valid = False
        while not valid:
            valid = True
            for idx in range(len(priorities)):
                if priorities[idx] == 0:
                    valid = False
                    newrandexp = randexp[idx]-np.random.uniform(low=0.0, high=step, size=(1))
                    self.tree_indices[idx] = self._retrieve(0,newrandexp)
                    priorities[idx] = self.tree[self.tree_indices[idx]]
                    break

        # Importance Sampling Weighting:
        priorities = np.array(priorities, dtype=np.float32)
        self.importanceSamplingWeights = np.power( len(self) * priorities , -self.beta)

        data_indices = np.array([tidx-self.capacity+1 for tidx in self.tree_indices])
        data = self.cat(keys=keys, indices=data_indices)

        return data, self.importanceSamplingWeights
        '''


class FasterPrioritizedReplayStorage(PrioritizedReplayStorage):
    def __init__(self, capacity, alpha=0.2, beta=1.0, keys=None, circular_keys={'succ_s':'s'}):
        super(FasterPrioritizedReplayStorage, self).__init__(capacity=capacity,
                                                             alpha=alpha,
                                                             beta=beta,
                                                             keys=keys,
                                                             circular_keys=circular_keys)
    # Attempted to accelerate the sampling function: not viable, it is already fast enough,
    # it might be better to tackle the update/propagate function...
    # Indeed, upon testing, it is the update functions that have the biggest time complexity...
    def _retrieve(self, idx, s, twoleaves=False):
        leftidx = 2*idx+1
        rightidx = leftidx+1

        if leftidx >= len(self.tree):
            return idx

        if s <= self.tree[leftidx] :
            retidx = self._retrieve(leftidx, s)
        else :
            retidx = self._retrieve(rightidx, s-self.tree[leftidx])

        if twoleaves:
            return retidx, retidx+1

        return retidx

    def sample(self, batch_size, keys=None):
        if keys is None:    keys = self.keys + self.circular_keys.keys()

        # Random Experience Sampling with priority
        prioritysum = self.total()
        low = 0.0
        #step = (prioritysum-low) / (batch_size//2)
        #randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size))
        step = (prioritysum-low) / (batch_size//2)
        randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size//2))

        #self.tree_indices = [self._retrieve(0,rexp) for rexp in randexp]
        self.tree_indices = []
        for r in randexp:
            i1, i2 = self._retrieve(0, r, twoleaves=True)
            self.tree_indices.append(i1)
            self.tree_indices.append(i2)

        priorities = [self.tree[tidx] for tidx in self.tree_indices]

        #Check that priorities are valid:
        valid = False
        while not valid:
            valid = True
            for idx in range(len(priorities)):
                if priorities[idx] == 0:
                    valid = False
                    newrandexp = randexp[idx//2]-np.random.uniform(low=0.0, high=step, size=(1))
                    self.tree_indices[idx] = self._retrieve(0,newrandexp)
                    priorities[idx] = self.tree[self.tree_indices[idx]]
                    break

        # Importance Sampling Weighting:
        priorities = np.array(priorities, dtype=np.float32)
        self.importanceSamplingWeights = np.power( len(self) * priorities , -self.beta)

        data_indices = np.array([tidx-self.capacity+1 for tidx in self.tree_indices])
        data = self.cat(keys=keys, indices=data_indices)

        return data, self.importanceSamplingWeights
