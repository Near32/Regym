import numpy as np


class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = np.zeros(self.capacity, dtype=object)
        self.position = 0
        self.current_size = 0

    def push(self, experience):
        self.memory[self.position] = experience
        self.position = int((self.position+1) % self.capacity)
        self.current_size = min(self.capacity, self.current_size + 1)

    def sample(self, batch_size):
        return np.random.choice(self.memory[:self.current_size], batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        path += '.rb'
        np.savez(path, memory=self.memory, position=np.asarray(self.position))

    def load(self, path):
        path += '.rb.npz'
        data = np.load(path)
        self.memory = data['memory']
        self.position = int(data['position'])



class ReplayStorage():
    def __init__(self, capacity, keys=None, circular_keys={'succ_s':'s'}, circular_offsets={'succ_s':1}):
        '''
        Use a different circular offset['succ_s']=n to implement truncated n-step return...
        '''
        if keys is None:    keys = ['s', 'a', 'r', 'non_terminal', 'rnn_state']
        # keys = keys + ['s', 'a', 'r', 'succ_s', 'non_terminal',
        #                'v', 'q', 'pi', 'log_pi', 'ent',
        #                'adv', 'ret', 'qa', 'log_pi_a',
        #                'mean', 'action_logits', 'rnn_state']
        self.keys = keys
        self.circular_keys = circular_keys
        self.circular_offsets = circular_offsets
        self.capacity = capacity
        self.position = dict()
        self.current_size = dict()
        self.reset()

    def add_key(self, key):
        self.keys += [key]
        setattr(self, key, np.zeros(self.capacity+1, dtype=object))
        self.position[key] = 0
        self.current_size[key] = 0

    def add(self, data):
        for k, v in data.items():
            #assert k in self.keys or k in self.circular_keys, f"Tried to add value key({k}, {v}), but {k} is not registered."
            if not(k in self.keys or k in self.circular_keys):  continue
            if k in self.circular_keys: continue
            getattr(self, k)[self.position[k]] = v
            self.position[k] = int((self.position[k]+1) % self.capacity)
            self.current_size[k] = min(self.capacity, self.current_size[k]+1)

    def reset(self):
        for k in self.keys:
            if k in self.circular_keys: continue
            setattr(self, k, np.zeros(self.capacity+1, dtype=object))
            self.position[k] = 0
            self.current_size[k] = 0

    def cat(self, keys, indices=None):
        data = []
        for k in keys:
            assert k in self.keys or k in self.circular_keys, f"Tried to get value from key {k}, but {k} is not registered."
            indices_ = indices
            cidx=0
            if k in self.circular_keys: 
                cidx=self.circular_offsets[k]
                k = self.circular_keys[k]
            v = getattr(self, k)
            if indices_ is None: indices_ = np.arange(self.current_size[k]-1-cidx)
            else:
                # Check that all indices are in range:
                for idx in range(len(indices_)):
                    if self.current_size[k]>0 and indices_[idx]>=self.current_size[k]-1-cidx:
                        indices_[idx] = np.random.randint(self.current_size[k]-1-cidx)
                        # propagate to argument:
                        indices[idx] = indices_[idx]
            '''
            '''
            indices_ = cidx+indices_
            values = v[indices_]
            data.append(values)
        return data 

    def __len__(self):
        return len(self.s)

    def sample(self, batch_size, keys=None):
        if keys is None:    keys = self.keys + self.circular_keys.keys()
        min_current_size = self.capacity
        for idx_key in reversed(range(len(keys))):
            key = keys[idx_key]
            if key in self.circular_keys:   key = self.circular_keys[key]
            if self.current_size[key] == 0:
                continue
            if self.current_size[key] < min_current_size:
                min_current_size = self.current_size[key]

        indices = np.random.choice(np.arange(min_current_size-1), batch_size)
        data = self.cat(keys=keys, indices=indices)
        return data
