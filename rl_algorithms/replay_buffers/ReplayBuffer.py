import numpy as np


class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = np.zeros(self.capacity, dtype=object)
        self.position = 0

    def push(self, experience):
        self.memory[self.position] = experience
        self.position = (self.position+1) % self.capacity
        self.position = int(self.position)

    def sample(self, batch_size):
        return np.random.choice(self.memory[:self.position], batch_size)

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
