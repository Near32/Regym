from ..algorithm import Algorithm

class AlgorithmWrapper(Algorithm):
    def __init__(self, algorithm):
        self.algorithm = algorithm 
        self.kwargs = self.algorithm.kwargs

    def get_models(self):
        return self.algorithm.get_models()

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        return self.algorithm.get_epsilon(nbr_steps, strategy=strategy)

    def get_nbr_actor(self):
        return self.algorithm.get_nbr_actor()

    def get_update_count(self):
        return self.algorithm.get_update_count()

    def reset_storages(self):
        self.algorithm.reset_storages()

    def store(self, exp_dict, actor_index=0):
        self.algorithm.store(exp_dict=exp_dict, actor_index=actor_index)

    def train(self, minibatch_size=None):
        self.algorithm.train(minibatch_size=minibatch_size)

    def clone(self):
        return self.algorithm.clone()