import torch

from ..replay_buffers import EXP
from ..networks import LeakyReLU, ActorNN, CriticNN
from ..DDPG import DeepDeterministicPolicyGradientAlgorithm


class DDPGAgent():
    def __init__(self, algorithm, name):
        """
        :param algorithm: algorithm class to use to optimize the network(s).
        """
        self.algorithm = algorithm
        self.training = True
        self.preprocess_function = self.algorithm.kwargs["preprocess"]

        self.kwargs = algorithm.kwargs

        self.nbr_steps = 0

        self.name = name

    def getModel(self):
        return [self.algorithm.model_actor, self.algorithm.model_critic]

    def handle_experience(self, s, a, r, succ_s, done=False):
        hs = self.preprocess_function(s)
        hsucc = self.preprocess_function(succ_s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)
        experience = EXP(hs, a, hsucc, r, done)
        self.algorithm.handle_experience(experience=experience)

        if self.training:
            self.algorithm.train(iteration=self.kwargs['nbrTrainIteration'])

    def take_action(self, state):
        return self.act(state=self.preprocess_function(state), exploitation=not(self.training), exploration_noise=None)

    def reset_eps(self):
        pass

    def act(self, state, exploitation=True, exploration_noise=None):
        if self.algorithm.use_cuda:
            state = state.cuda()
        action = self.algorithm.model_actor(state).detach().squeeze(0)

        if exploitation:
            return action.cpu().data.numpy()
        else:
            # exploration:
            if exploration_noise is not None:
                self.algorithm.noise.setSigma(exploration_noise)
            new_action = action.cpu().data.numpy() + self.algorithm.noise.sample()*self.algorithm.model_actor.action_scaler
            return new_action

    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned


class PreprocessFunction(object):
    def __init__(self, hash_function=None, use_cuda=False):
        self.hash_function = hash_function
        self.use_cuda = use_cuda

    def __call__(self, x):
        if self.hash_function is not None:
            x = self.hash_function(x)
        if self.use_cuda:
            return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
        else:
            return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def build_DDPG_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config:
        - 'nbrTrainIteration': int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
        - 'use_cuda': boolean to specify whether to use CUDA.
        - 'tau': float in (0, 1], float, soft-update rate [sensible: tau=1e-3]
        - 'gamma': Float in (0, 1] [sensible: 0.99]
        - 'use_cuda': Boolean, whether to use CUDA during model computations
        - 'action_scaler': float, [sensible: 1.0] TODO Find out what this is
        - 'use_HER': Boolean, whether or not to use Hindsight Experience Replay
        - 'HER_k': 2
        - 'HER_strategy': 'future'
        - 'HER_use_singlegoal': False
        - 'use_PER': True
        - 'PER_alpha': 0.7
        - 'min_capacity': Minimum numbers experiences that the experience replay will hold before training commences [sensible: 25e3]
        - 'memory_capacity': int, maximum capacity for the experience replay [sensible: 25.e03]
        - 'batch_size': int, batch size to use [sensible: batch_size=256].
        - 'learning_rate': float in (0, 1]. Learning rate used by the underlying Optimizer. [sensible: 3.0e-4]

    :param kwargs: THIS WILL BE FED TO ALGORITHM
        "path": str specifying where to save the model(s).
        "use_cuda": boolean to specify whether to use CUDA.
        "memory_capacity": int, capacity of the replay buffer to use.
        "min_capacity": int, minimal capacity before starting to learn.
        "batch_size": int, batch size to use [default: batch_size=256].
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "lr": float, learning rate [default: lr=1e-3].
        "tau": float, target update rate [default: tau=1e-3].
        "gamma": float, Q-learning gamma rate [default: gamma=0.999].
        "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
        "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
        "actfn": activation function to use in between each layer of the neural networks.
    '''

    kwargs = dict()

    preprocess = PreprocessFunction(use_cuda=config['use_cuda'])

    HER = {'k': config['HER_k'], 'strategy': config['HER_strategy'],
           'use_her': config['use_HER'], 'singlegoal': config['HER_use_singlegoal']}
    kwargs['HER'] = HER

    kwargs['nbrTrainIteration'] = config['nbrTrainIteration']
    kwargs["action_dim"] = task.action_dim
    kwargs["state_dim"] = task.observation_dim
    kwargs["action_scaler"] = config['action_scaler']

    kwargs["actfn"] = LeakyReLU

    # Create model architecture:
    actor = ActorNN(state_dim=task.observation_dim, action_dim=task.action_dim, action_scaler=config['action_scaler'], HER=HER['use_her'], use_cuda=config['use_cuda'])
    actor.share_memory()
    critic = CriticNN(state_dim=task.observation_dim, action_dim=task.action_dim, HER=config['use_HER'], use_cuda=config['use_cuda'])
    critic.share_memory()

    model_path = './' + agent_name
    path = model_path

    kwargs["path"] = path
    kwargs["use_cuda"] = config['use_cuda']

    kwargs["memory_capacity"] = config['memory_capacity']
    kwargs["min_capacity"]    = config['min_capacity']
    kwargs["batch_size"]      = config['batch_size']
    kwargs["use_PER"]         = config['use_PER']
    kwargs["PER_alpha"]       = config['PER_alpha']

    kwargs["lr"] = config['learning_rate']
    kwargs["tau"] = config['tau']
    kwargs["gamma"] = config['gamma']

    kwargs["preprocess"] = preprocess

    kwargs['replayBuffer'] = None

    DeepDeterministicPolicyGradient_algo = DeepDeterministicPolicyGradientAlgorithm(kwargs=kwargs, models={"actor": actor, "critic": critic})

    return DDPGAgent(algorithm=DeepDeterministicPolicyGradient_algo, name=agent_name)
