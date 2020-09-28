from typing import Generator, Callable
from copy import deepcopy
import pickle

import torch
import numpy as np
import gym
import minerl

from .action_discretisation import get_action_set, generate_action_parser


def trajectory_based_rl_loop(agent, minerl_trajectory_env: gym.Env,
                             action_parser: Callable):
    '''
    Feeds :param: agent a sequence of experiences coming from
    :param: minerl_trajectory_env.

    The agent is requested to take actions, but these are ignored by
    the underlying environment. This is because as all observations,
    rewards and actions are stored in the trajectory
    underlying :param: minerl_trajectory_env.

    :param agent: Ideally R2D2 agent. Off-policy agent.
    :param minerl_trajectory_env: Environment that's going to be fed to agent
    '''
    agent.reset_actors()
    
    obs = minerl_trajectory_env.reset()
    done = [False] * agent.nbr_actor
    while not all(done):
        # With r2d2, we explicitly need to have 'extra_inputs' in frame (rnn) state
        # TODO: we should add info['inventory']
        #agent.rnn_states['phi_body']['extra_inputs'] = {}

        # Taking an action is mandatory to propagate rnn_states, even if
        # action is ignored
        _ = agent.take_action(obs)

        succ_obs, reward, done, infos = minerl_trajectory_env.step(
            action_vector=[None] * agent.nbr_actor
        )

        import ipdb; ipdb.set_trace()

        agent.handle_experience(obs,
                                np.array([action_parser(info_i['a']) for info_i in infos]),
                                reward,
                                succ_obs,
                                done,
                                infos=infos)
        obs = succ_obs


class MineRLTrajectoryBasedEnv(gym.Env):

    def __init__(self, trajectory_generator: Generator,
                 action_parser: Callable = lambda x: x):
        '''
        :param trajectory_generator: Generator obtained by running
            >>> data_pipeline = minerl.data.make(MineRLEnvName)
            >>> traj_names = data_pipeline.get_trajectory_names()
            >>> data_generator = data_pipeline.load_data(traj_names[any_will_do])

        :param action_parser: Function that processes action. Default is identity.
                              Useful to transform continuous actions into discretized ones
        '''

        self.trajectory_generator = trajectory_generator
        self.action_parser = action_parser

        # Format: (observation, action, reward, succ_observation, done)
        self.current_experience: Tuple = None

        self.observation_space = minerl.herobraine.hero.spaces.Dict(
            pov=gym.spaces.Box(
                low=0, high=255, shape=(64, 64, 3)
            ),
            vector=gym.spaces.Box(
                low=-1.2, high=1.2, shape=(64,)
            )
        )

        self.action_space = minerl.herobraine.hero.spaces.Dict(
            vector=gym.spaces.Box(
                low=-1.049999, high=1.049999, shape=(64,)
            )
        )

        # Generator cannot be resetted, so we need to ensure that reset is only
        # called once.
        self.has_reset_been_called = False
        self.is_trajectory_done = False

    def reset(self):
        if self.has_reset_been_called:
            raise NotImplementedError('Resetting MineRLTrajectoryBasedEnv more than ONCE is not supported')
        self.has_reset_been_called = True

        (o, a, r, succ_o, d) = next(self.trajectory_generator)
        self.current_experience = (o, a, r, succ_o, d)
        return o

    def step(self, action):
        '''
        We ignore :param: action, because we are following a fixed trajectory
        info should contain the action in self.current_experience after being parsed!
        '''
        if self.is_trajectory_done:
            return None

        # Decide what we return
        (o, a, r, succ_o, d) = self.current_experience

        info = {'a': self.action_parser(a['vector']),
 'inventory_vector': o['vector']
               }

        # Update current experience
        self.is_trajectory_done = d
        if not self.is_trajectory_done:
            self.current_experience = next(self.trajectory_generator)

        return succ_o, r, d, info

    def render(self):
        raise NotImplementedError('Not supported')


if __name__ == "__main__":

    # Env: preliminaries
    data_pipeline = minerl.data.make('MineRLTreechopVectorObf-v0')
    traj_names = data_pipeline.get_trajectory_names()
    data_iterator = data_pipeline.load_data(traj_names[0])

    # Action set
    action_set = pickle.load(open('treechop_action_set.pickle', 'rb'))
    continuous_to_discrete_action_parser = generate_action_parser(action_set)

    # Making the env
    env = MineRLTrajectoryBasedEnv(data_iterator)

    # Agent
    agent = torch.load('./test_agent.pt')
    agent.algorithm.min_capacity = 10
    agent.kwargs['min_capacity'] = 10

    trajectory_based_rl_loop(agent, env,
                             action_parser=continuous_to_discrete_action_parser)
