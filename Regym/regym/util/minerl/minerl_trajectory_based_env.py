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
    nbr_actors = minerl_trajectory_env.get_nbr_envs()
    agent.reset_actors()

    obs = minerl_trajectory_env.reset()
    done = [False] * agent.nbr_actor
    previous_done = deepcopy(done)

    while not all(done):
        # Taking an action is mandatory to propagate rnn_states, even if
        # action is ignored
        dummy_action = agent.take_action(obs)

        succ_obs, reward, done, infos = minerl_trajectory_env.step(
            action_vector=dummy_action #[None] * agent.nbr_actor
        )

        action = np.array(
            [
                info_i['current_action'] # (1,) 
                for info_i in infos if info_i is not None
            ]
        )
        # (nbr_actor, 1)

        agent.handle_experience(obs,
                                action,
                                reward,
                                succ_obs,
                                done,
                                infos=infos)
        
        # Since we are not 'gathering' experiences,
        # we need to account for environments finishing
        # before the other ones:
        batch_index = -1
        batch_idx_done_actors_among_not_done = []
        for actor_index in range(nbr_actors):
            if previous_done[actor_index]:
                continue
            batch_index +=1

            # Bookkeeping of the actors whose episode just ended:
            d = done[actor_index]
            if ('real_done' in infos[actor_index]):
                d = infos[actor_index]['real_done']

            if d and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)

        obs = deepcopy(succ_obs)
        if len(batch_idx_done_actors_among_not_done):
            # Regularization of the agents' next observations:
            batch_idx_done_actors_among_not_done.sort(reverse=True)
            for batch_idx in batch_idx_done_actors_among_not_done:
                obs = np.concatenate( [obs[:batch_idx,...], obs[batch_idx+1:,...]], axis=0)

        previous_done = deepcopy(done)


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
        self.previous_action = np.array([0])
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

        info = {
            'current_action': self.action_parser(a['vector']), #(1,)
            'previous_action': self.previous_action, #(1,)
            'inventory': np.expand_dims(o['vector'], axis=0) # (1, 64)
        }

        # Update current experience
        self.is_trajectory_done = d
        if not self.is_trajectory_done:
            self.current_experience = next(self.trajectory_generator)
            self.previous_action = info['current_action']

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
