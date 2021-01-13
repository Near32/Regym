# Adapted from:
# https://github.com/PettingZoo-Team/PettingZoo/blob/01ca3e333ef31dca638f173735a6f68e7bd39524/pettingzoo/classic/hanabi/hanabi.py

from typing import Optional, Dict, List, Union
import numpy as np
import gym
from gym.utils import EzPickle

try:
    from hanabi_learning_environment.rl_env import HanabiEnv, make

except Exception as e:
    raise ImportError(
        (
            "Hanabi may not be installed.\n",
            "Run ´./hanabi_install.sh´ script from regym.environment.envs.hanabi."
        )
    )

class TinyAbstractHanabiGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # set of all required params
    required_keys: set = {
        'payout_matrix',
        'num_cards',
        'num_actions',
        'players',
        'random_start_player',
    }

    def __init__(
        self,
        payout_matrix: np.ndarray,
        num_cards: int = 2,
        num_actions: int = 3,
        players: int = 2,
        random_start_player: bool = False,
        seed: int = 0,
        observation_type: int = 0,
         ):

        """
        Parameter descriptions :
              - num_cards: int, Number of cards in [2,5].
              - num_actions: int, Number of actions in [3,5].
              - players: int, Number of players in [2,5].
              - random_start_player: bool, Random start player.
              - observation_type: int, type of observation given to each player.
                - 0: one-hot-encoded card, concatenated with one-hot-encoded previous player's action.
                - 1: one-hot-encoded car+previous_player_action all at once.
        """

        gym.Env.__init__(self)

        self._config = {
            'payout_matrix': payout_matrix,
            'num_cards': num_cards,
            'num_actions': num_actions,
            'players': players,
            'random_start_player': random_start_player,
            'seed': seed,
            'observation_type': observation_type,
        }
        
        self.agents = list(range(players))
        self.current_agent: int

        if self._config['observation_type'] == 0:
            agent_action_space = gym.spaces.Discrete(num_actions)
            self.action_space = agent_action_space
            
            self.observation_space_size = num_cards+num_actions
            agent_observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.observation_space_size,),
                dtype=np.float32
            )
            action_mask_observation_space = gym.spaces.Box(
                low=0, 
                high=1, 
                shape=(num_actions,), 
                dtype=np.int8
            )
        elif self._config['observation_type'] == 1:
            # We add a No-op action that is at idx=num_actions
            agent_action_space = gym.spaces.Discrete(num_actions+1)
            self.action_space = agent_action_space
            
            self.observation_space_size = num_cards*(num_actions+1)
            agent_observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.observation_space_size,),
                dtype=np.float32
            )
            action_mask_observation_space = gym.spaces.Box(
                low=0, 
                high=1, 
                shape=(num_actions+1,), 
                dtype=np.int8
            )
        else:
            raise NotImplementedError


        self.observation_space = agent_observation_space

        self.reset()

    def seed(self, seed=None):
        self._config['seed'] = seed
        np.random.seed(seed)

    def reset(self):
        self.player_actions = []

        all_cards = np.arange(self._config["num_cards"])

        self.player_cards = []
        self.player_cards_ohe = []
        for player in range(self._config["players"]):
            card_idx = np.random.choice(all_cards, size=1) 
            self.player_cards.append(card_idx)
            card_ohe = np.zeros((1,self._config["num_cards"]))
            card_ohe[0, card_idx] = 1
            self.player_cards_ohe.append(card_ohe)

        #self.current_agent = 0 
        self.current_agent = np.random.choice(list(range(self._config["players"])))

        # Only used with observation_type==1:
        self.previous_player_actions = [self._config["num_actions"] for _ in range(self._config["players"])]    #No-op for all...
        # Only used with observation_type==0:
        self.previous_player_actions_ohe = [np.zeros((1,self._config["num_actions"])) for _ in range(self._config["players"])]

        self.rewards = [0 for _ in self.agents]
        
        self.done = False 

        self._encode_observations()

        return self.observations, self.infos

    def _step_agents(self):
        self.current_agent = (self.current_agent+1)%len(self.agents)

    def step(self, actions: List[int]) -> Optional[Union[np.ndarray, List[List[dict]]]]:
        """ 
        Advances the environment by one step. 
        Actions must be within self.legal_moves, otherwise throws error.
        Returns:
            observation: Optional List of new observations of agent at turn after the action step is performed.
            By default a list of integers, describing the logic state of the game from the view of the agent.
        """
        assert not self.done

        agent_on_turn = self.current_agent
        
        legal_moves = self.legal_actions_as_int[agent_on_turn]
        action = actions[agent_on_turn].item()

        if action not in legal_moves:
            raise ValueError('Illegal action. Please choose between legal actions, as documented in dict self.infos')

        self.player_actions.append(action)

        # Iterate current_agent pointer: 
        self._step_agents()

        # Apply action
        next_agent = self.current_agent
        self.previous_player_actions[next_agent] = action
        self.previous_player_actions_ohe[next_agent] = np.zeros((1,self._config["num_actions"]))
        self.previous_player_actions_ohe[next_agent][0,action] = 1

        reward = 0
        self.done = False
        if len(self.player_actions)==2:
            # With player 0 starting:
            """
            reward = self._config["payout_matrix"][
                self.player_cards[0], 
                self.player_cards[1],
                self.player_actions[0],
                self.player_actions[1] 
            ]
            """
            # Generalisation to random player start:
            agent_on_previous_turn = (agent_on_turn-1)%len(self.agents) 
            reward = self._config["payout_matrix"][
                self.player_cards[agent_on_previous_turn], 
                self.player_cards[agent_on_turn],
                self.player_actions[0],
                self.player_actions[1] 
            ]
            self.done = True

        self._encode_observations()
        
        self.rewards = [reward for _ in self.agents]

        return self.observations, self.rewards, self.done, self.infos

    def _encode_observations(self):
        
        if self._config['observation_type'] == 0:
            self.observations = [
                np.concatenate([self.player_cards_ohe[player_idx], self.previous_player_actions_ohe[player_idx]],axis=-1)
                for player_idx in self.agents
            ]
        elif self._config['observation_type'] == 1:
            self.observations = []
            for player_idx in self.agents:
                obs_idx = self.player_cards[player_idx]*(self._config["num_actions"]+1)+self.previous_player_actions[player_idx]
                obs_ohe = np.zeros((1,self.observation_space_size))
                obs_ohe[0, obs_idx] = 1
                self.observations.append(obs_ohe)
        else:
            raise NotImplementedError
        
        self.legal_actions_as_int = []
        self.infos = []
        if self._config["observation_type"]==0:
            for player_idx in self.agents:
                legal_moves= []
                if player_idx==self.current_agent:
                    legal_moves = list(range(self._config["num_actions"]))
                self.legal_actions_as_int.append(legal_moves)
                
                action_mask=np.zeros((1,self._config["num_actions"]))
                np.put(action_mask, ind=legal_moves, v=1)
                
                info = {}
                info['action_mask'] = action_mask
                info['legal_actions'] = action_mask
                self.infos.append(info)
        elif self._config["observation_type"]==1:
            for player_idx in self.agents:
                # Only No-op:
                legal_moves= [self._config["num_actions"]]
                if player_idx==self.current_agent:
                    # Everything, except No-op:
                    legal_moves = list(range(self._config["num_actions"]))
                self.legal_actions_as_int.append(legal_moves)
                
                action_mask=np.zeros((1,self._config["num_actions"]+1))
                np.put(action_mask, ind=legal_moves, v=1)
                
                info = {}
                info['action_mask'] = action_mask
                info['legal_actions'] = action_mask
                self.infos.append(info)
        else:
            raise NotImplementedError

    def render(self, mode='human'):
        """ 
        Supports console print only. Prints player's data.
        """
        pass


class TinyAbstractHanabi2P2C3AGymEnv(TinyAbstractHanabiGymEnv,EzPickle):
    def __init__(self, seed: int = 0,):
        """
        """
        EzPickle.__init__(**locals())
        
        payout_matrix = [              
            [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
            [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
            [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
        ]

        payout_matrix = np.array(payout_matrix)

        TinyAbstractHanabiGymEnv.__init__(
            self=self,
            payout_matrix = payout_matrix,
            num_cards = 2,
            num_actions = 3,
            players = 2,
            random_start_player= False,
            seed=seed,
            observation_type=0,
        )


class TinyAbstractHanabiOHEObs2P2C3AGymEnv(TinyAbstractHanabiGymEnv,EzPickle):
    def __init__(self, seed: int = 0,):
        """
        """
        EzPickle.__init__(**locals())
        
        payout_matrix = [              
            [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
            [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
            [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
        ]

        payout_matrix = np.array(payout_matrix)

        TinyAbstractHanabiGymEnv.__init__(
            self=self,
            payout_matrix = payout_matrix,
            num_cards = 2,
            num_actions = 3,
            players = 2,
            random_start_player= False,
            seed=seed,
            observation_type=1,
        )

"""
 TODO:
    1. think about a penalty wrapper for illegal moves, maybe?
"""

class HanabiGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # set of all required params
    required_keys: set = {
        'colors',
        'ranks',
        'players',
        'hand_size',
        'max_information_tokens',
        'max_life_tokens',
        'observation_type',
        'random_start_player',
    }

    def __init__(self,
                 colors: int = 5,
                 ranks: int = 5,
                 players: int = 2,
                 hand_size: int = 5,
                 max_information_tokens: int = 8,
                 max_life_tokens: int = 3,
                 observation_type: int = 1,
                 random_start_player: bool = False,
                 seed: int = 0,
                 ):

        """
        Parameter descriptions :
              - colors: int, Number of colors in [2,5].
              - ranks: int, Number of ranks in [2,5].
              - players: int, Number of players in [2,5].
              - hand_size: int, Hand size in [2,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                    0: Minimal observation.
                    1: First-order common knowledge observation.
              - random_start_player: bool, Random start player.
        Common game configurations:
            Hanabi-Full (default) :  {
                "colors": 5,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "hand_size": (4 if players >= 4 else 5)
                "observation_type": 1,
                "hand_size": 2
                }
            Hanabi-Small : {
                "colors": 2,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 3
                "hand_size": 2,
                "max_life_tokens": 1
                "observation_type": 1}
            Hanabi-Very-Small : {
                "colors": 1,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 3
                "hand_size": 2,
                "max_life_tokens": 1
                "observation_type": 1}
        """
        #EzPickle.__init__(**locals())
        """
            self,
            colors,
            ranks,
            players,
            hand_size,
            max_information_tokens,
            max_life_tokens,
            observation_type,
            random_start_player,
        )
        """

        gym.Env.__init__(self)

        self._config = {
            'colors': colors,
            'ranks': ranks,
            'players': players,
            'hand_size': hand_size,
            'max_information_tokens': max_information_tokens,
            'max_life_tokens': max_life_tokens,
            'observation_type': observation_type,
            'random_start_player': random_start_player,
            'seed': seed
        }
        
        self.hanabi_env: HanabiEnv = HanabiEnv(config=self._config)

        self.agents = list(range(self.hanabi_env.players))
        self.current_agent: int

        # Sets hanabi game to clean state and updates all internal dictionaries
        self.reset()

        agent_action_space = gym.spaces.Discrete(self.hanabi_env.num_moves())
        #self.action_space = gym.spaces.Tuple([agent_action_space for _ in self.agents])
        self.action_space = agent_action_space
        
        """
        agent_observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.hanabi_env.vectorized_observation_shape()[0],),
                    dtype=np.float32
                ),
                'action_mask': gym.spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(self.hanabi_env.num_moves(),), 
                    dtype=np.int8
                )
            }
        )
        """
        agent_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.hanabi_env.vectorized_observation_shape()[0],),
            dtype=np.float32
        )

        action_mask_observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.hanabi_env.num_moves(),), 
            dtype=np.int8
        )

        #self.observation_space = gym.spaces.Tuple([agent_observation_space for _ in self.agents])
        self.observation_space = agent_observation_space


    def seed(self, seed=None):
        self._config['seed'] = seed
        self.hanabi_env = HanabiEnv(config=self._config)

    def reset(self):
        # Reset underlying hanabi reinforcement learning environment
        obs = self.hanabi_env.reset()

        self.current_agent = obs['current_player']
        
        self.rewards = [0 for _ in self.agents]
        
        # Reset internal state
        self._encode_observations(obs=obs)

        return self.observations, self.infos

    def _step_agents(self):
        self.current_agent = (self.current_agent+1)%len(self.agents)

    def step(self, actions: List[int]) -> Optional[Union[np.ndarray, List[List[dict]]]]:
        """ 
        Advances the environment by one step. 
        Actions must be within self.legal_moves, otherwise throws error.
        Returns:
            observation: Optional List of new observations of agent at turn after the action step is performed.
            By default a list of integers, describing the logic state of the game from the view of the agent.
        """
        assert not self.done

        agent_on_turn = self.dict_observation['current_player']
        assert self.current_agent == agent_on_turn
        self.legal_moves = self.dict_observation['player_observations'][agent_on_turn]['legal_moves_as_int']
        action = actions[agent_on_turn].item()

        if action not in self.legal_moves:
            raise ValueError('Illegal action. Please choose between legal actions, as documented in dict self.infos')

        # Iterate current_agent pointer: 
        self._step_agents()

        # Apply action
        next_observations, reward, done, _ = self.hanabi_env.step(action=action)

        # Update internal state
        self._encode_observations(obs=next_observations, reward=reward, done=done)

        return self.observations, self.rewards, self.done, self.infos

    def _encode_observations(self, obs: Dict, reward: Optional[float] = 0, done: Optional[bool] = False):
        self.dict_observation = obs 
        self.observations = [
            obs['player_observations'][player_idx]['vectorized']
            for player_idx in self.agents
        ]
        self.rewards = [reward for _ in self.agents]
        self.done = done

        self.infos = []
        for player_idx in self.agents:
            legal_moves=self.dict_observation['player_observations'][player_idx]['legal_moves_as_int']
            info = dict(
                #legal_moves=self.dict_observation['player_observations'][player_idx]['legal_moves_as_int'],
                #observations_dict=self.dict_observation['player_observations'][player_idx],
            )
            action_mask=np.zeros((1,self.hanabi_env.num_moves()))
            np.put(action_mask, ind=legal_moves, v=1)
            info['action_mask'] = action_mask
            info['legal_actions'] = action_mask
            self.infos.append(info)

    def render(self, mode='human'):
        """ 
        Supports console print only. Prints player's data.
        """
        player_data = self.dict_observation['player_observations']
        print("Active player:", f"Player_{self.agents[player_data[0]['current_player_offset']]}")
        for i, d in enumerate(player_data):
            print(f"Player_{self.agents[i]}")
            print("========")
            print(d['pyhanabi'])
            print()


class HanabiFullGymEnv(HanabiGymEnv, EzPickle):
    def __init__(self, seed: int = 0,):
        """
        Parameter descriptions :
              - colors: int, Number of colors in [2,5].
              - ranks: int, Number of ranks in [2,5].
              - players: int, Number of players in [2,5].
              - hand_size: int, Hand size in [2,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                    0: Minimal observation.
                    1: First-order common knowledge observation.
              - random_start_player: bool, Random start player.

        Hanabi-Full (default) :  {
                "colors": 5,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "hand_size": (4 if players >= 4 else 5)
                "observation_type": 1,
                "hand_size": 2
                }
        """
        EzPickle.__init__(**locals())
        
        HanabiGymEnv.__init__(
            self=self,
            colors = 5,
            ranks = 5,
            players = 2,
            hand_size= 5,
            max_information_tokens= 8,
            max_life_tokens= 3,
            observation_type= 1,
            random_start_player= False,
            seed= seed,
        )


class HanabiSmallGymEnv(HanabiGymEnv, EzPickle):
    def __init__(self, seed: int = 0,):
        """
        Parameter descriptions :
              - colors: int, Number of colors in [2,5].
              - ranks: int, Number of ranks in [2,5].
              - players: int, Number of players in [2,5].
              - hand_size: int, Hand size in [2,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                    0: Minimal observation.
                    1: First-order common knowledge observation.
              - random_start_player: bool, Random start player.

        Hanabi-Small : {
                "colors": 2,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 3
                "hand_size": 2,
                "max_life_tokens": 1
                "observation_type": 1}
        """
        EzPickle.__init__(**locals())
        
        HanabiGymEnv.__init__(
            self=self,
            colors = 2,
            ranks = 5,
            players = 2,
            hand_size= 2,
            max_information_tokens= 3,
            max_life_tokens= 1,
            observation_type= 1,
            random_start_player= False,
            seed= seed,
        )

class HanabiVerySmallGymEnv(HanabiGymEnv,EzPickle):
    def __init__(self, seed: int = 0,):
        """
        Parameter descriptions :
              - colors: int, Number of colors in [2,5].
              - ranks: int, Number of ranks in [2,5].
              - players: int, Number of players in [2,5].
              - hand_size: int, Hand size in [2,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                    0: Minimal observation.
                    1: First-order common knowledge observation.
              - random_start_player: bool, Random start player.

        Hanabi-Very-Small : {
                "colors": 1,
                "ranks": 5,
                "players": 2,
                "max_information_tokens": 3
                "hand_size": 2,
                "max_life_tokens": 1
                "observation_type": 1}
        """
        EzPickle.__init__(**locals())
        
        HanabiGymEnv.__init__(
            self=self,
            colors = 1,
            ranks = 5,
            players = 2,
            hand_size= 2,
            max_information_tokens= 3,
            max_life_tokens= 1,
            observation_type= 1,
            random_start_player= False,
            seed= seed,
        )
