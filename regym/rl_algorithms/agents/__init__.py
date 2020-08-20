from .gym_rock_paper_scissors_agent import MixedStrategyAgent
from .tabular_q_learning_agent import build_TabularQ_Agent, TabularQLearningAgent
from .dqn_agent import build_DQN_Agent, DQNAgent
from .r2d3_agent import build_R2D3_Agent, R2D3Agent
from .ther_agent import build_THER_Agent, THERAgent
from .ppo_agent import build_PPO_Agent, PPOAgent
from .reinforce_agent import build_Reinforce_Agent, ReinforceAgent
from .a2c_agent import build_A2C_Agent, A2CAgent
from .ddpg_agent import build_DDPG_Agent, DDPGAgent
from .td3_agent import build_TD3_Agent, TD3Agent
from .i2a_agent import build_I2A_Agent, I2AAgent
from .deterministic_agent import build_Deterministic_Agent, DeterministicAgent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent')
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0], name='PaperAgent')
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1], name='ScissorsAgent')
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3], name='RandomAgent')
