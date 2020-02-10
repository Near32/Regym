from .n_bits_swap_env import NBitsSwapEnv
from .n_bits_swap_mnist_env import NBitsSwapMNISTEnv

from gym.envs.registration import register

for n in range(40):
    register(
        id=f'{n}BitsSwap-v0',
        entry_point='regym.environments.envs.gym_envs.n_bits_swap_env:NBitsSwapEnv',
        kwargs={'n' : n, 'fixed_goal' : False},
    )
    register(
        id=f'{n}BitsSwap-FixedGoal-v0',
        entry_point='regym.environments.envs.gym_envs.n_bits_swap_env:NBitsSwapEnv',
        kwargs={'n' : n, 'fixed_goal' : True},
    )
    register(
        id=f'{n}BitsSwapMNIST-v0',
        entry_point='regym.environments.envs.gym_envs.n_bits_swap_mnist_env:NBitsSwapMNISTEnv',
        kwargs={'n' : n, 'fixed_goal' : False, 'train': True},
    )
