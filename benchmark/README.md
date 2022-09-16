# Benchmarks

The Regym framework is implementing many state-of-the-art *Single-* and *Multi-agent* Deep Reinforcement Learning algorithms, and they have been benchmarked on many environments.

All algorithms comes with a *distributed* implementation where a set of actors agents can gather experiences and synchronise their inner models with the unique learner agent that learns from the gathered experience.


## [DQN](https://arxiv.org/abs/1312.5602) & [PPO](https://arxiv.org/abs/1707.06347) and other baselines:

DQN and PPO  have been benchmarked on the Atari Benchmark, along with the following extensions when viable:
+ [Double DQN](https://arxiv.org/abs/1509.06461),
+ [Double Dueling DQN](https://arxiv.org/abs/1511.06581),
+ [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952),
+ [Deep Recurrent Q-Network](https://arxiv.org/abs/1507.06527)

[TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html) and [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) have been benchmarked on PyBullet/[Roboschool](https://openai.com/blog/roboschool/) replicates of the MuJoCo Continuous Control Benchmark.


## [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495) & [Textual HER (THER)/ HIGhER](https://arxiv.org/abs/1910.09451): 

DQN-HER and THER have been benchmarked on [BabyAI](https://github.com/mila-iqia/babyai) tasks for *instruction-following* benchmarking.

## [R2D2](https://openreview.net/forum?id=r1lyTjAqYX) & [R2D3]():

R2D2 and R2D3 have been benchmarked on many environments, primarily:
+ Atari Benchmark (Single-Agent RL),
+ [SimpleMemoryTesting](https://github.com/Near32/SimpleMemoryTestingEnv) (Single-Agent RL),
+ [CoMaze](https://github.com/Near32/comaze-gym) (Multi-Agent RL),
+ [MineRL](https://minerl.io/) (Single-Agent RL).


## [Simplified Action Decoder (SAD)](https://arxiv.org/abs/1912.02288) & [OtherPlay](https://arxiv.org/abs/2003.02979):

SAD and OP rely on the R2D2 algorithm and they have been benchmarked on many *single-* and *multi-agent* environments:
+ [CoMaze](https://github.com/Near32/comaze-gym)
+ [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment).


