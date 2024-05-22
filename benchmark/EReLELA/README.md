# EReLELA : Exploration in Reinforcement Learning via Emergent Language Abstractions

Instruction-following from prompts in Natural Languages (NLs) is an important benchmark for Human-AI collaboration. 
Training Embodied AI agents for instruction-following with Reinforcement Learning (RL) poses a strong exploration challenge.
Previous works have shown that NL-based state abstractions can help address the exploitation versus exploration trade-off in RL. 
However, NLs descriptions are not always readily available and are expensive to collect.
We therefore propose to use the Emergent Communication paradigm, where artificial agents are free to learn an emergent language (EL) via referential games, to bridge this gap.  
ELs constitute cheap and readily-available abstractions, as they are the result of an unsupervised learning approach.
In this paper, we investigate (i) how EL-based state abstractions compare to NL-based ones for RL in hard-exploration, procedurally-generated environments, and (ii) how properties of the referential games used to learn ELs impact the quality of the RL exploration and learning.
Results indicate that the EL-guided agent, namely EReLELA, achieves similar performance as its NL-based counterparts without its limitations.
Our work shows that Embodied RL agents can leverage unsupervised emergent abstractions to greatly improve their exploration skills in sparse reward settings, thus opening new research avenues between Embodied AI and Emergent Communication.


The following details how to reproduce the main experiments of the paper.


## Installation :

### Regym :

```bash
cd Regym; pip install -e .
```

### ReferentialGym :

```bash
cd ReferentialGym; pip install -e .
```

### Archi :

```bash
cd Archi; pip install -e .
```

### MiniGrid :

```bash
cd MiniGrid; pip install -e .
```

### Miscellianeous :

```bash
pip install wandb ipdb
```

## Reproduce Experiments :

The main experiments take place in the context of the KeyCorridor-S3-R2 environment from MiniGrid. 
All related scripts can be found in the Experiments folder and start with the denomination `keycorridor_S3_R2_dynamic_`.
Each script launchs a single agent for a 1M observation budget.
Please update the `--seed` hyperparameter to run each agent with different random seeds.

Logging is performed via Weights & Biases, thus you will be required to log in.


### RANDOM Agent :

```bash
cd Experiments; ./keycorridor_S3_R2_dynamic_POMDP+R2D2+ELA_NOTRAINING_run.sh
```

### Natural Language Abstractions (NLA) Agent :

```bash
cd Experiments; ./keycorridor_S3_R2_dynamic_POMDP+R2D2+NLA_run.sh
```

### EReLELA Agents :

With Impatient-Only loss function:

```bash
cd Experiments; ./keycorridor_S3_R2_dynamic_AgnosticPOMDP+R2D2+ELA+UniformDistrSampling_run_minimal.sh
```

With STGS-LazImpa loss function:

```bash
cd Experiments; ./keycorridor_S3_R2_dynamic_AgnosticPOMDP+R2D2+LazyELA+UniformDistrSampling_run_minimal.sh
```



