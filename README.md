<img src="./mqe.png" width="400px"></img>

## Multistep Quasimetric Estimation

Exploration and eventually practical implementation for the [Multistep Quasimetric Estimation](https://arxiv.org/abs/2511.07730) proposed by Zheng et al. of Berkeley.

This paper is a coming together of a few ideas: quasimetric distance spaces and successor representations, along with an action invariance loss and other loss designs

## Install

```bash
pip install MQE
```

## Usage

```python
import torch
from torch import nn

from MQE import MQE, MRN, Policy, ContinuousAction
from x_mlps_pytorch import MLP

state_dim, action_dim = 16, 4

mrn = MRN(
    sym_network = MLP(32, 64),
    asym_network = MLP(32, 64),
    distance_groups = 8
)

mqe = MQE(
    state_encoder = MLP(state_dim, 32),
    state_action_encoder = MLP(state_dim + action_dim, 32),
    metric_residual_network = mrn,
    critic_ensemble = 2        # as in the authors' code; extra members are re-initialised deep copies of the networks above
)

policy = Policy(
    action_dim = action_dim,
    dim = 32,
    state_encoder = MLP(state_dim, 32),
    goal_encoder = MLP(state_dim, 32),
    action_dist = ContinuousAction()
)

states = torch.randn(4, 10, state_dim)
actions = torch.randn(4, 10, action_dim)
goals = torch.randn(4, 10, state_dim)

# train critic from offline trajectories

critic_loss, _ = mqe(states, actions, goals)

critic_loss.backward()

# train actor using critic (ddpg + bc, following the authors' released code: same-trajectory goals,
# q term normalized by its mean magnitude, pessimistic over the critic ensemble, action taken at the distribution mean)

policy_loss, _ = mqe.extract_policy(
    policy,
    states,
    actions,
    goals,
    bc_loss_weight = 0.1,
    action_clamp = (-1., 1.)   # clip the q-term action to the action bounds
)

# the paper's eq. 15 (goals permuted across the batch, raw distances) is still available:
#   mqe.extract_policy(..., cross_batch_goals = True, normalize_q = False, use_mean_action = False)

policy_loss.backward()

# inference

action = policy(states[:, 0], goals[:, 0]).sample() # (4, 4)
```

## Matching the authors' released implementation

Three details that matter for policy extraction were taken from the authors' released code ([mqe-release](https://github.com/WJ2003B/mqe-release)) and are now the defaults: the MRN distance is divided by `sqrt(latent_dim)` (`MRN(normalize_by_dim = True)`), two critics are trained and the actor uses the pessimistic one (`MQE(critic_ensemble = 2)`), and `extract_policy` uses the same-trajectory goal for the Q term with the Q term normalized by its mean magnitude. On OGBench `cube-single-play-v0` these change the extracted policy from below behavior cloning (3%) to the level of the authors' code (17% vs 20%, one seed each); see `examples/ogbench_mqe.py` for the full training and evaluation script and `runs/REPORT.md` for the comparison.

## Citations

```bibtex
@misc{zheng2026multistepquasimetriclearningscalable,
    title   = {Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning},
    author  = {Bill Chunyuan Zheng and Vivek Myers and Benjamin Eysenbach and Sergey Levine},
    year    = {2026},
    eprint  = {2511.07730},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.07730},
}
```

```bibtex
@misc{liu2023metricresidualnetworkssample,
    title   = {Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning},
    author  = {Bo Liu and Yihao Feng and Qiang Liu and Peter Stone},
    year    = {2023},
    eprint  = {2208.08133},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2208.08133},
}
```
