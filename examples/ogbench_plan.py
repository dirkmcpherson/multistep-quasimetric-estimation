"""
model-predictive planning with the trained MQE critic as the cost: does the learned distance beat the policy extracted from it?

MPPI over H-step action sequences, imagined with the ensemble dynamics model from ogbench_dynamics.py, scored by
sum_t d(s_t, g) from the MQE critic (or a naive normalized euclidean distance to the goal state as a control),
optionally warm-started from the amortized policy. Receding horizon, OGBench evaluation protocol.

    python examples/ogbench_plan.py --ckpt runs/branch/cube-single-play-v0/seed0/ckpt_1000000.pt \
        --dynamics runs/dynamics/cube-single-play-v0.pt --cost critic --prior policy --out runs/plan/critic_policy.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ogbench_mqe import GoalConditionedPolicy
from ogbench_diagnostics import build_mqe
from ogbench_dynamics import DynamicsEnsemble

class Planner:
    def __init__(self, mqe, policy, dyn, cost = 'critic', prior = 'policy', horizon = 8, samples = 512, iters = 2,
                 noise = 0.3, temperature = 0.1, disagreement_penalty = 0., act_dim = None, device = 'cuda'):
        self.mqe, self.policy, self.dyn = mqe, policy, dyn
        self.cost, self.prior = cost, prior
        self.H, self.N, self.iters = horizon, samples, iters
        self.noise, self.temp, self.beta = noise, temperature, disagreement_penalty
        self.act_dim, self.device = act_dim, device
        self.mean = None

    def reset(self):
        self.mean = None

    @torch.no_grad()
    def stage_cost(self, states, goal):
        # states (N, D), goal (D,) -> (N,)
        if self.cost == 'critic':
            return self.mqe.predict_distance(states, goal.expand_as(states))
        elif self.cost == 'euclid':
            return ((states - goal) / self.dyn.obs_std).norm(dim = -1)
        raise ValueError(self.cost)

    @torch.no_grad()
    def nominal_from_policy(self, state, goal):
        seq, s = [], state.unsqueeze(0)
        for _ in range(self.H):
            a = self.policy(s, goal.unsqueeze(0)).mean.clamp(-1, 1)
            seq.append(a.squeeze(0)); s = self.dyn(s, a)
        return torch.stack(seq)  # (H, A)

    @torch.no_grad()
    def act(self, state, goal):
        state = torch.as_tensor(state, dtype = torch.float32, device = self.device)
        goal = torch.as_tensor(goal, dtype = torch.float32, device = self.device)
        if self.prior == 'policy':
            mean = self.nominal_from_policy(state, goal)
        elif self.mean is not None:
            mean = torch.cat([self.mean[1:], self.mean[-1:]])  # shift the previous plan
        else:
            mean = torch.zeros(self.H, self.act_dim, device = self.device)
        std = torch.full_like(mean, self.noise)
        for _ in range(self.iters):
            acts = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(self.H, self.N, self.act_dim, device = self.device)).clamp(-1, 1)
            if self.prior == 'policy':
                acts[:, 0] = mean  # keep the nominal plan as one candidate
            s = state.unsqueeze(0).expand(self.N, -1)
            cost = torch.zeros(self.N, device = self.device)
            for h in range(self.H):
                preds = self.dyn.forward_all(s, acts[h])          # (members, N, D)
                s = preds.mean(0)
                cost = cost + self.stage_cost(s, goal)
                if self.beta > 0:
                    cost = cost + self.beta * preds.std(0).norm(dim = -1)
            w = torch.softmax(-(cost - cost.min()) / self.temp, dim = 0)      # (N,)
            mean = (w[None, :, None] * acts).sum(1)
            std = ((w[None, :, None] * (acts - mean.unsqueeze(1)) ** 2).sum(1)).sqrt().clamp(min = 0.05)
        self.mean = mean
        return mean[0].clamp(-1, 1).cpu().numpy()

def main(ckpt, dynamics, cost = 'critic', prior = 'policy', horizon = 8, samples = 512, iters = 2, noise = 0.3, temperature = 0.1,
         disagreement_penalty = 0., eval_episodes = 50, num_tasks = 5, out = None, device = 'cuda', seed = 0):
    import ogbench
    torch.manual_seed(seed); np.random.seed(seed)
    state = torch.load(ckpt, map_location = device, weights_only = False)
    config = state['config']; env_name = config['env_name']
    env = ogbench.make_env_and_datasets(env_name, env_only = True)
    obs_dim = env.observation_space.shape[0]; act_dim = env.action_space.shape[0]
    mqe = build_mqe(config, obs_dim, act_dim).to(device); mqe.load_state_dict(state['mqe']); mqe.eval()
    policy = GoalConditionedPolicy(obs_dim, act_dim, config['hidden_dim']).to(device); policy.load_state_dict(state['policy']); policy.eval()
    d = torch.load(dynamics, map_location = device, weights_only = False)
    dyn = DynamicsEnsemble(d['obs_dim'], d['act_dim'], members = d['members']).to(device); dyn.load_state_dict(d['state_dict']); dyn.eval()
    planner = Planner(mqe, policy, dyn, cost = cost, prior = prior, horizon = horizon, samples = samples, iters = iters, noise = noise,
                      temperature = temperature, disagreement_penalty = disagreement_penalty, act_dim = act_dim, device = device)
    results, t0 = {}, time.time()
    for task_id in range(1, num_tasks + 1):
        succ = []
        for _ in range(eval_episodes):
            ob, info = env.reset(options = dict(task_id = task_id, render_goal = False))
            goal = np.asarray(info['goal'], dtype = np.float32); planner.reset()
            done, success = False, 0.
            while not done:
                ob, r, term, trunc, info = env.step(planner.act(ob, goal))
                done = term or trunc; success = float(info['success'])
            succ.append(success)
        results[f'task{task_id}'] = float(np.mean(succ))
        print(f'task {task_id}: {results[f"task{task_id}"]:.3f}  ({time.time() - t0:.0f}s)', flush = True)
    results['overall'] = float(np.mean([results[f'task{i}'] for i in range(1, num_tasks + 1)]))
    results.update(dict(cost = cost, prior = prior, horizon = horizon, samples = samples, iters = iters, noise = noise, temperature = temperature,
                        disagreement_penalty = disagreement_penalty, eval_episodes = eval_episodes, ckpt = str(ckpt), dynamics = str(dynamics), seconds = time.time() - t0))
    print(json.dumps({k: v for k, v in results.items() if k.startswith(('task', 'overall'))}))
    if out:
        Path(out).parent.mkdir(parents = True, exist_ok = True); Path(out).write_text(json.dumps(results, indent = 2))

if __name__ == '__main__':
    fire.Fire(main)
