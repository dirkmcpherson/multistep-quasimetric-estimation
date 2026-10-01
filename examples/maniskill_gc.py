"""
goal-conditioned ManiSkill tasks in OGBench's interface, so the same training / evaluation code runs on both

- dataset: ManiSkill demonstrations (replayed to `state` observations) converted to OGBench's npz layout
  (observations incl. the final state of every episode, actions, terminals), written to ~/.ogbench/data/<name>.npz
  and <name>-val.npz, so `ogbench.make_env_and_datasets('<name>')` loads them without downloading
- env: `gymnasium.make('maniskill-<task>-v0')` -> GoalConditionedManiSkill, whose
  reset(options = dict(task_id = k)) resets to the initial state of the k-th held-out demonstration and returns
  info['goal'] = that demonstration's final observation; step() reports the task's own success flag in info['success']

run the conversion once per task:

    python examples/maniskill_gc.py convert --task PickCube-v1
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

DEMO_DIR = Path(os.environ.get('MANISKILL_DEMO_DIR', '~/.maniskill/demos')).expanduser()
DATA_DIR = Path(os.environ.get('OGBENCH_DATA_DIR', '~/.ogbench/data')).expanduser()

TASKS = {
    # ogbench-style dataset name -> maniskill task id
    'maniskill-pickcube-play-v0': 'PickCube-v1',
    'maniskill-pushcube-play-v0': 'PushCube-v1',
}
CONTROL_MODE = 'pd_ee_delta_pos'
NUM_VAL = 100          # held-out demonstrations: validation split and evaluation goals
NUM_EVAL_TASKS = 5     # ogbench evaluates 5 "tasks"; each maps to a block of held-out demonstrations
MAX_EPISODE_STEPS = 100

def dataset_name_for(task_id):
    return {v: k for k, v in TASKS.items()}[task_id]

def demo_file(task_id):
    return DEMO_DIR / task_id / 'motionplanning' / f'trajectory.state.{CONTROL_MODE}.physx_cpu.h5'

def load_demos(task_id):
    """returns list of dicts with obs (T+1, D), actions (T, A), success (bool), env_state_0 (dict)"""
    f = demo_file(task_id)
    meta = json.load(open(f.with_suffix('.json')))
    demos = []
    with h5py.File(f) as h:
        for ep in meta['episodes']:
            g = h[f"traj_{ep['episode_id']}"]
            obs = np.asarray(g['obs'], dtype = np.float32)
            if task_id == 'PickCube-v1':
                obs[0, 18] = 0.   # the recorder writes is_grasped = 1 at t = 0 although the cube starts on the table; the live env reports 0
            acts = np.asarray(g['actions'], dtype = np.float32)
            succ = bool(np.asarray(g['success'])[-1]) if 'success' in g else True
            state0 = {k: np.asarray(v[0]) for k, v in flatten_h5(g['env_states']).items()}
            demos.append(dict(obs = obs, actions = acts, success = succ, env_state_0 = state0, episode_id = ep['episode_id'], seed = ep.get('episode_seed', ep.get('reset_kwargs', {}).get('seed', 0))))
    return demos

def flatten_h5(group, prefix = ''):
    out = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            out[prefix + k] = v
        else:
            out.update(flatten_h5(v, prefix + k + '/'))
    return out

def unflatten(d):
    out = {}
    for k, v in d.items():
        cur = out
        parts = k.split('/')
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def convert(task):
    """write ogbench-format train / val npz files from the replayed demonstrations"""
    demos = [d for d in load_demos(task) if d['success']]
    rng = np.random.RandomState(0)
    order = rng.permutation(len(demos))
    val_idx, train_idx = order[:NUM_VAL], order[NUM_VAL:]
    name = dataset_name_for(task)
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    for split, idxs in (('', train_idx), ('-val', val_idx)):
        obs, acts, terms = [], [], []
        for i in idxs:
            d = demos[i]
            T = d['actions'].shape[0]
            assert d['obs'].shape[0] == T + 1, (d['obs'].shape, T)
            obs.append(d['obs'])
            acts.append(np.concatenate([d['actions'], np.zeros_like(d['actions'][:1])]))   # final state has no action
            t = np.zeros(T + 1, dtype = np.float32); t[-1] = 1.
            terms.append(t)
        np.savez(DATA_DIR / f'{name}{split}.npz', observations = np.concatenate(obs), actions = np.concatenate(acts), terminals = np.concatenate(terms))
        print(f'{name}{split}.npz: {len(idxs)} episodes, {sum(len(o) for o in obs)} states, obs dim {obs[0].shape[1]}, act dim {acts[0].shape[1]}')
    # held-out demonstrations used as evaluation goals
    np.save(DATA_DIR / f'{name}-evalgoals.npy', np.array([demos[i]['episode_id'] for i in val_idx]))
    print(f'{len(demos)} successful demos of {len(load_demos(task))}; {NUM_VAL} held out for validation / evaluation goals')

class GoalConditionedManiSkill(gym.Env):
    """ogbench-style goal-conditioned wrapper: goals are held-out demonstration end states"""

    metadata = {'render_modes': []}

    def __init__(self, task_id, max_episode_steps = MAX_EPISODE_STEPS):
        import mani_skill.envs  # noqa: registers tasks
        self.task_id = task_id
        self.env = gym.make(task_id, obs_mode = 'state', control_mode = CONTROL_MODE, render_mode = None, num_envs = 1, sim_backend = 'physx_cpu', max_episode_steps = max_episode_steps)
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (self.env.observation_space.shape[-1],), dtype = np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape = (self.env.action_space.shape[-1],), dtype = np.float32)
        name = dataset_name_for(task_id)
        eval_ids = set(np.load(DATA_DIR / f'{name}-evalgoals.npy').tolist())
        self.goal_demos = [d for d in load_demos(task_id) if d['episode_id'] in eval_ids]
        self.num_tasks = NUM_EVAL_TASKS
        self.task_infos = [dict(task_name = f'task{i + 1}') for i in range(NUM_EVAL_TASKS)]   # ogbench's evaluation loop reads these
        self._per_task = len(self.goal_demos) // NUM_EVAL_TASKS
        self._counters = {}
        self._t = 0

    def reset(self, seed = None, options = None):
        options = options or {}
        task = int(options.get('task_id', 1))
        n = self._counters.get(task, 0); self._counters[task] = n + 1
        demo = self.goal_demos[(task - 1) * self._per_task + n % self._per_task]
        self.env.reset(seed = int(demo['seed']))
        self.env.unwrapped.set_state_dict(unflatten({k: np.asarray(v)[None] for k, v in demo['env_state_0'].items()}))
        ob = self._obs(self.env.unwrapped.get_obs())
        self._t = 0
        return ob, dict(goal = demo['obs'][-1].astype(np.float32))

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype = np.float32)[None])
        self._t += 1
        success = bool(np.asarray(info['success']).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0]) or self._t >= self.max_episode_steps
        return self._obs(ob), float(success), False, truncated, dict(success = float(success))

    @staticmethod
    def _obs(ob):
        ob = ob.cpu().numpy() if hasattr(ob, 'cpu') else np.asarray(ob)
        return ob.reshape(-1).astype(np.float32)

    def close(self):
        self.env.close()

for _name, _task in TASKS.items():
    _env_id = _name.replace('-play-', '-')   # 'maniskill-pickcube-v0', what ogbench derives from the dataset name
    if _env_id not in gym.registry:
        gym.register(id = _env_id, entry_point = 'maniskill_gc:GoalConditionedManiSkill', kwargs = dict(task_id = _task))

if __name__ == '__main__':
    import fire
    fire.Fire(dict(convert = convert))
