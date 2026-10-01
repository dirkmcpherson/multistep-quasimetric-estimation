from __future__ import annotations
from math import log, sqrt
from copy import deepcopy
from functools import partial, wraps

import torch
from torch import nn, is_tensor, tensor, Tensor
import torch.nn.functional as F

from torch.nn import Module, Linear

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# policy related

import torchvision.models as models
from torch.distributions import Categorical, Normal, Beta
from x_mlps_pytorch import create_filmable_mlp

# helpers

from torch_einops_utils import batched_index_select, lens_to_mask

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def check_lens(lens, states):
    if not exists(lens):
        return None

    timesteps, device = states.shape[1], states.device

    lens = tensor(lens, device = device) if not is_tensor(lens) else lens.to(device)
    lens = lens.long()

    assert (lens <= timesteps).all(), f'all sequence lengths must be <= timesteps ({timesteps})'

    mask = lens_to_mask(lens, max_len = timesteps)
    assert mask[:, 1].all(), 'all sequences must have at least 2 timesteps'

    return lens

# constants

LinearNoBias = partial(Linear, bias = False)

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

# huberized linex

def huberize(
    clamp_max = 5.,
    clamp_min = None
):
    # clip the argument of any convex loss on the residual, adding the displacement back as a linear tail

    def decorator(loss_fn):
        @wraps(loss_fn)
        def huberized(residual):
            clipped = residual.clamp(min = clamp_min, max = clamp_max)
            return loss_fn(clipped) + (residual - clipped).abs()

        return huberized

    return decorator

@huberize(clamp_max = 5.)
def linex_loss(delta):
    # eq (10) - the paper prints exp(d - d') - d', which has no minimum, so the intended form with gradient exp(d - d') - 1 is used

    return delta.exp() - delta - 1

# quasimetric distance

def default_sym_fn(x, y):
    return (x - y).norm(p = 2, dim = -1)

def default_asym_fn(x, y):
    return (x - y).relu().amax(dim = -1)

def quasimetric_distance(
    x, y,
    asym_x = None,
    asym_y = None,
    *,
    sym_fn = default_sym_fn,
    asym_fn = default_asym_fn
):
    asym_x, asym_y = default(asym_x, x), default(asym_y, y)

    assert x.shape[-1] == y.shape[-1] == asym_x.shape[-1] == asym_y.shape[-1]

    # symmetric

    sym = sym_fn(x, y)

    # asymmetric

    asym = asym_fn(asym_x, asym_y)

    # eq (4)

    distance = sym + asym

    return distance

# metric residual network
# https://arxiv.org/abs/2208.08133

class MetricResidualNetwork(Module):
    def __init__(
        self,
        *,
        sym_network: Module,
        asym_network: Module,
        distance_groups = 8,
        normalize_by_dim = True
    ):
        super().__init__()

        # the two network backbones, producing inputs for symmetric and asymmetric half of quasimetric distance

        self.sym_network = sym_network
        self.asym_network = asym_network

        # distance related

        self.distance_groups = distance_groups

        # the authors' released code divides the mrn distance by sqrt(latent_dim), which keeps initial distances (and the linex residuals) small

        self.normalize_by_dim = normalize_by_dim

    def forward(
        self,
        encoded_left,
        encoded_right,
        reduce_groups = True
    ):
        encoded = [encoded_left, encoded_right]

        sym_x, sym_y = [self.sym_network(t) for t in encoded]
        asym_x, asym_y = [self.asym_network(t) for t in encoded]

        dim_embed = sym_x.shape[-1]
        assert divisible_by(dim_embed, self.distance_groups)

        sym_x, sym_y, asym_x, asym_y = (rearrange(t, '... (g d) -> ... g d', g = self.distance_groups) for t in (sym_x, sym_y, asym_x, asym_y))

        distance = quasimetric_distance(sym_x, sym_y, asym_x, asym_y)

        if self.normalize_by_dim:
            distance = distance / sqrt(dim_embed)

        if not reduce_groups:
            return distance

        return reduce(distance, '... g -> ...', 'mean')

# critic

class Critic(Module):
    def __init__(
        self,
        state_encoder: Module,
        state_action_encoder: Module,
        metric_residual_network: MetricResidualNetwork,
        discount_factor = 0.95,
        action_invariance_loss_weight = 1.,
        paired_loss_weight = 0.5
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.state_action_encoder = state_action_encoder
        self.metric_residual_network = metric_residual_network

        # hyperparameters

        self.discount_factor = discount_factor

        # loss related

        self.action_invariance_loss_weight = action_invariance_loss_weight
        self.paired_loss_weight = paired_loss_weight
        self.has_paired_loss_weight = paired_loss_weight > 0

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def extract_policy(self, policy: Module, states, actions, goals, **kwargs):
        return extract_policy_loss([self], policy, states, actions, goals, **kwargs)

    def actor_distance(
        self,
        states,
        actions,
        goals
    ):
        # d((s, a), g) per sample, used as -Q by the actor

        encoded_state_actions = self.state_action_encoder((states, actions))
        encoded_goals = self.state_encoder(goals)

        return self.metric_residual_network(
            encoded_state_actions,
            encoded_goals
        )

    def actor_loss(self, states, actions, goals):
        return self.actor_distance(states, actions, goals).mean()

    # predicting distances

    def predict_distance(
        self,
        states,
        goals,
        actions = None,
        reduce_groups = True
    ):
        if exists(actions):
            encoded_states = self.state_action_encoder((states, actions))
        else:
            encoded_states = self.state_encoder(states)

        encoded_goals = self.state_encoder(goals)

        return self.metric_residual_network(
            encoded_states,
            encoded_goals,
            reduce_groups = reduce_groups
        )

    def forward(
        self,
        states,
        actions,
        goals,
        waypoints,
        waypoint_dist, # int(b)
    ):
        γ, batch = self.discount_factor, states.shape[0]
        state_encoder, state_action_encoder, metric_residual_network = self.state_encoder, self.state_action_encoder, self.metric_residual_network

        encoded_states = state_encoder(states)
        encoded_state_actions = state_action_encoder((states, actions))
        encoded_waypoints = state_encoder(waypoints)
        encoded_goals = state_encoder(goals)

        # eq (11) - cross-batch goals

        encoded_state_actions_i = rearrange(encoded_state_actions, 'i d -> i 1 d')
        encoded_waypoints_i = rearrange(encoded_waypoints, 'i d -> i 1 d')
        encoded_goals_j = rearrange(encoded_goals, 'j d -> 1 j d')

        dist_q_to_goal = metric_residual_network(
            encoded_state_actions_i,
            encoded_goals_j
        )

        dist_waypoint_to_goal = metric_residual_network(
            encoded_waypoints_i,
            encoded_goals_j
        )

        waypoint_dist = rearrange(waypoint_dist, 'i -> i 1')

        # handle loss

        residual = dist_q_to_goal - (dist_waypoint_to_goal.detach() - waypoint_dist * log(γ))
        loss_matrix = linex_loss(residual)

        loss = loss_matrix.mean()

        if self.has_paired_loss_weight:
            loss = torch.lerp(loss, loss_matrix.diag().mean(), self.paired_loss_weight)

        # section 4.2 - action invariance

        dist_action_invariance = metric_residual_network(
            encoded_states,
            encoded_state_actions,
            reduce_groups = False
        )

        loss_action_invariance = F.mse_loss(dist_action_invariance.neg().exp(), torch.ones_like(dist_action_invariance))

        total_loss = loss + loss_action_invariance * self.action_invariance_loss_weight

        return total_loss, (loss, loss_action_invariance)

# policy extraction - behavior-regularized ddpg (ddpg + bc), following the authors' released code
# https://github.com/WJ2003B/mqe-release/blob/main/impls/agents/mqe.py

def extract_policy_loss(
    critics,
    policy: Module,
    states,
    actions,
    goals,
    bc_loss_weight = 0.1,
    is_image = False,
    lens = None,
    cross_batch_goals = False,   # eq. 15 of the paper permutes goals across the batch for the q term; the authors' code uses the same (future, same-trajectory) goal as the bc term
    normalize_q = True,          # divide the q term by its detached mean magnitude so bc_loss_weight is on ogbench's ddpg+bc scale
    use_mean_action = True,      # evaluate q at the distribution mean (the authors use a constant-std actor and its mode); otherwise rsample
    action_clamp = None          # e.g. (-1., 1.) to clip the q-term action to the action bounds, as the authors do
):
    batch, device = states.shape[0], states.device

    is_seq = states.ndim == (5 if is_image else 3)

    if is_seq:
        lens = check_lens(lens, states)

        if goals.ndim == states.ndim:
            goals = batched_index_select(goals, lens - 1) if exists(lens) else goals[:, -1]

        states = states[:, 0]
        actions = actions[:, 0]

    # behavior cloning loss

    action_dist = policy(states, goals)

    bc_loss = states.new_zeros(())

    if bc_loss_weight > 0.:
        log_prob = action_dist.log_prob(actions)
        log_prob = reduce(log_prob, 'b ... -> b', 'sum')
        bc_loss = -log_prob.mean()

    # q term - minimizing the distance is maximizing q

    if cross_batch_goals:
        goals_q = goals[torch.randperm(batch, device = device)]
        action_dist_q = policy(states, goals_q)
    else:
        goals_q, action_dist_q = goals, action_dist

    if use_mean_action and hasattr(action_dist_q, 'mean'):
        pred_actions = action_dist_q.mean
    else:
        pred_actions = action_dist_q.rsample()

    if exists(action_clamp):
        pred_actions = pred_actions.clamp(*action_clamp)

    # pessimistic over the critic ensemble: min q = max distance

    dist = torch.stack([critic.actor_distance(states, pred_actions, goals_q) for critic in critics]).amax(dim = 0)

    if normalize_q:
        dist = dist / (dist.abs().mean().detach() + 1e-6)

    q_loss = dist.mean()

    total_loss = q_loss + bc_loss_weight * bc_loss

    return total_loss, (q_loss, bc_loss)

# helpers for the critic ensemble

def reinit_(module: Module):
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    return module

# main class

class MultistepQuasimetricEstimation(Module):
    def __init__(
        self,
        state_encoder: Module,
        state_action_encoder: Module,
        metric_residual_network: MetricResidualNetwork,
        discount_factor = 0.95,
        waypoint_discount = 0.95,
        max_waypoint_dist = None,
        next_timestep_prob = 0.2,
        action_invariance_loss_weight = 1.,
        paired_loss_weight = 0.5,
        critic_ensemble = 2
    ):
        super().__init__()

        # the authors train an ensemble of 2 critics (independently initialised copies) and take the pessimistic one for the actor
        # additional members are deep copies of the given networks with their parameters re-initialised

        def make_critic(state_encoder, state_action_encoder, metric_residual_network):
            return Critic(
                state_encoder = state_encoder,
                state_action_encoder = state_action_encoder,
                metric_residual_network = metric_residual_network,
                discount_factor = discount_factor,
                action_invariance_loss_weight = action_invariance_loss_weight,
                paired_loss_weight = paired_loss_weight
            )

        assert critic_ensemble >= 1

        critics = [make_critic(state_encoder, state_action_encoder, metric_residual_network)]

        for _ in range(critic_ensemble - 1):
            critics.append(make_critic(*[reinit_(deepcopy(net)) for net in (state_encoder, state_action_encoder, metric_residual_network)]))

        self.critics = nn.ModuleList(critics)

        self.max_waypoint_dist = max_waypoint_dist
        self.waypoint_discount = waypoint_discount
        self.discount_factor = discount_factor
        self.next_timestep_prob = next_timestep_prob

    @property
    def critic(self):
        # first ensemble member, kept for backwards compatibility
        return self.critics[0]

    def extract_policy(
        self,
        policy,
        states,
        actions,
        goals,
        **kwargs
    ):
        return extract_policy_loss(list(self.critics), policy, states, actions, goals, **kwargs)

    # predicting distance and steps

    def predict_distance(
        self,
        *args,
        return_steps = False,
        ensemble_reduce = 'mean',  # 'mean' | 'max' (pessimistic) | None (stack over members)
        **kwargs
    ):
        dist = torch.stack([critic.predict_distance(*args, **kwargs) for critic in self.critics])

        if ensemble_reduce == 'mean':
            dist = dist.mean(dim = 0)
        elif ensemble_reduce == 'max':
            dist = dist.amax(dim = 0)

        if not return_steps:
            return dist

        # convert to expected steps: d = -k * log(γ)  =>  k = d / |log(γ)|

        steps_per_unit = abs(log(self.discount_factor))
        return dist / steps_per_unit

    def forward(
        self,
        states,
        actions,
        goals = None,
        lens = None
    ):
        # default goals to states (terminal frame of trajectory window) if not explicitly given

        goals = default(goals, states)
        batch, timesteps, device = *states.shape[:2], states.device

        assert timesteps >= 2, f'sequence must have at least 2 timesteps (got {timesteps})'

        lens = check_lens(lens, states)

        # max waypoint distance, capped at goal distance K (lens - 1 or timesteps - 1)

        max_waypoint = (lens - 1) if exists(lens) else (timesteps - 1)

        if exists(self.max_waypoint_dist):
            max_waypoint = max_waypoint.clamp(max = self.max_waypoint_dist) if is_tensor(max_waypoint) else min(self.max_waypoint_dist, max_waypoint)

        # section 4.1 - multistep returns with quasimetric metric residual network

        is_next_timestep = torch.full((batch,), self.next_timestep_prob, device = device).bernoulli() == 1

        # eq. (8) of the paper - waypoint distance capped at the goal distance K, where the goal is the last frame of the trajectory window (k' ~ min(geometric(1 - lambda), K))

        # note: on cuda, geometric_ can return 0 (curand uniform includes 1.), so clamp to the documented support {1, 2, ...}

        k_prime = torch.empty((batch,), device = device).geometric_(1. - self.waypoint_discount).long().clamp(min = 1)
        waypoint_dist = k_prime.clamp(max = max_waypoint)
        waypoint_dist = torch.where(is_next_timestep, 1, waypoint_dist)

        # waypoints selected, then waypoints and their sampled timesteps from starting state is used to calculate the loss

        waypoints = batched_index_select(states, waypoint_dist)

        # select goal from last frame of episode (per sample if lens is given)

        if goals.ndim == states.ndim:
            goals = batched_index_select(goals, lens - 1) if exists(lens) else goals[:, -1]

        # same batch, waypoints and goals for every ensemble member, losses summed

        total_loss, total_multistep, total_invariance = 0., 0., 0.

        for critic in self.critics:
            loss, (multistep_loss, invariance_loss) = critic(states[:, 0], actions[:, 0], goals, waypoints, waypoint_dist)
            total_loss, total_multistep, total_invariance = total_loss + loss, total_multistep + multistep_loss, total_invariance + invariance_loss

        return total_loss, (total_multistep, total_invariance)

# shorthand

MRN = MetricResidualNetwork
MQE = MultistepQuasimetricEstimation

# policy

class ResNet34Encoder(Module):
    def __init__(
        self,
        *,
        pretrained = False,
        pool = True
    ):
        super().__init__()
        resnet = models.resnet34(pretrained = pretrained)

        modules = list(resnet.children())[:-1]

        if not pool:
            modules = modules[:-1]
            modules.append(Rearrange('b c h w -> b (h w) c'))
        else:
            modules.append(Rearrange('b c 1 1 -> b c'))

        self.encoder = nn.Sequential(*modules)

    @property
    def output_dim(self):
        return 512

    def forward(self, x):
        return self.encoder(x)

# action distributions

class ActionDistribution(Module):
    @property
    def expansion_factor(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class DiscreteAction(ActionDistribution):
    @property
    def expansion_factor(self):
        return 1

    def forward(self, x):
        return Categorical(logits = x)

class ContinuousAction(ActionDistribution):
    def __init__(self, min_log_std = -5.0, max_log_std = 2.0):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    @property
    def expansion_factor(self):
        return 2

    def forward(self, x):
        mean, log_std = x.chunk(2, dim = -1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return Normal(mean, log_std.exp())

class BetaAction(ActionDistribution):
    @property
    def expansion_factor(self):
        return 2

    def forward(self, x):
        alpha, beta = x.chunk(2, dim = -1)
        alpha, beta = [F.softplus(t) + 1. for t in (alpha, beta)]
        return Beta(alpha, beta)

# policy

class Policy(Module):
    def __init__(
        self,
        *,
        action_dim,
        dim = None,
        action_dist = None,
        state_encoder = None,
        goal_encoder = None,
        pretrained = False,
        mlp_depth = 3,
        mlp_hidden_dim = 256
    ):
        super().__init__()
        self.action_dist = default(action_dist, ContinuousAction())

        self.state_encoder = default(state_encoder, ResNet34Encoder(pretrained = pretrained, pool = False))
        self.goal_encoder = default(goal_encoder, ResNet34Encoder(pretrained = pretrained, pool = True))

        if not exists(dim):
            assert hasattr(self.state_encoder, 'output_dim'), 'dim must be given if using custom state_encoder'
            dim = self.state_encoder.output_dim

        dim_out = action_dim * self.action_dist.expansion_factor

        self.mlp = create_filmable_mlp(
            mlp_hidden_dim,
            mlp_depth,
            dim_in = dim,
            dim_out = dim_out,
            cond_dim = dim
        )

    def forward(self, state, goal):
        state_tokens = self.state_encoder(state)
        goal_tokens = self.goal_encoder(goal)

        embed = reduce(state_tokens, 'b n d -> b d', 'mean') if state_tokens.ndim == 3 else state_tokens
        cond = reduce(goal_tokens, 'b n d -> b d', 'mean') if goal_tokens.ndim == 3 else goal_tokens

        out = self.mlp(embed, cond)

        return self.action_dist(out)
