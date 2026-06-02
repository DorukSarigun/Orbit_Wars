"""Reworked self-play rollout + PPO for the hierarchical policy.

This is a clean rewrite of the training loop around the redesigned stack:

* features   -> :mod:`reworked_pipeline` (``pack_reworked_batch``)
* policy     -> :class:`reworked_model.ReworkedHierarchicalPolicy`
              (Bernoulli attack/defend + Plackett-Luce priority + Beta commit
               + Beta defense posture)
* logistics  -> :mod:`reworked_allocator` (``allocate_launches``)

Key design choices (differ from the legacy ``rl/rollout.py``):

* **One transition per player per turn.**  The whole hierarchical decision is a
  single RL action; the scripted allocator is a *deterministic* function of it
  (no extra log-prob), so the natural credit unit is the turn, not the launch.
* **Summed log-prob.**  The stored action log-prob is the sum of every sampled
  component (see :meth:`ReworkedHierarchicalPolicy.act`); PPO re-evaluates it
  with :meth:`ReworkedHierarchicalPolicy.evaluate_actions`.
* **Fixed padded width ``P_MAX``.**  Every transition stores features / masks /
  action padded to ``P_MAX`` so the PPO minibatch is a trivial stack.
* **Simple, pluggable reward.**  Default = potential-based shaping on the
  owned-production differential + a terminal win/loss bonus.  Swap via the
  ``reward_fn`` argument.

Env contract is unchanged: ``env.step([launches_p0, launches_p1, ...])`` where
each launch is ``[from_planet_id, angle, ships]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # package layout
    from .obs_builder import build_observation
    from .feature_extraction import parse_planets, parse_fleets, BOARD_SIZE, DEFAULT_MAX_SHIP_SPEED
    from .navigation import compute_launch
    from .reworked_pipeline import pack_reworked_batch, build_allocator_inputs
    from .reworked_allocator import allocate_launches
    from .reworked_model import ReworkedHierarchicalPolicy
    from .reworked_features import (
        discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from obs_builder import build_observation  # type: ignore
    from feature_extraction import parse_planets, parse_fleets, BOARD_SIZE, DEFAULT_MAX_SHIP_SPEED  # type: ignore
    from navigation import compute_launch  # type: ignore
    from reworked_pipeline import pack_reworked_batch, build_allocator_inputs  # type: ignore
    from reworked_allocator import allocate_launches  # type: ignore
    from reworked_model import ReworkedHierarchicalPolicy  # type: ignore
    from reworked_features import (  # type: ignore
        discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM,
    )


P_MAX = 44   # fixed padded planet width (env worst case)
F_MAX = 64


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def _target_masks(state, p_max: int, player: int) -> "tuple[np.ndarray, np.ndarray]":
    """Per-planet attack / defend candidate masks padded to ``p_max``.

    * attack candidates = real planets **not** owned by ``player`` (enemy or
      neutral -- the latter being expansion);
    * defend candidates = real planets owned by ``player``.
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    attack = np.zeros(p_max, dtype=bool)
    defend = np.zeros(p_max, dtype=bool)
    attack[:n] = owner != int(player)
    defend[:n] = owner == int(player)
    return attack, defend


# ---------------------------------------------------------------------------
# Action -> concrete env launches
# ---------------------------------------------------------------------------
def action_to_env_launches(
    obs,
    player: int,
    action_row: dict,
    parsed_planets: list,
    angular_velocity: float,
    *,
    routing_cost: bool = True,
    capture_margin: float = 1.0,
    attack_require_full: bool = True,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> "tuple[list, list]":
    """Turn one sampled hierarchical action into env launches.

    ``action_row`` holds single-sample NumPy arrays (``attack_sel`` / ``defend_sel``
    ``(P,)``, ``order`` ``(P,)``, ``commit_frac`` ``(P,)``, ``potential_weight`` /
    ``horizon`` scalars).  Returns ``(env_action, launches)`` where ``env_action``
    is the ``[[id, angle, ships], ...]`` list for ``env.step`` and ``launches``
    is the raw :class:`reworked_allocator.Launch` list (for logging / trackers).
    """
    n = int(np.asarray(obs.planet_mask).sum())
    attack_sel = np.asarray(action_row["attack_sel"]).reshape(-1)
    defend_sel = np.asarray(action_row["defend_sel"]).reshape(-1)
    order = np.asarray(action_row["order"]).reshape(-1)
    commit_frac = np.asarray(action_row["commit_frac"], dtype=np.float64).reshape(-1)[:n]

    attack_targets = [t for t in range(n) if attack_sel[t] > 0.5]
    defend_targets = [t for t in range(n) if defend_sel[t] > 0.5]
    sel = set(attack_targets) | set(defend_targets)
    priority_order = [int(t) for t in order if 0 <= int(t) < n and int(t) in sel]

    if not priority_order:
        return [], []

    inputs = build_allocator_inputs(
        obs, player,
        potential_weight=float(action_row["potential_weight"]),
        horizon=float(action_row["horizon"]),
        routing_cost=routing_cost,
        max_speed=max_speed,
    )
    launches = allocate_launches(
        **inputs,
        player=player,
        priority_order=priority_order,
        attack_targets=attack_targets,
        defend_targets=defend_targets,
        commit_fraction=commit_frac,
        capture_margin=capture_margin,
        attack_require_full=attack_require_full,
    )

    planet_by_id = {int(p.id): p for p in parsed_planets}
    ids = np.asarray(obs.planet_ids).reshape(-1)
    env_action: list = []
    for L in launches:
        src_id = int(ids[L.src]) if 0 <= L.src < len(ids) else -1
        tgt_id = int(ids[L.tgt]) if 0 <= L.tgt < len(ids) else -1
        src = planet_by_id.get(src_id)
        tgt = planet_by_id.get(tgt_id)
        if src is None or tgt is None:
            continue
        res = compute_launch(src, tgt, int(L.ships), parsed_planets, angular_velocity,
                             max_speed=max_speed)
        if res is None:
            continue
        env_action.append([src.id, float(res["angle"]), int(res["ships"])])
    return env_action, launches


# ---------------------------------------------------------------------------
# Reward: potential-based shaping (Ng et al. 1999) on an economic potential
# ---------------------------------------------------------------------------
# The ONLY ground-truth objective is the terminal win/loss (+/-1).  Everything
# else is *potential-based* shaping, which is provably policy-invariant (it
# never changes the optimal policy) and hack-proof (it telescopes, so any cycle
# of moves nets zero shaping reward -> no ping-pong exploit).
#
# Division of labour:
#   * the POTENTIAL Phi(s) decides *what* a state is worth -- pluggable below;
#   * the SHAPING FORMULA  F = gamma * Phi(s') - Phi(s)  is fixed and principled
#     (it is what makes the shaping reward-neutral);
#   * GAE / gamma then propagate each turn's reward back to the decision -- the
#     shaping only shortens that propagation distance by placing the reward next
#     to the event that earned it.


@dataclass
class RewardConfig:
    """Reward weights for the potential-based shaping scheme."""
    gamma: float = 0.99                 # RL discount used INSIDE the shaping term
    gamma_eco: float = DEFAULT_GAMMA    # economic horizon discount in the dpp potential
    potential_norm: float = DEFAULT_NORM  # common currency scale (keeps Phi ~O(1))
    ship_weight: float = 1.0            # value of one ship-in-hand vs. one future ship
    count_inflight: bool = True         # count own in-flight fleets as ship stock
    shaping_weight: float = 1.0
    terminal_weight: float = 1.0
    enable_shaping: bool = True         # ablation switch (shaping on/off)


def _ship_stock(planets, fleets, *, count_inflight: bool) -> dict:
    """Total ships per owner = garrisons (+ owned in-flight fleets).

    Counting in-flight fleets is what makes a *launch* leave the potential
    unchanged: ships only move between buckets, they are not destroyed.  Only
    real combat losses shrink an owner's stock.
    """
    stock: dict = {}
    for pl in planets:
        if pl.owner != -1:
            stock[pl.owner] = stock.get(pl.owner, 0.0) + float(pl.ships)
    if count_inflight:
        for fl in fleets:
            if fl.owner != -1:
                stock[fl.owner] = stock.get(fl.owner, 0.0) + float(fl.ships)
    return stock


def economic_potential(env_obs, player: int, remaining_turns: int, cfg: RewardConfig) -> float:
    r"""Default potential ``Phi(s)`` = owned **total economic asset** differential.

    Two asset classes, measured in one common currency (ships):

    * **Production** -- ``dpp_raw = prod * H(R)`` is literally "how many ships
      this planet will yield over the remaining horizon", i.e. *future* ships.
    * **Ships in hand** -- garrisons plus owned in-flight fleets, valued at
      ``ship_weight`` per ship (``~1`` => a ship now ~ a future ship).

    .. math::

        \Phi(s) = \frac{1}{\text{norm}}\Big[
            \sum_{g\in\text{mine}} \big(\mathrm{dpp\_raw}(g) + w\,\text{ships}\big)
          - \sum_{g\in\text{enemy}} (\dots) \Big]

    Because ships now enter ``Phi``, spending/losing ships to capture a planet
    automatically subtracts from the shaped reward: a capture only pays off when
    the production gained exceeds the ships lost.  Still antisymmetric in
    self-play, so the shaped game stays zero-sum and policy-invariant.
    """
    planets = parse_planets(env_obs)
    if not planets:
        return 0.0
    fleets = parse_fleets(env_obs) if cfg.count_inflight else []

    prod = np.array([float(p.production) for p in planets], dtype=np.float64)
    owner = np.array([int(p.owner) for p in planets], dtype=np.int64)
    # dpp_raw (future ships) on the same currency as ships, then /norm at the end.
    dpp_raw = np.asarray(
        discounted_production_potential(prod, int(remaining_turns),
                                        gamma=cfg.gamma_eco, norm=1.0),
        dtype=np.float64,
    )
    stock = _ship_stock(planets, fleets, count_inflight=cfg.count_inflight)

    def side_value(is_mine: bool) -> float:
        mask = (owner == int(player)) if is_mine else ((owner != int(player)) & (owner != -1))
        prod_val = float(dpp_raw[mask].sum())
        ship_val = sum(v for k, v in stock.items()
                       if (k == player) == is_mine and k != -1)
        return prod_val + cfg.ship_weight * float(ship_val)

    return (side_value(True) - side_value(False)) / max(cfg.potential_norm, 1e-6)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
@dataclass
class HierTransition:
    player: int
    node_feats: np.ndarray       # (P, 15)
    edge_feats: np.ndarray       # (P, P, 4)
    global_feats: np.ndarray     # (14,)
    planet_mask: np.ndarray      # (P,)
    attack_mask: np.ndarray      # (P,)
    defend_mask: np.ndarray      # (P,)
    # action components (padded to P / fixed)
    attack_sel: np.ndarray       # (P,)
    defend_sel: np.ndarray       # (P,)
    order: np.ndarray            # (P,) long
    commit_frac: np.ndarray      # (P,)
    posture_u: np.ndarray        # (2,)
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    ret: float = 0.0


class ReworkedBuffer:
    """Flat transition store with per-player GAE."""

    def __init__(self):
        self.data: list[HierTransition] = []

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: HierTransition) -> int:
        self.data.append(tr)
        return len(self.data) - 1

    def add_reward(self, idx: int, r: float) -> None:
        self.data[idx].reward += float(r)

    def compute_gae(self, *, gamma: float = 0.99, lam: float = 0.95) -> None:
        players = sorted({tr.player for tr in self.data})
        for player in players:
            idx = [i for i, tr in enumerate(self.data) if tr.player == player]
            last_gae = 0.0
            next_value = 0.0
            for i in reversed(idx):
                tr = self.data[i]
                mask = 0.0 if tr.done else 1.0
                delta = tr.reward + gamma * next_value * mask - tr.value
                last_gae = delta + gamma * lam * mask * last_gae
                tr.advantage = last_gae
                tr.ret = last_gae + tr.value
                next_value = tr.value

    def stack(self, device=None) -> dict:
        def st(attr, dtype):
            return torch.as_tensor(np.stack([getattr(tr, attr) for tr in self.data]), dtype=dtype)

        batch = {
            "node_feats": st("node_feats", torch.float32),
            "edge_feats": st("edge_feats", torch.float32),
            "global_feats": st("global_feats", torch.float32),
            "planet_mask": st("planet_mask", torch.bool),
            "attack_mask": st("attack_mask", torch.bool),
            "defend_mask": st("defend_mask", torch.bool),
            "attack_sel": st("attack_sel", torch.float32),
            "defend_sel": st("defend_sel", torch.float32),
            "order": st("order", torch.long),
            "commit_frac": st("commit_frac", torch.float32),
            "posture_u": st("posture_u", torch.float32),
            "old_log_prob": torch.tensor([tr.log_prob for tr in self.data], dtype=torch.float32),
            "old_value": torch.tensor([tr.value for tr in self.data], dtype=torch.float32),
            "advantage": torch.tensor([tr.advantage for tr in self.data], dtype=torch.float32),
            "ret": torch.tensor([tr.ret for tr in self.data], dtype=torch.float32),
        }
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch


@dataclass
class EpisodeStats:
    length: int
    rewards: dict = field(default_factory=dict)
    winner: Optional[int] = None
    n_launches: int = 0


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def collect_episode(
    policy: ReworkedHierarchicalPolicy,
    buffer: ReworkedBuffer,
    *,
    max_steps: int = 300,
    reward_cfg: Optional[RewardConfig] = None,
    potential_fn: Optional[Callable[[object, int, int, "RewardConfig"], float]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    routing_cost: bool = True,
) -> EpisodeStats:
    """Run one 2-player self-play episode, writing one transition / player / turn.

    Reward = potential-based shaping ``gamma * Phi(s') - Phi(s)`` on the
    pluggable potential ``potential_fn`` (default :func:`economic_potential`),
    plus the terminal win/loss bonus.  ``Phi`` of a terminal state is taken as 0
    (no future), keeping the shaping telescoping and policy-invariant.
    """
    from kaggle_environments import make

    reward_cfg = reward_cfg or RewardConfig()
    potential_fn = potential_fn or economic_potential
    device = device or torch.device("cpu")
    players = [0, 1]

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    env.reset(num_agents=2)
    state = env.step([[] for _ in players])

    def obs_of(p):
        return state[p]["observation"]

    ep_rewards = {p: 0.0 for p in players}
    n_launches = 0

    for t in range(max_steps):
        env_actions = [[] for _ in players]
        pending: dict[int, int] = {}   # player -> buffer index
        phi_before: dict[int, float] = {}
        remaining_before = max(0, max_steps - t)
        remaining_after = max(0, max_steps - (t + 1))

        for p in players:
            env_obs = obs_of(p)
            av = env_obs.get("angular_velocity", 0.0) or 0.0
            obs = build_observation(env_obs, player=p, angular_velocity=av,
                                    p_max=P_MAX, f_max=F_MAX, turn=t, max_turns=max_steps)
            parsed = parse_planets(env_obs)
            phi_before[p] = potential_fn(env_obs, p, remaining_before, reward_cfg)

            batch = pack_reworked_batch([(obs, p)], device=device, p_max=P_MAX)
            st0 = batch["states"][0]
            attack_np, defend_np = _target_masks(st0, P_MAX, p)
            attack_mask = torch.from_numpy(attack_np)[None].to(device)
            defend_mask = torch.from_numpy(defend_np)[None].to(device)

            action, log_prob, value = policy.act(
                node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
                global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
                attack_mask=attack_mask, defend_mask=defend_mask,
            )
            action_row = {
                "attack_sel": action["attack_sel"][0].cpu().numpy(),
                "defend_sel": action["defend_sel"][0].cpu().numpy(),
                "order": action["order"][0].cpu().numpy(),
                "commit_frac": action["commit_frac"][0].cpu().numpy(),
                "potential_weight": float(action["potential_weight"][0]),
                "horizon": float(action["horizon"][0]),
            }
            env_action, launches = action_to_env_launches(
                obs, p, action_row, parsed, av, routing_cost=routing_cost)
            env_actions[p] = env_action
            n_launches += len(env_action)

            tr = HierTransition(
                player=p,
                node_feats=batch["node_feats"][0].cpu().numpy(),
                edge_feats=batch["edge_feats"][0].cpu().numpy(),
                global_feats=batch["global_feats"][0].cpu().numpy(),
                planet_mask=batch["planet_mask"][0].cpu().numpy(),
                attack_mask=attack_np,
                defend_mask=defend_np,
                attack_sel=action_row["attack_sel"],
                defend_sel=action_row["defend_sel"],
                order=action_row["order"].astype(np.int64),
                commit_frac=action_row["commit_frac"],
                posture_u=action["posture_u"][0].cpu().numpy(),
                log_prob=float(log_prob[0]),
                value=float(value[0]),
            )
            pending[p] = buffer.add(tr)

        state = env.step(list(env_actions))
        done = bool(getattr(env, "done", False)) or any(
            getattr(state[p], "status", "ACTIVE") == "DONE" for p in players)

        for p in players:
            r = 0.0
            if reward_cfg.enable_shaping:
                # Phi(terminal) := 0 (no future) -> shaping stays telescoping.
                phi_after = 0.0 if done else potential_fn(obs_of(p), p, remaining_after, reward_cfg)
                r += reward_cfg.shaping_weight * (reward_cfg.gamma * phi_after - phi_before[p])
            if done:
                rp = getattr(state[p], "reward", None)
                term = float(rp) if rp is not None else 0.0
                r += reward_cfg.terminal_weight * term
            buffer.data[pending[p]].reward += r
            buffer.data[pending[p]].done = done
            ep_rewards[p] += r

        if done:
            rewards = {}
            for p in players:
                rp = getattr(state[p], "reward", None)
                rewards[p] = float(rp) if rp is not None else 0.0
            winner = max(rewards, key=rewards.get) if rewards else None
            return EpisodeStats(length=t + 1, rewards=ep_rewards, winner=winner, n_launches=n_launches)

    return EpisodeStats(length=max_steps, rewards=ep_rewards, winner=None, n_launches=n_launches)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    # Per-rollout KL early stop: if an epoch's mean approx_kl exceeds this, the
    # remaining inner epochs are skipped (stabilises the Beta/Categorical heads).
    # Set to None to disable.
    target_kl: Optional[float] = 0.05


@dataclass
class PPOStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    epochs_run: int = 0


def ppo_update(
    policy: ReworkedHierarchicalPolicy,
    buffer: ReworkedBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: Optional[PPOConfig] = None,
    *,
    device: Optional[torch.device] = None,
) -> PPOStats:
    """One PPO update over the buffered hierarchical transitions."""
    cfg = cfg or PPOConfig()
    device = device or torch.device("cpu")
    if len(buffer) == 0:
        return PPOStats()

    data = buffer.stack(device=device)
    N = data["node_feats"].shape[0]
    adv = data["advantage"]
    if cfg.normalize_adv and N > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = PPOStats()
    n_batches = 0
    for _ in range(cfg.epochs):
        epoch_kl = 0.0
        epoch_batches = 0
        perm = torch.randperm(N, device=device)
        for start in range(0, N, cfg.minibatch_size):
            mb = perm[start:start + cfg.minibatch_size]
            action = {
                "attack_sel": data["attack_sel"][mb],
                "defend_sel": data["defend_sel"][mb],
                "order": data["order"][mb],
                "commit_frac": data["commit_frac"][mb],
                "posture_u": data["posture_u"][mb],
            }
            ev = policy.evaluate_actions(
                node_feats=data["node_feats"][mb], edge_feats=data["edge_feats"][mb],
                global_feats=data["global_feats"][mb], planet_mask=data["planet_mask"][mb],
                attack_mask=data["attack_mask"][mb], defend_mask=data["defend_mask"][mb],
                action=action,
            )
            new_logp = ev["log_prob"]
            ratio = torch.exp(new_logp - data["old_log_prob"][mb])
            a = adv[mb]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(ev["value"], data["ret"][mb])
            entropy = ev["entropy"].mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                batch_kl = float((data["old_log_prob"][mb] - new_logp).mean())
                stats.policy_loss += float(policy_loss)
                stats.value_loss += float(value_loss)
                stats.entropy += float(entropy)
                stats.approx_kl += batch_kl
            epoch_kl += batch_kl
            epoch_batches += 1
            n_batches += 1

        stats.epochs_run += 1
        if cfg.target_kl is not None and epoch_batches:
            if (epoch_kl / epoch_batches) > cfg.target_kl:
                break  # KL budget exceeded -> stop early to avoid policy collapse

    if n_batches:
        stats.policy_loss /= n_batches
        stats.value_loss /= n_batches
        stats.entropy /= n_batches
        stats.approx_kl /= n_batches
    return stats


__all__ = [
    "P_MAX", "F_MAX",
    "action_to_env_launches",
    "RewardConfig", "economic_potential",
    "HierTransition", "ReworkedBuffer", "EpisodeStats",
    "collect_episode", "PPOConfig", "PPOStats", "ppo_update",
]
