"""Reworked node features for the Orbit Wars graph policy (clean rewrite).

This module is a **fresh, self-contained** feature implementation that does
NOT import from or depend on the legacy metric code
(``model_graph_pyg._graph_node_metrics``, ``rl/held_production.py``,
``rl/rewards.py``).  It is meant to grow into a redesigned node-feature
pipeline; each feature is an independent, vectorized pure function with
both NumPy and (optional) PyTorch support so it can be used in the
observation builder, the reward path, or offline analysis without
coupling to the existing channels.

Features implemented so far
---------------------------
1. ``discounted_production_potential`` — a planet's intrinsic economic
   value: the discounted sum of its future per-turn production over the
   remaining game horizon.  Owner-independent (pure economic worth) and
   decreasing as the game nears its end.
2. ``closeness_centrality`` — a planet's distance-based closeness
   centrality on the fully-connected planet graph (edge weight =
   normalised pairwise distance).  Owner- and production-independent
   pure structural signal; the GNN learns to combine it with production
   into a logistics / force-projection notion rather than us
   hand-engineering the product.
3. ``standard_node_features`` — the raw observation-derived block:
   ownership (4 slots: self + 3 enemy + neutral), log ship count, raw
   production, radius, rotating flag, comet flag.  These are deliberately
   kept *raw* so the GNN can combine them freely with the derived
   metrics above.
4. ``fleet_speed`` — the game's exact fleet-speed formula (speed grows
   with ship count), vectorized for NumPy / torch.
5. ``potential_threat`` — distance-AND-speed-weighted *potential* enemy
   garrison pressure on each planet.  Larger enemy garrisons travel faster,
   so their effective distance (= ETA in turns) shrinks; threat therefore
   scales both with enemy ship mass and with arrival speed.
6. ``potential_support`` — the friendly mirror of ``potential_threat``:
   how much reinforcement the agent's *other* planets could launch.
7. ``friendly_compactness`` — per-planet compactness of the agent's
   territory (closeness restricted to owned planets): high = deep interior,
   low = exposed frontier.
8. ``edge_features`` — directed per-edge features for *in-flight* fleets:
   the ETA-weighted friendly / enemy ship flow currently travelling along
   each edge (plus static distance / sun-crossing channels).  This is the
   "actual / in transit" complement to the "potential / garrison" node
   metrics above.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:  # torch is optional; NumPy path always works.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

ArrayLike = Union[np.ndarray, "torch.Tensor"]
Number = Union[int, float]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Per-turn economic discount.  ``gamma < 1`` makes near-future production
# worth more than far-future production; the horizon factor saturates at
# ``1 / (1 - gamma)`` for large remaining horizons.
DEFAULT_GAMMA: float = 0.99
# Normalisation constant so the output sits roughly in ``[0, 1]`` for
# typical production rates.  With ``gamma = 0.99`` the horizon factor
# saturates at 100, so a planet producing ~5/turn saturates near
# ``5 * 100 / 500 = 1.0``.
DEFAULT_NORM: float = 500.0

# ---- standard_node_features normalisation / layout ------------------------
# Max enemy one-hot slots (mirrors the legacy obs_builder convention so the
# enemy slot assignment is consistent across the codebase).
MAX_ENEMY_OWNER_SLOTS: int = 3
SHIP_LOG_BASE: float = 1000.0    # log_ships = log(s + 1) / log(SHIP_LOG_BASE)
PRODUCTION_NORM: float = 5.0     # production / PRODUCTION_NORM
RADIUS_NORM: float = 5.0         # radius / RADIUS_NORM

# ---- fleet speed / enemy threat -------------------------------------------
# Exact mirror of ``feature_extraction.fleet_speed`` (and orbit_wars.py):
#     speed = 1 + (MAX_SHIP_SPEED - 1) * (ln(ships) / ln(SPEED_LOG_BASE)) ^ 1.5
# Speed grows with ship count, so a bigger fleet covers the same distance in
# fewer turns -> its *effective* distance shrinks.
DEFAULT_MAX_SHIP_SPEED: float = 6.0
SPEED_LOG_BASE: float = 1000.0
SPEED_EXPONENT: float = 1.5
# Board edge length used to turn normalised edge distances back into raw
# board units so ETA comes out in real turns (eta = raw_dist / speed).
DEFAULT_BOARD_SIZE: float = 100.0
# Floor (in turns) added to ETA so a zero-distance / co-located enemy does not
# blow the gravity weight up to infinity; ~1 turn is the minimum sensible ETA.
DEFAULT_THREAT_ETA_EPS: float = 1.0

# ---- edge_features (in-flight fleet flux) ---------------------------------
# In-flight fleets are weighted by how imminent their arrival is.  ``exp_decay``
# (weight = ships * exp(-eta / tau)) is the default and matches the
# "exponential threat horizon" we use elsewhere: a fleet about to land counts
# almost at full mass, a far-off one is heavily discounted.
FLUX_MODE_RAW = "raw"
FLUX_MODE_LINEAR = "linear"
FLUX_MODE_EXP_DECAY = "exp_decay"
DEFAULT_FLUX_MODE: str = FLUX_MODE_EXP_DECAY
DEFAULT_FLUX_TAU: float = 10.0        # exp_decay time constant (turns)
DEFAULT_FLUX_ETA_NORM: float = 50.0   # linear-mode ETA normaliser (turns)
DEFAULT_FLUX_SHIP_NORM: float = 50.0  # ship-count normaliser for flux channels

# Channel layout of ``edge_features`` output (directed edge (i -> j)).
EDGE_IDX_DISTANCE = 0       # normalised geometric distance d(i, j)
EDGE_IDX_CROSSES_SUN = 1    # 1 if the straight i->j path is blocked by the sun
EDGE_IDX_FRIENDLY_FLUX = 2  # ETA-weighted friendly ship flow i -> j
EDGE_IDX_ENEMY_FLUX = 3     # ETA-weighted enemy ship flow i -> j
REWORKED_EDGE_FEATURE_DIM = 4

# Channel layout of ``standard_node_features`` output (10 columns).
STD_IDX_IS_SELF = 0
STD_IDX_ENEMY_0 = 1
STD_IDX_ENEMY_1 = 2
STD_IDX_ENEMY_2 = 3
STD_IDX_IS_NEUTRAL = 4
STD_IDX_LOG_SHIPS = 5
STD_IDX_PRODUCTION = 6
STD_IDX_RADIUS = 7
STD_IDX_IS_ROTATING = 8
STD_IDX_IS_COMET = 9
STANDARD_NODE_FEATURE_DIM = 10


def geometric_horizon_factor(
    remaining_turns: Union[int, ArrayLike],
    gamma: float = DEFAULT_GAMMA,
    *,
    eps: float = 1e-6,
) -> ArrayLike:
    r"""Return the discounted horizon factor.

    .. math::

        H(R) = \sum_{k=0}^{R-1} \gamma^k = \frac{1 - \gamma^R}{1 - \gamma}

    Interpretation: the number of "effective" future turns once each turn
    is discounted by ``gamma``.  ``H(0) = 0`` (no turns left), and as
    ``R \to \infty`` it saturates at ``1 / (1 - gamma)``.

    Parameters
    ----------
    remaining_turns:
        Scalar ``R = max_turns - turn`` (clamped at 0).  May also be an
        array (NumPy) for per-planet horizons, though the common case is a
        single scalar shared by all planets in a state.
    gamma:
        Per-turn discount in ``(0, 1]``.  ``gamma = 1`` degenerates to the
        linear case ``H(R) = R``.

    Returns
    -------
    float | np.ndarray
        Same type/shape as ``remaining_turns`` (scalar in → float out).
    """
    g = float(gamma)
    if isinstance(remaining_turns, np.ndarray):
        r = np.clip(remaining_turns.astype(np.float64), 0.0, None)
        if abs(g - 1.0) < eps:
            return r.astype(np.float32)
        return ((1.0 - np.power(g, r)) / (1.0 - g)).astype(np.float32)

    if torch is not None and isinstance(remaining_turns, torch.Tensor):
        r = remaining_turns.clamp_min(0).to(dtype=torch.float32)
        if abs(g - 1.0) < eps:
            return r
        return (1.0 - torch.pow(torch.tensor(g), r)) / (1.0 - g)

    # Scalar path.
    r = max(0.0, float(remaining_turns))
    if abs(g - 1.0) < eps:
        return float(r)
    return float((1.0 - g ** r) / (1.0 - g))


def discounted_production_potential(
    production: ArrayLike,
    remaining_turns: Union[int, ArrayLike],
    *,
    gamma: float = DEFAULT_GAMMA,
    norm: float = DEFAULT_NORM,
) -> ArrayLike:
    r"""Intrinsic economic value of a planet over the remaining horizon.

    .. math::

        V_g = \frac{\text{prod}_g \cdot H(R)}{\text{norm}},
        \qquad H(R) = \frac{1 - \gamma^R}{1 - \gamma}

    This is the **discounted total production** the planet will yield from
    now until the end of the game.  Key properties:

    * **Owner-independent.** It reflects the worth of the planet itself,
      not who holds it; ownership / survival overlays are deliberately
      kept as separate features so this channel stays a clean economic
      signal.
    * **Decreases as the game ends.** As ``R \to 0`` the factor ``H(R)``
      (and thus ``V``) goes to 0 — capturing a planet with few turns left
      is nearly worthless.
    * **Discounted, not linear.** Near-future turns dominate; very long
      horizons saturate at ``prod / (1 - gamma) / norm``.

    Parameters
    ----------
    production:
        Per-turn production rate per planet (NumPy array or torch tensor).
        Real units (ships/turn), not pre-normalised.
    remaining_turns:
        ``R = max_turns - turn``.  Scalar (shared by all planets) or array.
    gamma, norm:
        See :data:`DEFAULT_GAMMA` / :data:`DEFAULT_NORM`.

    Returns
    -------
    Same array type as ``production``; non-negative.
    """
    factor = geometric_horizon_factor(remaining_turns, gamma)
    scale = max(float(norm), 1e-6)

    if isinstance(production, np.ndarray):
        prod = np.clip(production.astype(np.float64), 0.0, None)
        f = np.asarray(factor, dtype=np.float64)
        return (prod * f / scale).astype(np.float32)

    if torch is not None and isinstance(production, torch.Tensor):
        prod = production.clamp_min(0.0)
        if not isinstance(factor, torch.Tensor):
            factor = torch.as_tensor(float(factor), dtype=prod.dtype, device=prod.device)
        return prod * factor / scale

    # Scalar path.
    prod = max(0.0, float(production))
    return float(prod * float(factor) / scale)


def closeness_centrality(
    distance_matrix: ArrayLike,
    *,
    valid_mask: "ArrayLike | None" = None,
    eps: float = 1e-6,
) -> ArrayLike:
    r"""Distance-based closeness centrality on the fully-connected graph.

    .. math::

        C(g) = \frac{N_g - 1}{\sum_{h \neq g} d(g, h)}

    where ``d(g, h)`` is the (normalised) pairwise distance edge weight and
    ``N_g`` is the number of valid planets reachable from ``g``.  A planet
    with a short *average* distance to all others is central → high value.

    Pure structural signal:

    * **Owner-independent** and **production-independent** — it only depends
      on the constellation geometry, so the GNN is free to combine it with
      production / ownership features as it sees fit.
    * **Diagonal excluded** (a planet's distance to itself is not counted).
    * Padded / invalid planets are excluded from every sum via
      ``valid_mask`` and receive a centrality of 0.

    Parameters
    ----------
    distance_matrix:
        ``(n, n)`` or batched ``(B, n, n)`` pairwise distances.  Expected to
        already be normalised (e.g. ``edge_bias[..., 0] = ‖Δpos‖ / BOARD_SIZE``)
        but any consistent positive scale works; the output scale follows.
    valid_mask:
        Optional boolean ``(n,)`` / ``(B, n)`` marking real planets.  When
        given, invalid rows/cols are dropped from the sums and invalid
        planets output 0.
    eps:
        Guard added to the distance sum to avoid division by zero when a
        planet has no valid neighbours.

    Returns
    -------
    ``(n,)`` or ``(B, n)`` closeness scores (same array type as input).
    Note the value is *not* bounded to ``[0, 1]``; downstream LayerNorm in
    the encoder handles scaling.  Typical range with board-normalised
    distances is roughly ``[0.7, 3]``.
    """
    # ---- NumPy path -------------------------------------------------------
    if isinstance(distance_matrix, np.ndarray):
        d = distance_matrix.astype(np.float64)
        batched = d.ndim == 3
        if not batched:
            d = d[None]
        B, n, _ = d.shape
        pair_valid = np.ones((B, n, n), dtype=np.float64)
        node_valid = np.ones((B, n), dtype=np.float64)
        if valid_mask is not None:
            m = np.asarray(valid_mask, dtype=bool)
            if m.ndim == 1:
                m = m[None]
            node_valid = m.astype(np.float64)
            pair_valid = (m[:, :, None] & m[:, None, :]).astype(np.float64)
        # Exclude the diagonal from both the distance sum and the count.
        eye = np.eye(n, dtype=np.float64)[None]
        pair_valid = pair_valid * (1.0 - eye)
        dist_sum = (d * pair_valid).sum(axis=2)
        count = pair_valid.sum(axis=2)
        out = count / (dist_sum + eps)
        out = out * node_valid
        out = out.astype(np.float32)
        return out[0] if not batched else out

    if torch is not None and isinstance(distance_matrix, torch.Tensor):
        d = distance_matrix.to(dtype=torch.float32)
        batched = d.dim() == 3
        if not batched:
            d = d.unsqueeze(0)
        B, n, _ = d.shape
        if valid_mask is not None:
            m = valid_mask.to(dtype=torch.bool)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            node_valid = m.to(dtype=d.dtype)
            pair_valid = (m[:, :, None] & m[:, None, :]).to(dtype=d.dtype)
        else:
            node_valid = torch.ones((B, n), dtype=d.dtype, device=d.device)
            pair_valid = torch.ones((B, n, n), dtype=d.dtype, device=d.device)
        eye = torch.eye(n, dtype=d.dtype, device=d.device).unsqueeze(0)
        pair_valid = pair_valid * (1.0 - eye)
        dist_sum = (d * pair_valid).sum(dim=2)
        count = pair_valid.sum(dim=2)
        out = (count / (dist_sum + eps)) * node_valid
        return out.squeeze(0) if not batched else out

    raise TypeError(
        "distance_matrix must be a numpy.ndarray or torch.Tensor; "
        f"got {type(distance_matrix)!r}"
    )


def standard_node_features(
    owner: np.ndarray,
    ships: np.ndarray,
    *,
    player: int,
    production: np.ndarray,
    radius: np.ndarray,
    is_rotating: np.ndarray,
    is_comet: np.ndarray,
    ship_log_base: float = SHIP_LOG_BASE,
    production_norm: float = PRODUCTION_NORM,
    radius_norm: float = RADIUS_NORM,
) -> np.ndarray:
    """Assemble the raw observation-derived node feature block.

    Returns an ``(n, STANDARD_NODE_FEATURE_DIM)`` ``float32`` matrix whose
    columns follow the ``STD_IDX_*`` layout:

        [is_self, enemy_0, enemy_1, enemy_2, is_neutral,
         log_ships, production, radius, is_rotating, is_comet]

    These channels are intentionally **raw** (no derived aggregation): the
    GNN combines them with the structural / economic metrics
    (:func:`closeness_centrality`, :func:`discounted_production_potential`,
    ...) itself.

    Ownership encoding mirrors the legacy ``obs_builder`` convention so the
    three enemy slots are assigned consistently:

        slot = (owner - player - 1) mod (MAX_ENEMY_OWNER_SLOTS + 1)

    Parameters
    ----------
    owner:
        ``(n,)`` int array; ``-1`` = neutral, otherwise the owning player id.
    ships, production, radius:
        ``(n,)`` float arrays in real units.
    is_rotating, is_comet:
        ``(n,)`` bool / 0-1 arrays.
    player:
        The agent's player id (perspective for ``is_self`` / enemy slots).

    All inputs must share the same leading length ``n``.
    """
    owner = np.asarray(owner).reshape(-1)
    n = owner.shape[0]
    ships = np.asarray(ships, dtype=np.float64).reshape(-1)
    production = np.asarray(production, dtype=np.float64).reshape(-1)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    is_rotating = np.asarray(is_rotating, dtype=np.float64).reshape(-1)
    is_comet = np.asarray(is_comet, dtype=np.float64).reshape(-1)

    out = np.zeros((n, STANDARD_NODE_FEATURE_DIM), dtype=np.float32)

    is_self = owner == int(player)
    is_neutral = owner == -1
    is_enemy = (~is_self) & (~is_neutral)
    # Consistent enemy-slot assignment (same as obs_builder).
    slot = np.mod(owner - int(player) - 1, MAX_ENEMY_OWNER_SLOTS + 1)

    out[:, STD_IDX_IS_SELF] = is_self.astype(np.float32)
    for j in range(MAX_ENEMY_OWNER_SLOTS):
        out[:, STD_IDX_ENEMY_0 + j] = (is_enemy & (slot == j)).astype(np.float32)
    out[:, STD_IDX_IS_NEUTRAL] = is_neutral.astype(np.float32)

    log_base = max(float(ship_log_base), 1.0 + 1e-6)
    out[:, STD_IDX_LOG_SHIPS] = (
        np.log(np.clip(ships, 0.0, None) + 1.0) / np.log(log_base)
    ).astype(np.float32)
    out[:, STD_IDX_PRODUCTION] = (
        np.clip(production, 0.0, None) / max(float(production_norm), 1e-6)
    ).astype(np.float32)
    out[:, STD_IDX_RADIUS] = (
        np.clip(radius, 0.0, None) / max(float(radius_norm), 1e-6)
    ).astype(np.float32)
    out[:, STD_IDX_IS_ROTATING] = (is_rotating > 0.5).astype(np.float32)
    out[:, STD_IDX_IS_COMET] = (is_comet > 0.5).astype(np.float32)
    return out


def fleet_speed(
    ships: ArrayLike,
    *,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> ArrayLike:
    r"""Vectorized fleet speed (board units per turn) as a function of size.

    Exact mirror of the environment's ``feature_extraction.fleet_speed``:

    .. math::

        v(s) = 1 + (v_{\max} - 1)
               \left(\frac{\ln s}{\ln 1000}\right)^{1.5}

    with the same edge cases as the engine:

    * ``s <= 0`` → ``0``   (no fleet)
    * ``s == 1`` → ``1``   (avoids ``ln 1`` degeneracy)

    Speed is monotone increasing in ``s`` and saturates at ``v_max`` near
    ``s = 1000``.  A larger fleet therefore reaches a target in fewer turns,
    which is exactly what makes big enemy garrisons disproportionately
    threatening.

    Parameters
    ----------
    ships:
        Scalar, NumPy array or torch tensor of ship counts.
    max_speed:
        Speed ceiling ``v_max`` (engine default 6.0).

    Returns
    -------
    Same type/shape as ``ships`` (scalar in → float out).
    """
    vmax = float(max_speed)
    log_base = np.log(SPEED_LOG_BASE)

    if isinstance(ships, np.ndarray):
        s = ships.astype(np.float64)
        safe = np.clip(s, 1.0, None)
        speed = 1.0 + (vmax - 1.0) * np.power(np.log(safe) / log_base, SPEED_EXPONENT)
        speed = np.where(s <= 1.0, 1.0, speed)
        speed = np.where(s <= 0.0, 0.0, speed)
        return speed.astype(np.float32)

    if torch is not None and isinstance(ships, torch.Tensor):
        s = ships.to(dtype=torch.float32)
        safe = s.clamp_min(1.0)
        ratio = (torch.log(safe) / float(log_base)).clamp_min(0.0)
        speed = 1.0 + (vmax - 1.0) * ratio.pow(SPEED_EXPONENT)
        speed = torch.where(s <= 1.0, torch.ones_like(speed), speed)
        speed = torch.where(s <= 0.0, torch.zeros_like(speed), speed)
        return speed

    # Scalar path.
    s = float(ships)
    if s <= 0.0:
        return 0.0
    if s <= 1.0:
        return 1.0
    return float(1.0 + (vmax - 1.0) * (np.log(s) / log_base) ** SPEED_EXPONENT)


def _speed_weighted_pressure(
    distance_matrix: np.ndarray,
    ships: np.ndarray,
    source_mask: np.ndarray,
    *,
    board_size: float,
    max_speed: float,
    eta_eps: float,
    norm: float,
) -> np.ndarray:
    r"""Shared core for speed-aware gravity pressure (threat / support).

    For every ordered pair ``(h -> g)`` the *effective* distance is the travel
    time of a fleet leaving source ``h`` at full strength:

    .. math::

        \text{eta}(h \to g) = \frac{d(h, g)}{v(s_h)}, \qquad
        P(g) = \sum_{h\,\in\,\text{src}} \frac{s_h}{\text{eta}(h \to g) + \varepsilon}

    Only sources with ``source_mask[h] = True`` contribute, and the diagonal
    (``h == g``) is always excluded.  ``potential_threat`` and
    ``potential_support`` are this function with an enemy / friendly source
    mask respectively.

    Shapes: ``distance_matrix`` is ``(n, n)`` or ``(B, n, n)``; ``ships`` and
    ``source_mask`` are ``(n,)`` or ``(B, n)``.  Returns ``(n,)`` / ``(B, n)``.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    batched = d.ndim == 3
    if not batched:
        d = d[None]
    B, n, _ = d.shape

    ships_a = np.asarray(ships, dtype=np.float64).reshape(B, n)
    mask_a = np.asarray(source_mask).reshape(B, n).astype(np.float64)

    raw_d = d * float(board_size)
    # Source speed broadcasts across columns: eta[b, h, g] = raw_d / v(s_h).
    speed = np.asarray(fleet_speed(ships_a, max_speed=max_speed), dtype=np.float64)
    speed_safe = np.clip(speed, 1e-6, None)
    eta = raw_d / speed_safe[:, :, None]

    src_mass = mask_a * np.clip(ships_a, 0.0, None)                          # (B, n)
    weight = 1.0 / (eta + float(eta_eps))                                   # (B, n, n)
    eye = np.eye(n, dtype=np.float64)[None]
    weight = weight * (1.0 - eye)                                           # drop h == g

    out = (src_mass[:, :, None] * weight).sum(axis=1)                       # sum over h
    out = (out / max(float(norm), 1e-6)).astype(np.float32)
    return out[0] if not batched else out


def potential_threat(
    distance_matrix: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    *,
    player: int,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    eta_eps: float = DEFAULT_THREAT_ETA_EPS,
    norm: float = 1.0,
) -> np.ndarray:
    r"""Speed-aware *potential* enemy garrison pressure on every planet.

    "Potential" because it scores the force enemy planets *could* launch from
    their current garrisons — the in-flight, already-launched fleets are a
    separate, complementary signal carried on the edges (see
    :func:`edge_features`).

    For each ordered pair ``(h -> g)`` the *effective* distance is the travel
    time of an enemy fleet leaving source ``h`` at full strength:

    .. math::

        \text{eta}(h \to g) = \frac{d(h, g)}{v(s_h)}, \qquad
        \text{threat}(g) = \sum_{h \in \text{enemy}}
            \frac{s_h}{\text{eta}(h \to g) + \varepsilon}

    Substituting the speed law gives
    ``threat(g) = Σ_h s_h · v(s_h) / (d + ε·v(s_h))``, i.e. each enemy planet
    contributes its **ship mass times its arrival speed** discounted by raw
    distance.  This bakes in the two effects you asked for at once:

    * **more ships → more force** (the ``s_h`` factor), and
    * **more ships → faster fleet → shorter ETA** (the ``v(s_h)`` factor),

    so a heavily-garrisoned enemy planet at a fixed distance is far more
    threatening than a lightly-garrisoned one — both because it can throw more
    ships and because those ships arrive sooner.

    This is also strictly more faithful than the legacy intercept kernel,
    which fixes the fleet size at a reference (≈50 ships); here the speed is
    derived from each source's *actual* garrison.

    Parameters
    ----------
    distance_matrix:
        ``(n, n)`` or ``(B, n, n)`` pairwise distances.  Assumed **normalised**
        (e.g. ``edge_bias[..., 0] = ‖Δpos‖ / BOARD_SIZE``); ``board_size``
        converts them back to raw board units so ETA is expressed in turns.
        Pass ``board_size=1.0`` if the matrix already holds raw distances.
    owner:
        ``(n,)`` / ``(B, n)`` int array; ``-1`` = neutral, else owning player.
    ships:
        ``(n,)`` / ``(B, n)`` garrison sizes (real units) — used **both** for
        the force term and (via :func:`fleet_speed`) for the speed term.
    player:
        The agent's id; enemies are ``owner != player`` and ``owner != -1``.
    board_size, max_speed, eta_eps, norm:
        See module-level defaults.  ``norm`` only rescales the output.

    Returns
    -------
    ``(n,)`` / ``(B, n)`` non-negative threat per *target* planet ``g``
    (sum over enemy sources ``h``).  Diagonal (``h == g``) is excluded.
    """
    owner_a = np.asarray(owner)
    is_enemy = (owner_a != int(player)) & (owner_a != -1)
    return _speed_weighted_pressure(
        distance_matrix,
        ships,
        is_enemy,
        board_size=board_size,
        max_speed=max_speed,
        eta_eps=eta_eps,
        norm=norm,
    )


def potential_support(
    distance_matrix: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    *,
    player: int,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    eta_eps: float = DEFAULT_THREAT_ETA_EPS,
    norm: float = 1.0,
) -> np.ndarray:
    r"""Speed-aware *potential* friendly reinforcement capacity per planet.

    "Potential" because it scores reinforcement the agent's other planets
    *could* launch from their current garrisons; fleets already in transit are
    the complementary edge signal (see :func:`edge_features`).

    The mirror image of :func:`potential_threat`: if planet ``g`` comes under
    attack, how quickly and how strongly can the agent's **other** planets
    reinforce it?  Same speed-aware gravity kernel, friendly source mask:

    .. math::

        \text{eta}(h \to g) = \frac{d(h, g)}{v(s_h)}, \qquad
        \text{support}(g) = \sum_{\substack{h \in \text{self} \\ h \neq g}}
            \frac{s_h}{\text{eta}(h \to g) + \varepsilon}

    Notes
    -----
    * Sources are the agent's planets (``owner == player``); the diagonal is
      excluded, so ``g``'s **own** garrison is *not* counted here — that is the
      immediate on-site defence, already captured by the ``log_ships`` channel
      in :func:`standard_node_features`.  This metric is purely the
      *reinforcement* that can arrive from elsewhere.
    * Same double effect as the threat metric: a large friendly garrison both
      carries more ships and arrives faster (higher ``v(s_h)`` → smaller ETA).
    * Pairing this with :func:`potential_threat` lets the GNN reason about net
      defensibility (support vs. threat) without us hand-coding the difference.

    Parameters mirror :func:`potential_threat`.

    Returns
    -------
    ``(n,)`` / ``(B, n)`` non-negative reinforcement capacity per planet ``g``
    (sum over friendly sources ``h != g``).
    """
    owner_a = np.asarray(owner)
    is_friendly = owner_a == int(player)
    return _speed_weighted_pressure(
        distance_matrix,
        ships,
        is_friendly,
        board_size=board_size,
        max_speed=max_speed,
        eta_eps=eta_eps,
        norm=norm,
    )


def friendly_compactness(
    distance_matrix: ArrayLike,
    owner: ArrayLike,
    *,
    player: int,
    valid_mask: "ArrayLike | None" = None,
    eps: float = 1e-6,
) -> ArrayLike:
    r"""Per-planet compactness of the agent's territory (interior vs frontier).

    Distance-based closeness centrality restricted to the agent's **own**
    planets — i.e. :func:`closeness_centrality` evaluated over the friendly
    sub-graph:

    .. math::

        \text{compact}(g) = \frac{N_{\text{mine}} - 1}
                                  {\sum_{\substack{h \in \text{mine} \\ h \neq g}} d(g, h)}

    Interpretation:

    * **High** → ``g`` is surrounded by the agent's other planets: a deep
      *interior* planet with short internal reinforcement lines, easy to hold.
    * **Low** → ``g`` is far from friendly planets: an exposed *frontier* /
      flank planet.
    * A lone owned planet (no other friendly planets) gets 0 — maximally
      "frontier".

    Only the agent's planets receive a value; enemy / neutral planets output 0
    (compactness of *my* territory is meaningless for them, and the ownership
    one-hot lets the GNN gate it).  Pairs nicely with :func:`potential_threat`
    so the network can flag "exposed frontier planet under threat".

    Parameters
    ----------
    distance_matrix:
        ``(n, n)`` or ``(B, n, n)`` normalised pairwise distances (same input
        convention as :func:`closeness_centrality`).
    owner:
        ``(n,)`` / ``(B, n)`` owner ids (``-1`` = neutral).  Must match the
        array backend (NumPy / torch) of ``distance_matrix``.
    player:
        The agent's id; friendly planets are ``owner == player``.
    valid_mask:
        Optional ``(n,)`` / ``(B, n)`` padding mask; combined with the friendly
        mask so padded slots never contribute.
    eps:
        Guard for the distance-sum denominator (see :func:`closeness_centrality`).

    Returns
    -------
    ``(n,)`` / ``(B, n)`` compactness scores (same backend as input); 0 for
    non-owned / padded planets.
    """
    if isinstance(distance_matrix, np.ndarray):
        is_friendly = np.asarray(owner) == int(player)
        if valid_mask is not None:
            is_friendly = is_friendly & np.asarray(valid_mask, dtype=bool)
        return closeness_centrality(distance_matrix, valid_mask=is_friendly, eps=eps)

    if torch is not None and isinstance(distance_matrix, torch.Tensor):
        is_friendly = owner == int(player)
        if valid_mask is not None:
            is_friendly = is_friendly & valid_mask.to(dtype=torch.bool)
        return closeness_centrality(distance_matrix, valid_mask=is_friendly, eps=eps)

    raise TypeError(
        "distance_matrix must be a numpy.ndarray or torch.Tensor; "
        f"got {type(distance_matrix)!r}"
    )


def _flux_weight(
    ships: np.ndarray,
    eta: np.ndarray,
    *,
    mode: str,
    tau: float,
    eta_norm: float,
) -> np.ndarray:
    """Per-fleet imminence weight (see :data:`DEFAULT_FLUX_MODE`).

    * ``raw``       → ``ships``
    * ``linear``    → ``ships * max(0, 1 - eta / eta_norm)``
    * ``exp_decay`` → ``ships * exp(-eta / tau)``
    """
    s = np.clip(np.asarray(ships, dtype=np.float64), 0.0, None)
    e = np.clip(np.asarray(eta, dtype=np.float64), 0.0, None)
    if mode == FLUX_MODE_RAW:
        return s
    if mode == FLUX_MODE_LINEAR:
        return s * np.maximum(0.0, 1.0 - e / max(float(eta_norm), 1e-6))
    if mode == FLUX_MODE_EXP_DECAY:
        return s * np.exp(-e / max(float(tau), 1e-6))
    raise ValueError(f"unknown flux mode: {mode!r}")


def inflight_flux_matrices(
    n: int,
    fleet_src: np.ndarray,
    fleet_tgt: np.ndarray,
    fleet_eta: np.ndarray,
    fleet_ships: np.ndarray,
    fleet_is_self: np.ndarray,
    fleet_is_enemy: np.ndarray,
    *,
    fleet_mask: "np.ndarray | None" = None,
    mode: str = DEFAULT_FLUX_MODE,
    tau: float = DEFAULT_FLUX_TAU,
    eta_norm: float = DEFAULT_FLUX_ETA_NORM,
) -> "tuple[np.ndarray, np.ndarray]":
    r"""Directed in-flight ship-flow matrices, one per owner class.

    Self-contained re-implementation (does not import ``rl/flux.py``).  Each
    in-flight fleet ``k`` travels from source planet ``fleet_src[k]`` to target
    ``fleet_tgt[k]`` carrying ``fleet_ships[k]`` ships, arriving in
    ``fleet_eta[k]`` turns.  It contributes its imminence weight (see
    :func:`_flux_weight`) to ``M[src, tgt]`` of its owner's matrix.

    Returns
    -------
    (friendly, enemy):
        Two ``(n, n)`` ``float32`` matrices with ``M[i, j]`` = weighted ship
        flow from ``i`` to ``j``.  Fleets with unknown/out-of-range
        ``src``/``tgt`` or masked-out slots are skipped.

    Notes
    -----
    The per-fleet ETA already reflects that fleet's *actual* speed (the
    navigation layer derives it via :func:`fleet_speed` from the fleet's ship
    count), so the "bigger fleet → shorter ETA" effect is baked in for free
    here — no need to re-apply :func:`fleet_speed` on the edge side.
    """
    friendly = np.zeros((n, n), dtype=np.float32)
    enemy = np.zeros((n, n), dtype=np.float32)
    if n <= 0:
        return friendly, enemy

    src = np.asarray(fleet_src, dtype=np.int64).reshape(-1)
    tgt = np.asarray(fleet_tgt, dtype=np.int64).reshape(-1)
    eta = np.asarray(fleet_eta, dtype=np.float64).reshape(-1)
    ships = np.asarray(fleet_ships, dtype=np.float64).reshape(-1)
    is_self = np.asarray(fleet_is_self, dtype=bool).reshape(-1)
    is_enemy = np.asarray(fleet_is_enemy, dtype=bool).reshape(-1)

    if fleet_mask is None:
        mask = np.ones(src.shape, dtype=bool)
    else:
        mask = np.asarray(fleet_mask, dtype=bool).reshape(-1)

    valid = mask & (src >= 0) & (tgt >= 0) & (src < n) & (tgt < n)
    if not valid.any():
        return friendly, enemy

    w = _flux_weight(ships[valid], eta[valid], mode=mode, tau=tau, eta_norm=eta_norm)
    s = src[valid]
    t = tgt[valid]
    sf = is_self[valid]
    en = is_enemy[valid]

    # np.add.at accumulates correctly when several fleets share one (i, j) edge.
    if sf.any():
        np.add.at(friendly, (s[sf], t[sf]), w[sf].astype(np.float32))
    if en.any():
        np.add.at(enemy, (s[en], t[en]), w[en].astype(np.float32))
    return friendly, enemy


def edge_features(
    distance_matrix: np.ndarray,
    *,
    fleet_src: np.ndarray,
    fleet_tgt: np.ndarray,
    fleet_eta: np.ndarray,
    fleet_ships: np.ndarray,
    fleet_is_self: np.ndarray,
    fleet_is_enemy: np.ndarray,
    fleet_mask: "np.ndarray | None" = None,
    crosses_sun: "np.ndarray | None" = None,
    mode: str = DEFAULT_FLUX_MODE,
    tau: float = DEFAULT_FLUX_TAU,
    eta_norm: float = DEFAULT_FLUX_ETA_NORM,
    ship_norm: float = DEFAULT_FLUX_SHIP_NORM,
) -> np.ndarray:
    r"""Directed per-edge feature stack ``(n, n, REWORKED_EDGE_FEATURE_DIM)``.

    Entry ``[i, j]`` describes the directed edge ``i -> j`` with channels
    (``EDGE_IDX_*``):

        [ distance(i, j),        # static geometry
          crosses_sun(i, j),     # static: straight path blocked by the sun
          friendly_flux(i->j),   # ETA-weighted friendly ships in transit
          enemy_flux(i->j) ]     # ETA-weighted enemy ships in transit

    This is the **in-flight / actual** counterpart of the garrison-based
    **potential** node metrics (:func:`potential_threat` /
    :func:`potential_support`): a fleet literally occupies the edge between its
    source and target, so its mass belongs on that directed edge.  Pairing the
    edge flux with the target planet's ownership lets the GNN distinguish an
    *incoming attack* (enemy flux into my planet) from *reinforcement in
    transit* (friendly flux into my planet) without any hand-coding.

    Parameters
    ----------
    distance_matrix:
        ``(n, n)`` normalised pairwise distance (e.g. ``edge_bias[..., 0]``).
    fleet_src, fleet_tgt, fleet_eta, fleet_ships, fleet_is_self, fleet_is_enemy:
        Per-fleet routing arrays (``(F,)`` each) as recorded on the
        observation.  See :func:`inflight_flux_matrices`.
    fleet_mask:
        Optional ``(F,)`` validity mask for padded fleet slots.
    crosses_sun:
        Optional ``(n, n)`` 0/1 matrix; channel set to 0 when omitted.
    mode, tau, eta_norm:
        Flux imminence weighting (see :func:`_flux_weight`).
    ship_norm:
        Divides the flux channels for scale parity with the node metrics.

    Returns
    -------
    ``(n, n, 4)`` ``float32`` directed edge feature tensor.
    """
    d = np.asarray(distance_matrix, dtype=np.float32)
    n = int(d.shape[0])
    out = np.zeros((n, n, REWORKED_EDGE_FEATURE_DIM), dtype=np.float32)
    if n <= 0:
        return out

    out[:, :, EDGE_IDX_DISTANCE] = d
    if crosses_sun is not None:
        cs = np.asarray(crosses_sun, dtype=np.float32)
        out[:, :, EDGE_IDX_CROSSES_SUN] = np.clip(cs, 0.0, 1.0)

    friendly, enemy = inflight_flux_matrices(
        n,
        fleet_src, fleet_tgt, fleet_eta, fleet_ships, fleet_is_self, fleet_is_enemy,
        fleet_mask=fleet_mask,
        mode=mode, tau=tau, eta_norm=eta_norm,
    )
    scale = 1.0 / max(float(ship_norm), 1e-6)
    out[:, :, EDGE_IDX_FRIENDLY_FLUX] = friendly * scale
    out[:, :, EDGE_IDX_ENEMY_FLUX] = enemy * scale
    return out


DEFAULT_AFFORD_NORM: float = 50.0
DEFAULT_CAPTURE_MARGIN: float = 1.0


def capture_affordability(
    distance_matrix: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    *,
    player: int,
    incoming_enemy: "np.ndarray | None" = None,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    capture_margin: float = DEFAULT_CAPTURE_MARGIN,
    norm: float = DEFAULT_AFFORD_NORM,
) -> np.ndarray:
    r"""Per-planet **signed capacity margin** (deployable force − demand), in
    normalised ship units.

    A single legible channel that mirrors the allocator's demand sizing so the
    policy can perceive *capacity* — "can I actually afford / hold this?" —
    without re-deriving the logistics solver from raw channels.  Positive ⇒ a
    standalone-rational target (we could fund it if we focused); negative ⇒
    over-reach.  The *joint* budgeting across several selected targets is still
    the Plackett-Luce ranking + allocator's job; this is only the per-target
    standalone gauge.

    Deployable mass reachable to planet ``T`` is the agent's total ship stock
    that could converge on it (all own garrisons except ``T``'s own):

    .. math::
        \text{reach}(T) = \sum_{h\in\text{mine},\, h\ne T} s_h

    **Attack** (``T`` not mine).  Mirror of the allocator's attack demand
    (neutrals do not reinforce, enemies do):

    .. math::
        \text{demand}(T) = s_T + r_T\,\text{eta}_{\min}(T) + \text{margin},\quad
        r_T = \begin{cases}0 & \text{neutral}\\ \text{prod}_T & \text{enemy}\end{cases}

    with ``eta_min`` the nearest own-source ETA.  Margin ``= reach − demand``.
    If no own source can reach ``T`` the attack is infeasible ⇒ ``−demand``.

    **Defend** (``T`` mine).  Garrison plus reachable reinforcement against the
    incoming enemy estimate: ``margin = s_T + reach(T) − incoming(T)``.

    Returns ``(n,)`` ``float32`` margin ``/ norm`` (``~O(1)``).
    """
    owner_a = np.asarray(owner).reshape(-1)
    n = owner_a.shape[0]
    ships_a = np.asarray(ships, dtype=np.float64).reshape(n)
    prod_a = np.asarray(production, dtype=np.float64).reshape(n)
    is_mine = owner_a == int(player)
    is_neutral = owner_a == -1

    raw = np.asarray(distance_matrix, dtype=np.float64) * float(board_size)  # (n, n)
    speed = np.asarray(fleet_speed(ships_a, max_speed=max_speed), dtype=np.float64)
    speed_safe = np.clip(speed, 1e-6, None)
    eta = raw / speed_safe[:, None]  # eta[h, T] from source h

    # Nearest OWN-source ETA to each target (exclude own garrison h == T).
    eta_mine = np.where(is_mine[:, None], eta, np.inf)
    np.fill_diagonal(eta_mine, np.inf)
    eta_nearest = eta_mine.min(axis=0)            # (n,)
    has_src = np.isfinite(eta_nearest)
    eta_for_demand = np.where(has_src, eta_nearest, 0.0)

    total_mine = float(np.where(is_mine, ships_a, 0.0).sum())
    reach = total_mine - np.where(is_mine, ships_a, 0.0)  # exclude T's own garrison

    reinforce = np.where(is_neutral, 0.0, prod_a)         # neutrals don't reinforce
    attack_demand = ships_a + reinforce * eta_for_demand + float(capture_margin)
    margin_attack = np.where(has_src, reach - attack_demand, -attack_demand)

    inc = (np.zeros(n) if incoming_enemy is None
           else np.asarray(incoming_enemy, dtype=np.float64).reshape(n))
    margin_defend = ships_a + reach - inc

    margin = np.where(is_mine, margin_defend, margin_attack)
    return (margin / max(float(norm), 1e-6)).astype(np.float32)


def incoming_enemy_estimate(
    distance_matrix: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    *,
    player: int,
    fleet_src: np.ndarray,
    fleet_tgt: np.ndarray,
    fleet_eta: np.ndarray,
    fleet_ships: np.ndarray,
    fleet_is_self: np.ndarray,
    fleet_is_enemy: np.ndarray,
    fleet_mask: "np.ndarray | None" = None,
    horizon: float = 15.0,
    potential_weight: float = 0.0,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> np.ndarray:
    r"""Per-planet estimate of enemy ships about to hit it (defense demand).

    Produces the ``incoming_enemy`` array consumed by
    :func:`reworked_allocator.allocate_launches` to size defend demands.  It is
    expressed in **raw ship units** (not gravity-weighted) so it can be
    compared directly against a planet's garrison.

    Two ship-unit components:

    1. **In-flight (committed).**  The total enemy ship mass already *en route*
       to each planet — the column-sum of the raw enemy flux matrix
       (:func:`inflight_flux_matrices` with ``mode="raw"``).  These fleets are
       launched and will arrive, so they count at full mass.
    2. **Potential within horizon (optional).**  Enemy garrisons that could
       *reach* the planet within ``horizon`` turns: for each enemy planet ``h``
       with ``eta(h -> T) = dist / fleet_speed(ships_h) <= horizon``, add
       ``ships_h``.  Scaled by ``potential_weight`` (default 0 -> in-flight
       only).  Raising it makes defense more proactive at the cost of
       over-reserving.

    .. math::

        \text{incoming}(T) = \underbrace{\sum_{k:\,\text{enemy fleet} \to T} s_k}_{\text{in-flight}}
            + w \cdot \underbrace{\sum_{\substack{h:\,\text{enemy} \\ \text{eta}(h\to T)\le H}} s_h}_{\text{potential}}

    Returns
    -------
    ``(n,)`` ``float32`` non-negative incoming enemy ship estimate per planet.

    Notes
    -----
    In-flight fleets are counted raw (timing-agnostic); a future refinement
    could discount fleets that arrive far beyond the defense window.
    """
    n = int(np.asarray(owner).shape[0])

    _, enemy_flux = inflight_flux_matrices(
        n,
        fleet_src, fleet_tgt, fleet_eta, fleet_ships, fleet_is_self, fleet_is_enemy,
        fleet_mask=fleet_mask,
        mode=FLUX_MODE_RAW,
    )
    incoming = enemy_flux.sum(axis=0).astype(np.float64)            # (n,) inbound raw ships

    if potential_weight > 0.0:
        owner_a = np.asarray(owner).reshape(n)
        ships_a = np.asarray(ships, dtype=np.float64).reshape(n)
        raw_d = np.asarray(distance_matrix, dtype=np.float64) * float(board_size)
        speed = np.asarray(fleet_speed(ships_a, max_speed=max_speed), dtype=np.float64)
        speed_safe = np.clip(speed, 1e-6, None)
        eta = raw_d / speed_safe[:, None]                          # eta[h, T]
        is_enemy = (owner_a != int(player)) & (owner_a != -1)
        within = (eta <= float(horizon))
        np.fill_diagonal(within, False)
        contrib = (is_enemy.astype(np.float64) * ships_a)[:, None] * within
        incoming = incoming + float(potential_weight) * contrib.sum(axis=0)

    return incoming.astype(np.float32)
