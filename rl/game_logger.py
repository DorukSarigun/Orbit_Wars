"""Per-game JSON debug logger.

Writes a single ``*.jsonl`` file per episode capturing the full trace
of a game: planet / fleet state at each turn, every action the policy
(and any scripted opponent) executed, the reward that landed on each
transition, and a compact slice of the network's outputs (value
estimate, log-prob, chosen src/tgt/ship for every PPO decision).

The output is intentionally human-readable and lossless w.r.t. the
gameplay event stream so a developer can later open the file in
``jq``, a notebook, or any text editor and reconstruct exactly *what*
the policy saw and *how* it reacted, without re-running the rollout.

Format
------
One JSON object per line (``.jsonl``).  The first line is a
``"event": "header"`` record carrying episode-level metadata; one
``"event": "turn"`` record per env step follows; the last line is a
``"event": "summary"`` record with terminal rewards and totals.

The logger is a passive sink: ``collect_episode`` calls
``GameLogger.log_*`` at well-defined points and the logger itself
holds the open file handle and a small in-memory turn buffer until
``GameLogger.finish`` is invoked.

This module has *no* dependency on torch or torch_geometric so it can
be imported regardless of optional backends being available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch / dataclass objects to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # NaN / inf are not valid JSON.  Replace with None so the dump
        # never crashes, but log values that *should* be finite.
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    # Fallback: best-effort str().  This keeps the logger total-coverage
    # even if a new object type slips in.
    try:
        return str(obj)
    except Exception:
        return None


def _round(v: float, ndigits: int = 4) -> float:
    """Trim float precision to keep the log file small and diff-friendly."""
    if v is None:
        return None  # type: ignore[return-value]
    try:
        return round(float(v), ndigits)
    except Exception:
        return v


def parse_env_planet(p: Any) -> dict[str, Any]:
    """Compact dict of a feature_extraction.Planet."""
    out = {
        "id": int(getattr(p, "id", -1)),
        "owner": int(getattr(p, "owner", -1)),
        "ships": int(getattr(p, "ships", 0)),
        "production": int(getattr(p, "production", 0)),
        "radius": _round(float(getattr(p, "radius", 0.0)), 3),
        "x": _round(float(getattr(p, "x", 0.0)), 3),
        "y": _round(float(getattr(p, "y", 0.0)), 3),
    }
    if getattr(p, "is_comet", False):
        out["is_comet"] = True
    return out


def parse_env_fleet(f: Any) -> dict[str, Any]:
    """Compact dict of a feature_extraction.Fleet."""
    return {
        "src": int(getattr(f, "src_planet_id", getattr(f, "src", -1))),
        "tgt": int(getattr(f, "tgt_planet_id", getattr(f, "tgt", -1))),
        "owner": int(getattr(f, "owner", -1)),
        "ships": int(getattr(f, "ships", 0)),
        "eta": _round(float(getattr(f, "eta", 0.0)), 2),
    }


class GameLogger:
    """Append-only JSONL writer for a single episode.

    Usage::

        log = GameLogger(path, meta={"iteration": 5, "episode": 3})
        for turn in range(T):
            log.start_turn(turn, planets, fleets, ...)
            log.log_action(player=0, decision={...})
            log.log_rewards(turn_rewards, graph_metrics={...})
        log.finish(terminal_rewards=[1.0, -1.0], winner=0)
    """

    def __init__(self, path: Path | str, *, meta: Optional[Mapping[str, Any]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False
        self._turn: Optional[dict[str, Any]] = None
        self._actions_buffer: dict[int, list[dict[str, Any]]] = {}
        header = {
            "event": "header",
            "meta": _to_jsonable(dict(meta) if meta else {}),
        }
        self._write(header)

    # ----------------------------------------------------------------- IO
    def _write(self, obj: Mapping[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._fh.write(json.dumps(_to_jsonable(obj), ensure_ascii=False, separators=(",", ":")))
            self._fh.write("\n")
        except Exception:
            # The logger must never crash the rollout.  Best-effort write;
            # downstream tooling will see a truncated trace.
            pass

    # ----------------------------------------------------------- per-turn
    def start_turn(
        self,
        turn: int,
        *,
        planets: Sequence[Any],
        fleets: Sequence[Any],
        global_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Open a new turn record.  Subsequent log_action / log_rewards
        calls attach to it until ``end_turn()`` is invoked (or another
        ``start_turn``)."""
        if self._turn is not None:
            # Defensive flush -- caller forgot end_turn().
            self.end_turn()
        self._turn = {
            "event": "turn",
            "turn": int(turn),
            "planets": [parse_env_planet(p) for p in planets],
            "fleets": [parse_env_fleet(f) for f in fleets],
            "global": _to_jsonable(global_state or {}),
            "actions": {},
            "rewards": {},
        }
        self._actions_buffer = {}

    def log_action(self, *, player: int, decision: Mapping[str, Any]) -> None:
        """Append one decision (one row of the multi-launch loop) to the
        current player's action list.  ``decision`` is a free-form dict;
        recommended keys: src, tgt, ship_value, is_noop, launched,
        invalid, log_prob, value."""
        if self._turn is None:
            return
        bucket = self._actions_buffer.setdefault(int(player), [])
        bucket.append({
            k: _round(v, 4) if isinstance(v, (float, np.floating)) else _to_jsonable(v)
            for k, v in decision.items()
        })

    def log_rewards(
        self,
        rewards: Sequence[float],
        *,
        graph_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
        invalid_actions: Optional[Sequence[int]] = None,
    ) -> None:
        """Attach the per-player step reward + (optional) graph metric
        summary to the active turn record."""
        if self._turn is None:
            return
        self._turn["rewards"] = {
            str(p): _round(float(r), 6) for p, r in enumerate(rewards)
        }
        if invalid_actions is not None:
            self._turn["invalid_actions"] = {
                str(p): int(v) for p, v in enumerate(invalid_actions)
            }
        if graph_metrics is not None:
            self._turn["graph_metrics"] = [
                _to_jsonable(gm or {}) for gm in graph_metrics
            ]

    def end_turn(self) -> None:
        """Flush the current turn record to disk."""
        if self._turn is None:
            return
        # Move the buffered actions into the turn record under string keys.
        self._turn["actions"] = {
            str(p): acts for p, acts in self._actions_buffer.items()
        }
        self._write(self._turn)
        self._turn = None
        self._actions_buffer = {}

    # ------------------------------------------------------------ finish
    def finish(
        self,
        *,
        terminal_rewards: Optional[Sequence[float]] = None,
        cum_rewards: Optional[Sequence[float]] = None,
        winner: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Flush any pending turn, write the summary record, close the file."""
        if self._closed:
            return
        if self._turn is not None:
            self.end_turn()
        summary: dict[str, Any] = {"event": "summary"}
        if terminal_rewards is not None:
            summary["terminal_rewards"] = [_round(float(r), 6) for r in terminal_rewards]
        if cum_rewards is not None:
            summary["cum_rewards"] = [_round(float(r), 6) for r in cum_rewards]
        if winner is not None:
            summary["winner"] = int(winner)
        if extra:
            summary["extra"] = _to_jsonable(dict(extra))
        self._write(summary)
        try:
            self._fh.close()
        finally:
            self._closed = True

    def __enter__(self) -> "GameLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Even on exception we want a closed, well-formed file.
        try:
            self.finish()
        except Exception:
            pass


__all__ = ["GameLogger", "parse_env_planet", "parse_env_fleet"]
