#!/usr/bin/env python3
"""Record, replay and render full Orbit Wars games for final agents.

Examples:

    # Record final RL checkpoint vs orbit scripted, with terminal logs + HTML.
    ../orbit_wars_env/bin/python scripts/final_game_recorder.py record \
      --agent final_rl --ckpt checkpoints_final_rl/latest.pt \
      --opponent orbit --games 2 --max-steps 300 --html

    # Replay one JSON log in the terminal.
    ../orbit_wars_env/bin/python scripts/final_game_recorder.py replay \
      --log game_logs/final_game_000_p0_seed1000.json --sleep 0.05

    # Re-render HTML from a saved JSON log.
    ../orbit_wars_env/bin/python scripts/final_game_recorder.py render \
      --log game_logs/final_game_000_p0_seed1000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

logging.disable(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_extraction import parse_fleets, parse_planets  # noqa: E402
from orbit_match import make_controller, production_share_margin  # noqa: E402


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if hasattr(obj, "toJSON"):
        try:
            return _jsonable(obj.toJSON())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return repr(obj)


def _state_snapshot(state) -> list[dict[str, Any]]:
    return [_jsonable(state[p]) for p in range(len(state))]


def _obs_summary(obs: Any, player: int, num_agents: int) -> dict[str, Any]:
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)
    prod = {}
    ships = {}
    planets_by_owner = {}
    fleet_ships = {}
    for owner in [-1] + list(range(num_agents)):
        prod[str(owner)] = sum(int(p.production) for p in planets if int(p.owner) == owner)
        ships[str(owner)] = sum(int(p.ships) for p in planets if int(p.owner) == owner)
        planets_by_owner[str(owner)] = sum(1 for p in planets if int(p.owner) == owner)
        fleet_ships[str(owner)] = sum(int(f.ships) for f in fleets if int(f.owner) == owner)
    return {
        "player": int(player),
        "num_planets": len(planets),
        "num_fleets": len(fleets),
        "production": prod,
        "planet_ships": ships,
        "fleet_ships": fleet_ships,
        "planet_count": planets_by_owner,
        "prod_margin": production_share_margin(obs, int(player), int(num_agents)),
    }


class _NoopController:
    def act(self, env_obs, player, *, current_turn: int, max_turns: int) -> list[list]:
        return []


class _FinalRLDebugController:
    """Final RL controller that also exposes sampled action and reject reason."""

    def __init__(self, ckpt: str, *, device: str = "cpu", greedy: bool = True):
        import torch

        from final_obs import build_final_observation
        from final_rollout import _action_to_launch, _obs_to_tensors
        from final_train_rl import load_checkpoint

        self.torch = torch
        self.device = torch.device(device)
        self.policy, self.ckpt = load_checkpoint(ckpt, self.device)
        self.policy.eval()
        self.greedy = bool(greedy)
        self.build_obs = build_final_observation
        self.obs_to_tensors = _obs_to_tensors
        self.action_to_launch = _action_to_launch

    def act_debug(self, env_obs, player, *, current_turn: int, max_turns: int) -> tuple[list[list], dict[str, Any]]:
        obs = self.build_obs(env_obs, int(player))
        tensors = self.obs_to_tensors(obs, self.device)
        with self.torch.no_grad():
            action = self.policy.act(**tensors, greedy=self.greedy)
        launches, reason = self.action_to_launch(
            env_obs,
            obs,
            action,
            int(player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
        )
        src_slot = int(action.source_idx.item())
        tgt_slot = int(action.target_idx.item())
        src_id = int(obs.planet_ids[src_slot]) if 0 <= src_slot < len(obs.planet_ids) else -1
        tgt_id = int(obs.planet_ids[tgt_slot]) if 0 <= tgt_slot < len(obs.planet_ids) else -1
        return launches, {
            "reason": reason,
            "source_slot": src_slot,
            "target_slot": tgt_slot,
            "source_id": src_id,
            "target_id": tgt_id,
            "amount": float(action.amount.item()),
            "logprob": float(action.logprob.item()),
            "value": float(action.value.item()),
        }

    def act(self, env_obs, player, *, current_turn: int, max_turns: int) -> list[list]:
        launches, _debug = self.act_debug(
            env_obs, player, current_turn=current_turn, max_turns=max_turns)
        return launches


def _make_named_controller(kind: str, *, ckpt: str | None, device: str, sample: bool):
    if kind == "none":
        return _NoopController(), "none"
    if kind == "final_rl":
        if not ckpt:
            raise ValueError("--ckpt is required for --agent/--opponent final_rl")
        return _FinalRLDebugController(ckpt, device=device, greedy=not sample), f"final_rl:{ckpt}"
    if kind == "final_micro":
        from final_agent import FinalMicroAgent

        return FinalMicroAgent(), "final_micro"
    return make_controller(kind), kind


def _act_with_debug(controller, obs, player: int, *, current_turn: int, max_turns: int) -> tuple[list[list], dict[str, Any]]:
    t0 = time.perf_counter()
    if hasattr(controller, "act_debug"):
        actions, debug = controller.act_debug(
            obs, player, current_turn=current_turn, max_turns=max_turns)
    else:
        actions = controller.act(
            obs, player, current_turn=current_turn, max_turns=max_turns) or []
        debug = {"reason": "controller_action" if actions else "controller_noop"}
    elapsed = time.perf_counter() - t0
    debug = dict(debug)
    debug["elapsed_ms"] = round(1000.0 * elapsed, 3)
    debug["num_launches"] = len(actions or [])
    return actions or [], debug


def record_one_game(
    out_path: Path,
    *,
    controllers,
    controller_names: list[str],
    seed: int,
    max_steps: int,
    num_agents: int = 2,
    html: bool = False,
) -> dict[str, Any]:
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.configuration["randomSeed"] = int(seed)
    random.seed(int(seed))
    env.reset(num_agents=int(num_agents))
    state = env.step([[] for _ in range(num_agents)])

    log: dict[str, Any] = {
        "schema": "orbit_wars_final_game_log_v1",
        "seed": int(seed),
        "max_steps": int(max_steps),
        "num_agents": int(num_agents),
        "controllers": controller_names,
        "configuration": _jsonable(env.configuration),
        "initial_state": _state_snapshot(state),
        "initial_summary": [
            _obs_summary(state[p]["observation"], p, num_agents)
            for p in range(num_agents)
        ],
        "turns": [],
    }

    final_state = state
    done = False
    turn = -1
    for turn in range(int(max_steps)):
        before = _state_snapshot(state)
        before_summary = [
            _obs_summary(state[p]["observation"], p, num_agents)
            for p in range(num_agents)
        ]
        env_actions = []
        debug_rows = []
        for player, controller in enumerate(controllers):
            obs = state[player]["observation"]
            actions, debug = _act_with_debug(
                controller,
                obs,
                player,
                current_turn=turn,
                max_turns=max_steps,
            )
            env_actions.append(actions)
            debug_rows.append(debug)

        next_state = env.step(env_actions)
        done = bool(env.done or any(
            getattr(next_state[p], "status", "ACTIVE") == "DONE"
            for p in range(num_agents)
        ))
        after_summary = [
            _obs_summary(next_state[p]["observation"], p, num_agents)
            for p in range(num_agents)
        ]
        rewards = [
            float(getattr(next_state[p], "reward", 0.0) or 0.0)
            for p in range(num_agents)
        ]
        statuses = [
            str(getattr(next_state[p], "status", "ACTIVE"))
            for p in range(num_agents)
        ]
        log["turns"].append({
            "turn": int(turn),
            "state_before": before,
            "summary_before": before_summary,
            "actions": _jsonable(env_actions),
            "debug": _jsonable(debug_rows),
            "state_after": _state_snapshot(next_state),
            "summary_after": after_summary,
            "rewards": rewards,
            "statuses": statuses,
            "done": done,
        })
        state = next_state
        final_state = next_state
        if done:
            break

    final_obs = final_state[0]["observation"]
    final_rewards = [
        float(getattr(final_state[p], "reward", 0.0) or 0.0)
        for p in range(num_agents)
    ]
    log["final"] = {
        "turns": int(turn + 1),
        "done": bool(done),
        "rewards": final_rewards,
        "prod_margin_p0": production_share_margin(final_obs, 0, num_agents),
        "total_launches": [
            sum(len(row["actions"][p]) for row in log["turns"])
            for p in range(num_agents)
        ],
        "reason_counts": _reason_counts(log["turns"], num_agents),
    }
    log["kaggle_steps"] = _jsonable(getattr(env, "steps", []))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")

    html_path = None
    if html:
        html_path = out_path.with_suffix(".html")
        html_path.write_text(env.render(mode="html"), encoding="utf-8")
        log["html_path"] = str(html_path)
        out_path.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "json": str(out_path),
        "html": str(html_path) if html_path else None,
        **log["final"],
    }


def _reason_counts(turns: list[dict[str, Any]], num_agents: int) -> list[dict[str, int]]:
    rows = [dict() for _ in range(num_agents)]
    for turn in turns:
        for p, debug in enumerate(turn.get("debug", [])):
            reason = str(debug.get("reason", "unknown"))
            rows[p][reason] = rows[p].get(reason, 0) + 1
    return rows


def cmd_record(args) -> None:
    out_dir = Path(args.out_dir)
    summaries = []
    for i in range(int(args.games)):
        seed = int(args.seed) + i
        agent, agent_name = _make_named_controller(
            args.agent, ckpt=args.ckpt, device=args.device, sample=bool(args.sample))
        opp, opp_name = _make_named_controller(
            args.opponent, ckpt=args.opponent_ckpt, device=args.device, sample=bool(args.sample_opponent))
        matchups = [([agent, opp], [agent_name, opp_name], "p0")]
        if args.seat_swap:
            agent2, agent2_name = _make_named_controller(
                args.agent, ckpt=args.ckpt, device=args.device, sample=bool(args.sample))
            opp2, opp2_name = _make_named_controller(
                args.opponent, ckpt=args.opponent_ckpt, device=args.device, sample=bool(args.sample_opponent))
            matchups.append(([opp2, agent2], [opp2_name, agent2_name], "p1"))

        for controllers, names, seat in matchups:
            out_path = out_dir / f"{args.prefix}_{i:03d}_{seat}_seed{seed}.json"
            summary = record_one_game(
                out_path,
                controllers=controllers,
                controller_names=names,
                seed=seed,
                max_steps=int(args.max_steps),
                num_agents=2,
                html=bool(args.html),
            )
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True))

    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary: {summary_path}")


def _format_actions(actions: list[list]) -> str:
    if not actions:
        return "[]"
    return "[" + ", ".join(
        f"{int(a[0])}:{int(a[2])}@{float(a[1]):+.2f}" for a in actions
    ) + "]"


def cmd_replay(args) -> None:
    log = json.loads(Path(args.log).read_text(encoding="utf-8"))
    turns = log.get("turns", [])
    start = max(0, int(args.start))
    end = len(turns) if args.end is None else min(len(turns), int(args.end))
    print(
        f"log={args.log} seed={log.get('seed')} controllers={log.get('controllers')} "
        f"turns={len(turns)} final={log.get('final')}"
    )
    for row in turns[start:end]:
        turn = int(row["turn"])
        summaries = row.get("summary_before", [])
        after = row.get("summary_after", summaries)
        actions = row.get("actions", [])
        debug = row.get("debug", [])
        rewards = row.get("rewards", [])
        print(f"\nturn {turn:04d} rewards={rewards} done={row.get('done')}")
        for p, summary in enumerate(summaries):
            prod = summary.get("production", {})
            ships = summary.get("planet_ships", {})
            after_prod = after[p].get("production", {}) if p < len(after) else prod
            dbg = debug[p] if p < len(debug) else {}
            print(
                f"  p{p}: prod={prod} -> {after_prod} ships={ships} "
                f"act={_format_actions(actions[p] if p < len(actions) else [])} "
                f"reason={dbg.get('reason')} elapsed={dbg.get('elapsed_ms')}ms"
            )
            if args.debug:
                print(f"      debug={json.dumps(dbg, sort_keys=True)}")
        if args.sleep > 0:
            time.sleep(float(args.sleep))


def cmd_render(args) -> None:
    from kaggle_environments import make

    log_path = Path(args.log)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    steps = log.get("kaggle_steps")
    if not steps:
        raise ValueError(f"{log_path} does not contain kaggle_steps; cannot render")
    env = make(
        "orbit_wars",
        configuration=log.get("configuration"),
        steps=steps,
        info=log.get("info", {}),
    )
    out = Path(args.out or log_path.with_suffix(".html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.render(mode="html"), encoding="utf-8")
    print(f"html: {out.resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record full game logs")
    rec.add_argument("--agent", default="final_rl", help="final_rl, final_micro, orbit, mcts, timed_mcts, none")
    rec.add_argument("--ckpt", default=None, help="Checkpoint for --agent final_rl")
    rec.add_argument("--opponent", default="orbit")
    rec.add_argument("--opponent-ckpt", default=None)
    rec.add_argument("--games", type=int, default=1)
    rec.add_argument("--seed", type=int, default=1000)
    rec.add_argument("--max-steps", type=int, default=300)
    rec.add_argument("--device", default="cpu")
    rec.add_argument("--sample", action="store_true")
    rec.add_argument("--sample-opponent", action="store_true")
    rec.add_argument("--no-seat-swap", action="store_false", dest="seat_swap")
    rec.set_defaults(seat_swap=True)
    rec.add_argument("--html", action="store_true", help="Also save an HTML replay beside each JSON")
    rec.add_argument("--out-dir", default="game_logs")
    rec.add_argument("--prefix", default="final_game")
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("replay", help="Replay JSON log in terminal")
    rep.add_argument("--log", required=True)
    rep.add_argument("--start", type=int, default=0)
    rep.add_argument("--end", type=int, default=None)
    rep.add_argument("--sleep", type=float, default=0.0)
    rep.add_argument("--debug", action="store_true")
    rep.set_defaults(func=cmd_replay)

    ren = sub.add_parser("render", help="Render HTML replay from JSON log")
    ren.add_argument("--log", required=True)
    ren.add_argument("--out", default=None)
    ren.set_defaults(func=cmd_render)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
