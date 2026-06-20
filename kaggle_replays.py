"""Download and index Kaggle Orbit Wars episode replays for imitation learning.

Requires ``~/.kaggle/kaggle.json`` (Kaggle → Account → Create New Token) and
competition rules accepted for ``orbit-wars``.

Typical flow for top leaderboard teams::

    leaderboard (teamId)
      → team-submissions (submission id)
      → episodes (episode id + agent roster)
      → replay JSON download
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

COMPETITION = "orbit-wars"


@dataclass
class EpisodeRecord:
    episode_id: int
    submission_id: int
    expert_agent_index: int
    expert_team_id: int
    expert_team_name: str
    path: str
    reward: Optional[float] = None


@dataclass
class TeamRecord:
    rank: int
    team_id: int
    team_name: str
    score: str
    submissions: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[EpisodeRecord] = field(default_factory=list)


@dataclass
class LeaderReplayManifest:
    competition: str
    downloaded_at: str
    top_n: int
    episodes_per_team: int
    teams: list[TeamRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition": self.competition,
            "downloaded_at": self.downloaded_at,
            "top_n": self.top_n,
            "episodes_per_team": self.episodes_per_team,
            "teams": [
                {
                    **{k: v for k, v in asdict(t).items() if k != "episodes"},
                    "episodes": [asdict(e) for e in t.episodes],
                }
                for t in self.teams
            ],
        }

    @classmethod
    def load(cls, path: Path | str) -> LeaderReplayManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        teams: list[TeamRecord] = []
        for t in raw.get("teams", []):
            eps = [EpisodeRecord(**e) for e in t.get("episodes", [])]
            teams.append(TeamRecord(
                rank=int(t["rank"]),
                team_id=int(t["team_id"]),
                team_name=str(t["team_name"]),
                score=str(t.get("score", "")),
                submissions=list(t.get("submissions", [])),
                episodes=eps,
            ))
        return cls(
            competition=str(raw.get("competition", COMPETITION)),
            downloaded_at=str(raw.get("downloaded_at", "")),
            top_n=int(raw.get("top_n", len(teams))),
            episodes_per_team=int(raw.get("episodes_per_team", 0)),
            teams=teams,
        )

    def iter_episode_paths(self) -> list[tuple[EpisodeRecord, Path]]:
        out: list[tuple[EpisodeRecord, Path]] = []
        for team in self.teams:
            for ep in team.episodes:
                out.append((ep, Path(ep.path)))
        return out


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s[:48] or "team"


def _kaggle_bin() -> str:
    kaggle = shutil.which("kaggle")
    if kaggle:
        return kaggle
    # Do not Path.resolve() sys.executable: venv shims point at base Python.
    for cand in (
        Path(sys.executable).parent / "kaggle",
        Path(sys.prefix) / "bin" / "kaggle",
    ):
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "kaggle CLI not found. Install with: pip install -U kaggle"
    )


def _run_kaggle_csv(args: list[str]) -> list[dict[str, str]]:
    cmd = [_kaggle_bin(), "competitions", *args, "-v"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows: list[dict[str, str]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < len(headers):
            parts.extend([""] * (len(headers) - len(parts)))
        rows.append({headers[i]: parts[i].strip() for i in range(len(headers))})
    return rows


def _run_kaggle_table(args: list[str]) -> list[dict[str, str]]:
    """Parse whitespace table output from kaggle CLI (no -v)."""
    cmd = [_kaggle_bin(), "competitions", *args]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [ln.rstrip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    # Drop header separator line if present (----)
    data_lines = [ln for ln in lines if not set(ln.strip()) <= {"-", " "}]
    if len(data_lines) < 2:
        return []
    header = data_lines[0].split()
    rows: list[dict[str, str]] = []
    for ln in data_lines[1:]:
        parts = ln.split()
        if len(parts) < len(header):
            continue
        row = {header[i]: parts[i] for i in range(len(header))}
        rows.append(row)
    return rows


def fetch_leaderboard_top(competition: str, top_n: int) -> list[dict[str, str]]:
    rows = _run_kaggle_csv(["leaderboard", competition, "--show", "--page-size", str(int(top_n))])
    if rows:
        return rows[: int(top_n)]
    # Fallback: table mode
    rows = _run_kaggle_table(["leaderboard", competition, "-s"])
    return rows[: int(top_n)]


def fetch_team_submissions(team_id: int) -> list[dict[str, str]]:
    return _run_kaggle_csv(["team-submissions", str(int(team_id))])


def fetch_submission_episodes(submission_id: int) -> list[dict[str, str]]:
    return _run_kaggle_csv(["episodes", str(int(submission_id))])


def _parse_float(value: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("-inf")


def _pick_submissions(rows: list[dict[str, str]], max_submissions: int) -> list[dict[str, str]]:
    if not rows:
        return []
    # Prefer highest publicScore; fall back to most recent id order.
    score_key = "publicScore" if any("publicScore" in r for r in rows) else "score"
    ranked = sorted(rows, key=lambda r: _parse_float(r.get(score_key, "0")), reverse=True)
    return ranked[: max(1, int(max_submissions))]


def _expert_index_for_team(episode_row: dict[str, str], team_id: int) -> int:
    """Best-effort agent index lookup from episodes CSV columns."""
    for key, val in episode_row.items():
        if "teamId" in key and str(val) == str(team_id):
            # column like agent0teamId → index 0
            m = re.search(r"agent(\d+)", key, flags=re.I)
            if m:
                return int(m.group(1))
    # episodes table may expose agents in nested JSON via API; CLI CSV often has agent0teamId...
    for i in range(8):
        if str(episode_row.get(f"agent{i}teamId", "")) == str(team_id):
            return i
    return 0


def download_episode_replay(episode_id: int, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_kaggle_bin(), "competitions", "replay", str(int(episode_id)), "-p", str(dest_dir), "-q"]
    subprocess.run(cmd, check=True)
    named = dest_dir / f"episode-{int(episode_id)}-replay.json"
    if named.is_file():
        return named
    candidates = sorted(dest_dir.glob(f"*{episode_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Replay for episode {episode_id} not found under {dest_dir}")


def download_leader_replays(
    *,
    out_dir: Path | str,
    top_n: int = 3,
    episodes_per_team: int = 12,
    max_submissions_per_team: int = 2,
    competition: str = COMPETITION,
) -> LeaderReplayManifest:
    """Download recent episode replays for the top ``top_n`` leaderboard teams."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = LeaderReplayManifest(
        competition=str(competition),
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        top_n=int(top_n),
        episodes_per_team=int(episodes_per_team),
    )

    board = fetch_leaderboard_top(competition, top_n)
    if not board:
        raise RuntimeError(f"Leaderboard empty for {competition!r}. Check auth and rules acceptance.")

    for rank, row in enumerate(board, start=1):
        team_id = int(row.get("teamId") or row.get("team_id") or 0)
        team_name = str(row.get("teamName") or row.get("team_name") or f"team_{team_id}")
        score = str(row.get("score") or "")
        if team_id <= 0:
            continue

        team_dir = out / f"rank{rank:02d}_{_slug(team_name)}"
        team_dir.mkdir(parents=True, exist_ok=True)
        team_rec = TeamRecord(rank=rank, team_id=team_id, team_name=team_name, score=score)

        sub_rows = fetch_team_submissions(team_id)
        picked = _pick_submissions(sub_rows, max_submissions_per_team)
        team_rec.submissions = picked

        episode_ids: list[tuple[int, int, int]] = []  # episode_id, submission_id, expert_idx
        for sub in picked:
            sid = int(sub.get("id") or sub.get("submissionId") or 0)
            if sid <= 0:
                continue
            ep_rows = fetch_submission_episodes(sid)
            # Most recent episodes last in CSV; take tail.
            for ep_row in ep_rows[-max(0, int(episodes_per_team)) :]:
                eid = int(ep_row.get("id") or ep_row.get("episodeId") or 0)
                if eid <= 0:
                    continue
                expert_idx = _expert_index_for_team(ep_row, team_id)
                episode_ids.append((eid, sid, expert_idx))

        # Deduplicate while preserving order; cap total per team.
        seen: set[int] = set()
        unique: list[tuple[int, int, int]] = []
        for item in episode_ids:
            if item[0] in seen:
                continue
            seen.add(item[0])
            unique.append(item)
        unique = unique[-int(episodes_per_team) :]

        for eid, sid, expert_idx in unique:
            ep_dir = team_dir / "episodes"
            src = download_episode_replay(eid, ep_dir)
            final = ep_dir / f"episode_{eid}.json"
            if src.resolve() != final.resolve():
                final.write_bytes(src.read_bytes())
            team_rec.episodes.append(EpisodeRecord(
                episode_id=int(eid),
                submission_id=int(sid),
                expert_agent_index=int(expert_idx),
                expert_team_id=int(team_id),
                expert_team_name=str(team_name),
                path=str(final.resolve()),
            ))
            print(f"[download] rank={rank} team={team_name!r} episode={eid} expert_idx={expert_idx}")

        manifest.teams.append(team_rec)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"[download] manifest -> {manifest_path} ({sum(len(t.episodes) for t in manifest.teams)} episodes)")
    return manifest


def load_replay_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
