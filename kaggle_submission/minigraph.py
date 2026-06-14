"""Minimal directed graph for Kaggle submission (no networkx dependency)."""

from __future__ import annotations

from typing import Any, Iterator


class MiniDiGraph:
    """Tiny ``networkx.DiGraph`` subset used by :mod:`orbit_graph`."""

    def __init__(self) -> None:
        self._nodes: dict[int, dict[str, Any]] = {}
        self._edges: dict[tuple[int, int], dict[str, Any]] = {}

    def add_node(self, node_id: int, **attrs: Any) -> None:
        self._nodes[int(node_id)] = dict(attrs)

    def add_edge(self, u: int, v: int, **attrs: Any) -> None:
        u, v = int(u), int(v)
        self._edges[(u, v)] = dict(attrs)

    def has_edge(self, u: int, v: int) -> bool:
        return (int(u), int(v)) in self._edges

    @property
    def edges(self) -> _EdgeView:
        return _EdgeView(self._edges)

    @property
    def nodes(self) -> _NodeView:
        return _NodeView(self._nodes)

    def in_edges(self, nbunch: int | None = None, *, data: bool = False):
        tid = int(nbunch) if nbunch is not None else None
        for (u, v), attrs in self._edges.items():
            if tid is not None and int(v) != tid:
                continue
            yield (u, v, attrs) if data else (u, v)

    def out_edges(self, nbunch: int | None = None, *, data: bool = False):
        sid = int(nbunch) if nbunch is not None else None
        for (u, v), attrs in self._edges.items():
            if sid is not None and int(u) != sid:
                continue
            yield (u, v, attrs) if data else (u, v)


class _EdgeView:
    def __init__(self, edges: dict[tuple[int, int], dict[str, Any]]) -> None:
        self._edges = edges

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        return self._edges[(int(key[0]), int(key[1]))]

    def __call__(self, *, data: bool = False) -> Iterator[tuple[int, int, dict[str, Any]]]:
        for (u, v), attrs in self._edges.items():
            yield (u, v, attrs) if data else (u, v)


class _NodeView:
    def __init__(self, nodes: dict[int, dict[str, Any]]) -> None:
        self._nodes = nodes

    def __getitem__(self, node_id: int) -> dict[str, Any]:
        return self._nodes[int(node_id)]
