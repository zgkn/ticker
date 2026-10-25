"""The two ticker lists: the watchlist and the candidates under consideration.

    watchlist.txt    tickers being actively watched
    candidates.txt   tickers under consideration for the watchlist

Both are refreshed by the daily run and shown in separate sections of the
Overview tab. A ticker on both lists counts as watchlist. Files are plain text,
one symbol per line, '#' starts a comment (comments are preserved when the
scripts edit a file).

    python scripts/lists.py watchlist  "NVDA AAPL"   # add to watchlist (and take off candidates)
    python scripts/lists.py candidates "PLTR, UBER"  # add to candidates (and take off watchlist)
    python scripts/lists.py remove     "RIVN"        # off both lists and delete its published data

Symbols are validated before anything is written: these values arrive as
workflow inputs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_site  # noqa: E402
import data_lib  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
WATCHLIST = os.path.join(ROOT, "watchlist.txt")
CANDIDATES = os.path.join(ROOT, "candidates.txt")

WATCH, CONSIDER, OTHER = "watch", "consider", "other"
ACTIONS = ("watchlist", "candidates", "remove")


def _token(line: str) -> str:
    return line.split("#")[0].strip().upper()


def read_symbols(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return data_lib.parse_symbols([_token(ln) for ln in fh if _token(ln)])


def read_lists(watch_path: str | None = None, cand_path: str | None = None) -> dict[str, str]:
    """symbol -> "watch" | "consider", watchlist first; a ticker on both counts as watch."""
    out = {s: WATCH for s in read_symbols(watch_path or WATCHLIST)}
    for s in read_symbols(cand_path or CANDIDATES):
        out.setdefault(s, CONSIDER)
    return out


def add_symbols(path: str, symbols: list[str]) -> list[str]:
    """Append symbols not already in the file. Returns those added."""
    have = set(read_symbols(path))
    new = [s for s in symbols if s not in have]
    if new:
        text = open(path).read() if os.path.exists(path) else ""
        with open(path, "w") as fh:
            fh.write(text + ("" if not text or text.endswith("\n") else "\n") + "".join(s + "\n" for s in new))
    return new


def remove_symbols(path: str, symbols: list[str]) -> list[str]:
    """Drop symbols from the file, keeping comments and other lines. Returns those removed."""
    if not os.path.exists(path):
        return []
    drop = set(symbols)
    with open(path) as fh:
        lines = fh.readlines()
    kept, removed = [], []
    for ln in lines:
        if _token(ln) in drop:
            removed.append(_token(ln))
        else:
            kept.append(ln)
    if removed:
        with open(path, "w") as fh:
            fh.writelines(kept)
    return removed


def apply(action: str, symbols: list[str], watch_path: str | None = None, cand_path: str | None = None, data_dir: str | None = None) -> dict[str, list[str]]:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
    w, c = watch_path or WATCHLIST, cand_path or CANDIDATES
    if action == "watchlist":
        return {"added": add_symbols(w, symbols), "removed_from_other": remove_symbols(c, symbols)}
    if action == "candidates":
        return {"added": add_symbols(c, symbols), "removed_from_other": remove_symbols(w, symbols)}
    gone = remove_symbols(w, symbols) + remove_symbols(c, symbols)
    return {"removed": sorted(set(gone)), "data_deleted": build_site.remove_tickers(symbols, data_dir)}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ACTIONS:
        print(__doc__)
        return 2
    symbols = data_lib.parse_symbols(argv[1:])
    if not symbols:
        print("no tickers given", file=sys.stderr)
        return 2
    print(apply(argv[0], symbols))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
