"""Session history: every rep score is saved so you can compare against your best rep / best session.

Stored as JSON in data/history.json (git-ignored). A session is one run of a drill (one `python live.py`
run for smash, one 12-rep set for footwork, or one analysed footwork video you chose to save).
"""
from __future__ import annotations

import json
import os
import time
import uuid

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.json")
MIN_REPS_FOR_BEST = 3   # a 1-rep "session" shouldn't count as your best session


def _avg(xs):
    return sum(xs) / len(xs) if xs else None


class Session:
    def __init__(self, hist: "History", rec: dict):
        self.hist, self.rec = hist, rec

    @property
    def id(self):
        return self.rec["id"]

    @property
    def scores(self) -> list[float]:
        return self.rec["scores"]

    @property
    def n(self):
        return len(self.scores)

    @property
    def avg(self):
        return _avg(self.scores)

    @property
    def best(self):
        return max(self.scores) if self.scores else None

    def add(self, score: float, detail: dict | None = None):
        self.rec["scores"].append(round(float(score), 1))
        if detail:
            self.rec.setdefault("details", []).append(detail)
        self.rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.hist._save_session(self.rec)

    def compare(self) -> list[str]:
        """Short lines comparing this session with your best and previous ones (same drill)."""
        h, d = self.hist, self.rec["drill"]
        lines = []
        if self.avg is not None:
            line = f"Session avg {self.avg:.0f} over {self.n} rep{'s' * (self.n != 1)}"
            bs = h.best_session(d, exclude=self.id)
            if bs:
                line += f"  |  best session {bs['avg']:.0f} ({self.avg - bs['avg']:+.0f})"
            lines.append(line)
        last = h.last_session(d, exclude=self.id)
        if last and self.avg is not None:
            lines.append(f"Last session {last['avg']:.0f} ({self.avg - last['avg']:+.0f})")
        prev = h.all_scores(d, exclude=self.id)
        if self.best is not None and prev and self.best > max(prev):
            lines.append(f"NEW best rep: {self.best:.0f} (was {max(prev):.0f})")
        elif prev:
            lines.append(f"Best rep ever {max(prev + self.scores):.0f}")
        return lines


class History:
    def __init__(self, path: str | None = None):
        self.path = path = path or DEFAULT_PATH
        self.sessions: list[dict] = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.sessions = json.load(f).get("sessions", [])
            except (OSError, ValueError):
                os.replace(path, path + ".corrupt")   # keep the bad file, start fresh
                self.sessions = []

    def start(self, drill: str, meta: dict | None = None) -> Session:
        rec = dict(id=uuid.uuid4().hex[:8], drill=drill, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                   ended=None, scores=[], meta=meta or {})
        return Session(self, rec)   # only written to disk once it has a rep

    def _save_session(self, rec: dict):
        if rec not in self.sessions:
            self.sessions.append(rec)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"sessions": self.sessions}, f, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ queries
    def for_drill(self, drill: str, exclude: str | None = None) -> list[dict]:
        out = []
        for s in self.sessions:
            if s["drill"] == drill and s["id"] != exclude and s["scores"]:
                out.append(dict(s, avg=_avg(s["scores"]), best=max(s["scores"]), n=len(s["scores"])))
        return out

    def all_scores(self, drill, exclude=None):
        return [x for s in self.for_drill(drill, exclude) for x in s["scores"]]

    def best_session(self, drill, exclude=None, min_reps=MIN_REPS_FOR_BEST):
        ss = [s for s in self.for_drill(drill, exclude) if s["n"] >= min_reps]
        return max(ss, key=lambda s: s["avg"]) if ss else None

    def last_session(self, drill, exclude=None):
        ss = self.for_drill(drill, exclude)
        return ss[-1] if ss else None

    def best_rep(self, drill):
        xs = self.all_scores(drill)
        return max(xs) if xs else None
