"""Chart pattern detectors (project copy) for any daily OHLCV DataFrame. Educational use only."""
from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION = {"Spring": "bull", "W": "bull", "Cup": "bull", "Tri": "bull", "M": "bear", "HS": "bear"}


def zigzag(c, thr):
    piv = []
    hi = lo = 0
    trend = 0
    for i in range(1, len(c)):
        if trend >= 0 and c[i] > c[hi]:
            hi = i
        if trend <= 0 and c[i] < c[lo]:
            lo = i
        if trend == 0:
            if c[i] >= c[lo] * (1 + thr):
                piv.append((lo, "T", i)); trend = 1; hi = i
            elif c[i] <= c[hi] * (1 - thr):
                piv.append((hi, "P", i)); trend = -1; lo = i
        elif trend == 1 and c[i] <= c[hi] * (1 - thr):
            piv.append((hi, "P", i)); trend = -1; lo = i
        elif trend == -1 and c[i] >= c[lo] * (1 + thr):
            piv.append((lo, "T", i)); trend = 1; hi = i
    return piv


def _dedup(cands, gap):
    out = []
    for x in sorted(cands, key=lambda x: x["done"]):
        if not out or x["done"] - out[-1]["done"] >= gap or x["pattern"] != out[-1]["pattern"]:
            out.append(x)
    return out


def springs(c, h, l):
    n = len(c)
    out = []
    for i in range(40, n - 3):
        W = 30
        seg = slice(i - W, i)
        floor = l[seg].min()
        if l[i] >= floor * 0.99:
            continue
        touch = np.where(l[seg] <= floor * 1.02)[0]
        visits = 1 + int(np.sum(np.diff(touch) > 3)) if len(touch) else 0
        if visits < 2:
            continue
        if (h[seg].max() - floor) / floor > 0.25:
            continue
        for j in range(i, i + 4):
            if c[j] > floor:
                out.append(dict(pattern="Spring", done=j, floor=float(floor), mark=i, start=i - W))
                break
    return _dedup(out, 15)


def doubles(c, zz, kind):
    n = len(c)
    out = []
    typ = "T" if kind == "W" else "P"
    mid = "P" if kind == "W" else "T"
    for a in range(len(zz) - 2):
        (i1, t1, _), (im, t2, _), (i2, t3, _) = zz[a], zz[a + 1], zz[a + 2]
        if (t1, t2, t3) != (typ, mid, typ):
            continue
        if abs(c[i2] / c[i1] - 1) > 0.03 or not 15 <= i2 - i1 <= 60:
            continue
        if kind == "W":
            if c[im] / max(c[i1], c[i2]) - 1 < 0.08:
                continue
            for j in range(i2 + 1, min(n, i2 + 41)):
                if c[j] > c[im]:
                    out.append(dict(pattern="W", done=j, pts=[i1, im, i2], level=float(c[im]), start=i1 - 20)); break
        else:
            if 1 - c[im] / min(c[i1], c[i2]) < 0.08:
                continue
            for j in range(i2 + 1, min(n, i2 + 41)):
                if c[j] < c[im]:
                    out.append(dict(pattern="M", done=j, pts=[i1, im, i2], level=float(c[im]), start=i1 - 20)); break
    return out


def head_shoulders(c, zz):
    n = len(c)
    out = []
    for a in range(len(zz) - 4):
        s = zz[a:a + 5]
        if [x[1] for x in s] != ["P", "T", "P", "T", "P"]:
            continue
        (p1, _, _), (t1, _, _), (p2, _, _), (t2, _, _), (p3, _, _) = s
        if p3 - p1 > 120:
            continue
        if not (c[p2] >= 1.03 * max(c[p1], c[p3])):
            continue
        if abs(c[p1] / c[p3] - 1) > 0.10:
            continue
        slope = (c[t2] - c[t1]) / (t2 - t1)
        for j in range(p3 + 1, min(n, p3 + 31)):
            if c[j] < c[t1] + slope * (j - t1):
                out.append(dict(pattern="HS", done=j, pts=[p1, t1, p2, t2, p3],
                                neck=[t1, float(c[t1]), t2, float(c[t2])], start=p1 - 20))
                break
    return out


def cups(c, v, v50, zz):
    n = len(c)
    out = []
    for (L, t, _) in zz:
        if t != "P" or L < 70:
            continue
        if c[L] / c[L - 63:L].min() < 1.30:
            continue
        R = None
        for k in range(L + 30, min(n - 5, L + 121)):
            if c[k] >= 0.90 * c[L]:
                R = k; break
        if R is None:
            continue
        B = L + int(np.argmin(c[L:R + 1]))
        depth = 1 - c[B] / c[L]
        if not 0.12 <= depth <= 0.35 or not 30 <= R - L <= 120 or B - L < 8:
            continue
        r2 = R + int(np.argmax(c[R:R + 11]))
        done = None
        for j in range(r2 + 1, min(n, r2 + 26)):
            if c[j] >= c[r2]:
                done = j; break
        if done is None or done - r2 < 5:
            continue
        pull = 1 - c[r2:done + 1].min() / c[r2]
        if not 0.05 <= pull <= 0.15:
            continue
        if np.nanmean(v[r2:done + 1]) >= v50[r2]:
            continue
        out.append(dict(pattern="Cup", done=done, pts=[L, B, r2], level=float(c[r2]), start=L - 15))
    return out


def triangles(c):
    n = len(c)
    zz = zigzag(c, 0.05)
    out = []
    for s in range(len(zz)):
        for e in range(s + 3, min(len(zz), s + 10)):
            seq = zz[s:e + 1]
            i0, i1 = seq[0][0], seq[-1][0]
            if not 20 <= i1 - i0 <= 80:
                continue
            pk = [x[0] for x in seq if x[1] == "P"]
            tr = [x[0] for x in seq if x[1] == "T"]
            if len(pk) < 2 or len(tr) < 2 or seq[-1][1] != "P":
                continue
            ceil = max(c[p] for p in pk)
            if any(c[p] < ceil * 0.98 for p in pk):
                continue
            if any(b - a < 5 for a, b in zip(pk, pk[1:])):
                continue
            if any(c[b] < c[a] * 1.01 for a, b in zip(tr, tr[1:])):
                continue
            if c[tr[0]] > ceil * 0.95:
                continue
            if c[i0:i1 + 1].max() > ceil * 1.005:
                continue
            for j in range(i1 + 1, min(n, i1 + 21)):
                if c[j] > ceil:
                    out.append(dict(pattern="Tri", done=j, pk=pk, tr=tr, level=float(ceil), start=i0 - 20))
                    break
    out = sorted(out, key=lambda x: x["done"])
    ded = []
    for x in out:
        if not ded or x["done"] - ded[-1]["done"] >= 5:
            ded.append(x)
    return ded


def detect_all(df: pd.DataFrame):
    c, h, l, v = (df[k].values for k in ("close", "high", "low", "volume"))
    v50 = pd.Series(v).rolling(50).mean().values
    zz6 = zigzag(c, 0.06)
    cands = springs(c, h, l) + doubles(c, zz6, "W") + doubles(c, zz6, "M") + head_shoulders(c, zz6) + cups(c, v, v50, zz6)
    cands = _dedup(cands, 5) + triangles(c)
    for x in cands:
        x["direction"] = DIRECTION[x["pattern"]]
        x["date"] = str(df.index[x["done"]].date())
    return sorted(cands, key=lambda x: (x["done"], x["pattern"]))
