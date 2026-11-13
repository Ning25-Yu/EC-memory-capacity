"""Exact finite-pulse engine for the 37-node network (asynchronous update, unit rates, ORIGINAL rules).

State space
-----------
For a context (four inputs clamped) and a set of pulse nodes, constants are obtained by logical percolation from
the inputs with the pulse nodes left symbolic, so they hold in both pulse phases. Dynamics are tracked on
  scope='core': the non-constant nodes in the upstream closure of Proliferation. Rules of core nodes refer only
                to core nodes and constants, so the projection onto the core is itself an exact Markov chain.
                The nodes outside the closure (Apoptosis, EMT, Invasion, MMP2, Metastasis, Migration, TFF3) are
                acyclic among themselves, so core fixed points / recurrent classes map one-to-one onto
                full-network fixed points / recurrent classes;
  scope='all' : every non-constant node (used when Invasion/Apoptosis read-outs are required).

Exact computation (`exact_pulse`)
---------------------------------
start distribution (point mass on a fixed point, or the exact within-class stationary distribution of a complex
recurrent class) -> pulse nodes set at onset and held for L (sparse matrix exponential) -> release -> exact
absorption probabilities into every terminal class (sparse linear solve). Used whenever the reachable set after
release is below `EXACT_CAP` states.

Replicated simulation (`simulate_pulse`)
----------------------------------------
For reachable sets above the cap (weakly canalised contexts), the SAME CTMC is sampled with a seeded, vectorised
Gillespie scheme on the same state space; trajectories stop on reaching a fixed point or a state of a known
complex recurrent class. Each replicate is an independent two-phase run; the absorption status is recorded at
t = 60 and t = 120 after release. The simulator is validated against exact cells (see 08_context_mechanism_scope.py).

Read-outs (kept distinct)
-------------------------
P_arrest_FP     probability of absorption into a fixed point with Proliferation = 0 (durable arrest);
P_prolif0_late  long-run probability that Proliferation = 0 (P_arrest_FP plus the Proliferation-OFF occupancy of
                any complex recurrent class that is re-entered).
"""

import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply, spsolve

import ecbn

NODES = ecbn.NODES
REG = {n: set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ecbn.RULES[n])) & set(NODES) for n in NODES}
PYR = {
    n: compile(
        ecbn.RULES[n].replace("!", " not ").replace("&", " and ").replace("|", " or ").strip(), n, "eval"
    )
    for n in NODES
}
VECR = {n: compile(ecbn.RULES[n].replace("!", "~").strip(), n, "eval") for n in NODES}
_SYM = {n: sympy.Symbol(n) for n in NODES}
_EXPR = {n: sympy.sympify(ecbn.RULES[n].replace("!", "~"), locals=_SYM) for n in NODES}
EXACT_CAP = 400_000
ABS_TOL = 1e-12  # certified bound on unabsorbed mass for the iterative absorption path
OUTPUTS = ("Proliferation", "Invasion", "Apoptosis")


def percolate(ctx, symbolic=()):
    const = dict(ecbn.ctx_bits(ctx))
    changed = True
    while changed:
        changed = False
        for n in NODES:
            if n in const or n in symbolic:
                continue
            e = _EXPR[n].subs({_SYM[k]: bool(v) for k, v in const.items()})
            if e in (sympy.true, sympy.false):
                const[n] = int(bool(e))
                changed = True
    return const


class Space:
    def __init__(self, ctx, pulse_nodes=(), scope="core"):
        self.ctx = ctx
        self.const = percolate(ctx, tuple(pulse_nodes))
        if scope == "core":
            tr = {"Proliferation"} - set(self.const)
            fr = list(tr)
            while fr:
                for r in REG[fr.pop()]:
                    if r not in tr and r not in self.const:
                        tr.add(r)
                        fr.append(r)
        else:
            tr = set(NODES) - set(self.const)
        self.nodes = sorted(tr)
        self.ix = {n: i for i, n in enumerate(self.nodes)}
        self.pulse_nodes = [p for p in pulse_nodes if p in self.ix]  # pulse nodes that can matter here

    def value(self, s, n):
        return s[self.ix[n]] if n in self.ix else self.const.get(n)

    def project(self, full):
        return tuple(int(full[n]) for n in self.nodes)

    def successors(self, s, clamp):
        env = dict(self.const)
        env.update(zip(self.nodes, s))
        out = []
        for n in self.nodes:
            if n in clamp:
                continue
            t = int(eval(PYR[n], {}, env))
            if t != s[self.ix[n]]:
                u = list(s)
                u[self.ix[n]] = t
                out.append(tuple(u))
        return out

    def explore(self, seeds, clamp, cap=None):
        order = list(dict.fromkeys(seeds))
        pos = {s: i for i, s in enumerate(order)}
        I = []
        J = []
        k = 0
        while k < len(order):
            if cap and len(order) > cap:
                return None
            for t in self.successors(order[k], clamp):
                if t not in pos:
                    pos[t] = len(order)
                    order.append(t)
                I.append(k)
                J.append(pos[t])
            k += 1
        n = len(order)
        A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
        Q = A - csr_matrix((np.asarray(A.sum(1)).ravel(), (range(n), range(n))), shape=(n, n))
        return order, pos, Q.tocsr(), A


def terminal_classes(A):
    nc, lab = connected_components(A, directed=True, connection="strong")
    return [m for m in (np.where(lab == c)[0] for c in range(nc)) if np.all(np.isin(A[m].nonzero()[1], m))]


def stationary(Q, mem):
    if len(mem) == 1:
        return np.array([1.0])
    M = Q[mem][:, mem].toarray().T
    M[-1, :] = 1.0
    b = np.zeros(len(mem))
    b[-1] = 1.0
    return np.linalg.solve(M, b)


def absorption(Q, classes, p0):
    lab = np.full(Q.shape[0], -1)
    for c, m in enumerate(classes):
        lab[m] = c
    res = np.array([p0[lab == c].sum() for c in range(len(classes))])
    T = np.where(lab < 0)[0]
    if len(T) and p0[T].sum() > 0:
        M = (-Q[T][:, T]).T.tocsc()
        b = p0[T]
        if len(T) > 5000:
            # Direct LU suffers fill-in on large transient sets. Iterate the uniformised chain P = I + Q/Lambda,
            # which has exactly the same absorption probabilities; the remaining transient mass is a certified
            # bound on the error of every absorption probability.
            Lam = float(-Q.diagonal().min()) or 1.0
            PT = (
                csr_matrix((np.ones(Q.shape[0]), (range(Q.shape[0]), range(Q.shape[0]))), shape=Q.shape)
                + Q / Lam
            ).T.tocsr()
            p = p0.copy()
            tr = lab < 0
            for it in range(2_000_000):
                p = PT @ p
                if it % 200 == 0 and p[tr].sum() < ABS_TOL:
                    break
            absorption.last_residual = float(p[tr].sum())
            return np.array([p[m].sum() for m in classes])
        y = spsolve(M, b)
        absorption.last_residual = 0.0
        QT = Q[T]
        for c, m in enumerate(classes):
            res[c] += float(y @ np.asarray(QT[:, m].sum(1)).ravel())
    return res


def settle_downstream(sp, s):
    """Values of all 37 nodes at a core fixed point (downstream nodes are acyclic and settle deterministically)."""
    env = {n: sp.value(s, n) for n in NODES if sp.value(s, n) is not None}
    for _ in range(len(NODES)):
        changed = False
        for n in NODES:
            if n in env and (n in sp.ix or n in sp.const):
                continue
            try:
                v = int(eval(PYR[n], {}, env))
            except NameError:
                continue
            if env.get(n) != v:
                env[n] = v
                changed = True
        if not changed:
            break
    return env


def _onset(sp, start, clamp):
    p = {}
    for s, w in start.items():
        u = list(s)
        for n, v in clamp.items():
            if n in sp.ix:
                u[sp.ix[n]] = v
        p[tuple(u)] = p.get(tuple(u), 0.0) + w
    return p


def exact_pulse(sp, start, clamp, L=8.0, cap=EXACT_CAP):
    """Exact two-phase pulse. Returns None if the reachable set exceeds `cap`."""
    clamp = {n: v for n, v in clamp.items() if n in sp.ix}
    p1 = _onset(sp, start, clamp)
    r1 = sp.explore(list(p1), set(clamp), cap)
    if r1 is None:
        return None
    o1, pos1, Q1, _ = r1
    v = np.zeros(len(o1))
    for s, w in p1.items():
        v[pos1[s]] = w
    if clamp and L > 0:
        v = expm_multiply(Q1.T * L, v)
    v = np.clip(v, 0, None)
    keep = np.where(v > 1e-15)[0]
    on = {
        k: float(sum(v[i] * sp.value(o1[i], k) for i in keep))
        for k in OUTPUTS
        if sp.value(o1[0], k) is not None
    }
    r2 = sp.explore([o1[i] for i in keep], set(), cap)
    if r2 is None:
        return None
    o2, pos2, Q2, A2 = r2
    p2 = np.zeros(len(o2))
    for i in keep:
        p2[pos2[o1[i]]] += v[i]
    cls = terminal_classes(A2)
    ab = absorption(Q2, cls, p2)
    arrest = late0 = 0.0
    off = {k: 0.0 for k in OUTPUTS}
    for c, m in enumerate(cls):
        pi = stationary(Q2, m)
        prol = np.array([o2[i][sp.ix["Proliferation"]] for i in m])
        if len(m) == 1 and prol[0] == 0:
            arrest += ab[c]
        late0 += ab[c] * float(pi @ (1 - prol))
        if len(m) == 1:
            full = settle_downstream(sp, o2[m[0]])
            for k in OUTPUTS:
                off[k] += ab[c] * full.get(k, np.nan)
        else:
            for k in OUTPUTS:
                if k in sp.ix or k in sp.const:
                    off[k] += ab[c] * float(pi @ np.array([sp.value(o2[i], k) for i in m]))
                else:
                    off[k] = np.nan
    return dict(
        P_arrest_FP=arrest,
        P_prolif0_late=late0,
        absorbed_mass=float(ab.sum() + v.sum() - v[keep].sum()),
        on=on,
        off=off,
        n_phase1=len(o1),
        n_phase2=len(o2),
        method="exact",
    )


# ------------------------------------------------------------------ attractor starts
def attractor_start(sp, row):
    """Exact start distribution on the tracked space for an A2 attractor row.
    Returns (dict state->prob, kind, class_state_set or None, class Proliferation-OFF occupancy or None)."""
    if str(row["type"]).startswith("fixed"):
        full = {n: int(row[n]) for n in NODES}
        for n, v in sp.const.items():
            assert full[n] == v, f"{sp.ctx}: fixed point violates constant {n}"
        s = sp.project(full)
        assert not sp.successors(s, set()), f"{sp.ctx}: A2 fixed point not fixed"
        return {s: 1.0}, "fixed point", None, None
    mem_states, pi, off = complex_class(sp, row)
    return {s: float(p) for s, p in zip(mem_states, pi)}, "complex recurrent class", set(mem_states), off


def complex_class(sp, row):
    osc = [n for n in NODES if str(row[n]).upper() == "X"]
    seeds = set()
    for bits in itertools.product([0, 1], repeat=len(osc)):
        st = {n: (int(row[n]) if n not in osc else 0) for n in NODES}
        st.update(dict(zip(osc, bits)))
        seeds.add(sp.project(st))
    o, pos, Q, A = sp.explore(list(seeds), set())
    cand = [
        m for m in terminal_classes(A) if len(m) > 1 and len({o[i][sp.ix["Proliferation"]] for i in m}) == 2
    ]
    assert len(cand) == 1, f"{sp.ctx}: expected one proliferative complex class, found {len(cand)}"
    m = cand[0]
    pi = stationary(Q, m)
    off = float(pi @ (1 - np.array([o[i][sp.ix["Proliferation"]] for i in m])))
    return [o[i] for i in m], pi, off


# ------------------------------------------------------------------ replicated simulation of the same CTMC
def _enc(sp, s):
    return int(sum(int(b) << i for i, b in enumerate(s)))


def simulate_pulse(sp, start, clamp, class_sets, class_offs, L=8.0, n=40000, seed=0, t_checks=(60.0, 120.0)):
    """Vectorised Gillespie on the tracked space; one independent two-phase replicate.
    class_sets: list of sets of state tuples (known complex recurrent classes in this context)."""
    rng = np.random.default_rng(seed)
    nb = len(sp.nodes)
    one = np.uint64(1)
    clamp = {k: v for k, v in clamp.items() if k in sp.ix}
    keys = list(start)
    w = np.array([start[k] for k in keys])
    w = w / w.sum()
    st = np.array([_enc(sp, k) for k in keys], dtype=np.uint64)[rng.choice(len(keys), size=n, p=w)]
    for k, v in clamp.items():
        b = one << np.uint64(sp.ix[k])
        st = (st | b) if v else (st & ~b)
    cls_arr = [np.array(sorted(_enc(sp, s) for s in cs), dtype=np.uint64) for cs in class_sets]
    const_env = {k: np.bool_(v) for k, v in sp.const.items()}
    upd = [k for k in sp.nodes]

    def enabled(x, frozen):
        env = dict(const_env)
        bits = {k: ((x >> np.uint64(sp.ix[k])) & one).astype(bool) for k in sp.nodes}
        env.update(bits)
        E = np.zeros((len(x), nb), dtype=bool)
        for k in upd:
            if k in frozen:
                continue
            E[:, sp.ix[k]] = np.broadcast_to(eval(VECR[k], {}, env), x.shape) != bits[k]
        return E

    def step(x, t, t_end, frozen, active):
        idx = np.where(active)[0]
        if len(idx) == 0:
            return False
        E = enabled(x[idx], frozen)
        kk = E.sum(1)
        dt = np.full(len(idx), np.inf)
        nz = kk > 0
        dt[nz] = rng.exponential(1.0 / kk[nz])
        go = t[idx] + dt <= t_end
        t[idx[~go]] = t_end
        active[idx[~go]] = False
        if go.any():
            gi = np.where(go)[0]
            r = (rng.random(len(gi)) * kk[gi]).astype(int)
            cum = np.cumsum(E[gi], axis=1)
            pick = (cum > r[:, None]).argmax(1)
            x[idx[gi]] ^= one << pick.astype(np.uint64)
            t[idx[gi]] += dt[gi]
        return True

    # phase 1 (clamped)
    t = np.zeros(n)
    active = np.ones(n, bool) if clamp and L > 0 else np.zeros(n, bool)
    while step(st, t, L, set(clamp), active):
        pass
    # phase 2 (released) until absorption, with status at t_checks
    t = np.zeros(n)
    status = np.full(n, -1)  # -1 transient, 0 arrest FP, 1 prolif FP, 2+c class c
    absorb_t = np.full(n, np.inf)
    prol_bit = np.uint64(sp.ix["Proliferation"])

    def classify(ids):
        x = st[ids]
        E = enabled(x, set())
        fp = E.sum(1) == 0
        s = np.full(len(ids), -1)
        s[fp] = ((x[fp] >> prol_bit) & one).astype(int)
        for c, arr in enumerate(cls_arr):
            inc = (~fp) & np.isin(x, arr)
            s[inc] = 2 + c
        return s

    tmax = max(t_checks)
    active = np.ones(n, bool)
    s0 = classify(np.arange(n))
    status[s0 >= 0] = s0[s0 >= 0]
    absorb_t[s0 >= 0] = 0.0
    active[s0 >= 0] = False
    while active.any():
        step(st, t, tmax, set(), active)
        ids = np.where(active)[0]
        if len(ids):
            s = classify(ids)
            hit = s >= 0
            status[ids[hit]] = s[hit]
            absorb_t[ids[hit]] = t[ids[hit]]
            active[ids[hit]] = False
    out = {}
    for tc in t_checks:
        done = absorb_t <= tc
        out[f"arrest_FP_by_t{int(tc)}"] = float(np.mean(done & (status == 0)))
        out[f"unresolved_at_t{int(tc)}"] = float(np.mean(~done))
    out["P_arrest_FP"] = float(np.mean(status == 0))
    out["P_prolif0_late"] = out["P_arrest_FP"] + sum(
        float(np.mean(status == 2 + c)) * class_offs[c] for c in range(len(cls_arr))
    )
    out["n"] = n
    out["seed"] = seed
    return out
