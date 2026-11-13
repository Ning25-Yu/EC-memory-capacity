"""Core utilities: load the fixed EC Boolean network (bnet), build MaBoSS models, run cMaBoSS simulations."""

import os
import tempfile

import cmaboss
import numpy as np
import pandas as pd

from paths import DATA

HERE = os.path.dirname(os.path.abspath(__file__))
BNET = os.path.join(DATA, "EC_network.bnet")
INPUTS = ["KLRB1", "IRF5", "STAT6", "TSC22D3"]
PHENO = ["Proliferation", "Apoptosis", "EMT", "Invasion", "Migration", "Metastasis"]


def load_rules(path=BNET):
    rules = {}
    for l in open(path):
        if "," in l:
            n, r = l.split(",", 1)
            rules[n.strip()] = r.strip()
    return rules


RULES = load_rules()
NODES = sorted(RULES)
MOLECULAR = [n for n in NODES if n not in PHENO]  # 31 molecular nodes (incl. 4 inputs)
INTERNAL_MOL = [n for n in MOLECULAR if n not in INPUTS]  # 27 non-input molecular nodes


def ctx_name(bits):  # bits ordered KLRB1, IRF5, STAT6, TSC22D3
    return "C" + "".join(str(int(b)) for b in bits)


def ctx_bits(name):
    return dict(zip(INPUTS, [int(c) for c in name[1:]]))


def bnd_string(rules=RULES, mutations=None):
    """MaBoSS bnd; mutations: dict node->0/1 (clamped by rule), consistent with GINsim export semantics."""
    mutations = mutations or {}
    s = []
    for n in NODES:
        r = rules[n]
        if n in mutations:
            r = "1" if mutations[n] else "0"
        r = r.replace("!", " NOT ").replace("&", " AND ").replace("|", " OR ")
        s.append(
            f"Node {n} {{\n  logic = ({r});\n  rate_up = @logic ? $u_{n} : 0;\n  rate_down = @logic ? 0 : $d_{n};\n}}\n"
        )
    return "\n".join(s)


def cfg_string(
    init=None,
    up=None,
    down=None,
    max_time=50,
    time_tick=0.5,
    sample_count=10000,
    seed=12345,
    outputs=PHENO,
    thread_count=4,
):
    """init: dict node->P(1) (default 0.5); up/down: dict node->rate (default 1)."""
    init = init or {}
    up = up or {}
    down = down or {}
    s = []
    for n in NODES:
        s.append(f"$u_{n} = {up.get(n,1.0):.6g};\n$d_{n} = {down.get(n,1.0):.6g};")
    for n in NODES:
        p = float(init.get(n, 0.5))
        p = min(max(p, 0.0), 1.0)
        s.append(f"[{n}].istate = {p:.6g}[1], {1-p:.6g}[0];")
    for n in NODES:
        s.append(f"{n}.is_internal = {0 if n in outputs else 1};")
    s.append(
        f"time_tick = {time_tick};\nmax_time = {max_time};\nsample_count = {sample_count};\n"
        f"discrete_time = 0;\nuse_physrandgen = 0;\nseed_pseudorandom = {seed};\nthread_count = {thread_count};\n"
        f"statdist_traj_count = 0;\nstatdist_cluster_threshold = 1;\ndisplay_traj = 0;"
    )
    return "\n".join(s)


def run(bnd, cfg):
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "m.bnd")
        c = os.path.join(d, "m.cfg")
        open(b, "w").write(bnd)
        open(c, "w").write(cfg)
        sim = cmaboss.MaBoSSSim(network=b, config=c)
        res = sim.run(only_last_state=True)
    return res


def last_dist(res):
    """Final joint-state distribution over non-internal nodes: pandas Series state->prob."""
    arr, t, states = res.get_last_probtraj()
    return pd.Series(np.asarray(arr).ravel(), index=list(states))


def last_marg(res):
    arr, t, nodes = res.get_last_nodes_probtraj()
    return dict(zip(nodes, np.asarray(arr).ravel()))


def node_marginals(dist, nodes=PHENO):
    m = {n: 0.0 for n in nodes}
    for st, p in dist.items():
        on = set(st.split(" -- ")) if st != "<nil>" else set()
        for n in nodes:
            if n in on:
                m[n] += p
    return m


def simulate(init=None, up=None, down=None, mutations=None, outputs=PHENO, **kw):
    res = run(bnd_string(mutations=mutations), cfg_string(init=init, up=up, down=down, outputs=outputs, **kw))
    dist = last_dist(res)
    m = {n: 0.0 for n in outputs}
    m.update(last_marg(res))
    return dist, m


from sklearn.mixture import GaussianMixture


def normalise(x):
    x = np.asarray(x, float)
    g1 = GaussianMixture(1, random_state=1).fit(x[:, None])
    g2 = GaussianMixture(2, random_state=1).fit(x[:, None])
    mu = g2.means_.ravel()
    sd = np.sqrt(g2.covariances_.ravel())
    w = g2.weights_
    bi = (
        np.sqrt(w[0] * w[1]) * abs(mu[0] - mu[1]) / np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2)
    )  # bimodality index (Wang 2009)
    if g1.bic(x[:, None]) - g2.bic(x[:, None]) > 10 and bi > 1.5 and min(w) > 0.1:
        post = g2.predict_proba(x[:, None])[:, int(np.argmax(mu))]
        kind = "bimodal-GMM"
    else:
        med = np.median(x)
        mad = 1.4826 * np.median(abs(x - med)) + 1e-6
        post = 1 / (1 + np.exp(-(x - med) / mad))
        kind = "logistic-robust-z"
    return np.clip(post, 0.01, 0.99), kind, bi
