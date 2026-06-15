"""Ultrametricity analysis of head-head overlap matrices Q_{hh'}.

Tests the spin-glass (Parisi/RSB) prediction that the head distances obey the
strong triangle inequality d(i,k) <= max(d(i,j), d(j,k)).  Operates as pure
post-processing on the Q matrices produced by the correlations stage.

Two complementary views:
  - triple statistics: the Rammal ultrametricity index over head triples,
    with per-head and layer-resolved aggregates.
  - hierarchical clustering: average-linkage tree, cophenetic correlation, and
    the residual D - C whose closest-than-tree outliers expose specific
    head-head correlations riding on top of the hierarchy.
"""

import itertools

import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis


# ── Overlap → distance ────────────────────────────────────────────────

def to_distance(Q, metric):
    """Convert an overlap/divergence matrix into a proper metric distance.

    frob_cosine: chordal distance sqrt(2(1-c)) between unit-normalized
        flattened W_QK (a Euclidean metric).
    *jensen_shannon: the JS distance sqrt(JSD) (the divergence is not a metric).
    """
    Q = np.asarray(Q, dtype=np.float64)
    if metric == "frob_cosine":
        c = np.clip(Q, -1.0, 1.0)
        D = np.sqrt(np.maximum(2.0 * (1.0 - c), 0.0))
    elif metric.endswith("jensen_shannon"):
        D = np.sqrt(np.maximum(Q, 0.0))
    else:
        raise ValueError(f"No distance conversion registered for metric {metric!r}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# ── Triple statistics ─────────────────────────────────────────────────

def _triple_indices(n, sample_cap, rng):
    """All C(n,3) triples when small enough, else a random distinct sample."""
    total = n * (n - 1) * (n - 2) // 6
    if total <= sample_cap:
        return np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64), False
    draw = int(sample_cap * 1.3) + 9
    idx = rng.integers(0, n, size=(draw, 3))
    distinct = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx = idx[distinct][:sample_cap]
    return idx, True


def triple_statistics(D, layers, sample_cap=2_000_000, seed=0, tol=0.05):
    """Compute ultrametricity diagnostics over head triples.

    For each triple the three sides are sorted d1<=d2<=d3 and summarized by the
    Rammal index u = (d3-d2)/(d3-d1) (0 = perfectly ultrametric).  Returns the
    P(u) histogram, the per-head mean u, the u-vs-layer-span profile, the
    tight-pair (sibling) co-occurrence matrix, and triangle-violation rate.
    """
    n = D.shape[0]
    layers = np.asarray(layers)
    rng = np.random.default_rng(seed)
    T, sampled = _triple_indices(n, sample_cap, rng)

    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    # columns map to pairs (a,b), (b,c), (a,c)
    S = np.stack([D[a, b], D[b, c], D[a, c]], axis=1)
    Ss = np.sort(S, axis=1)
    d1, d2, d3 = Ss[:, 0], Ss[:, 1], Ss[:, 2]

    span = d3 - d1
    u = np.where(span > 0, (d3 - d2) / span, 0.0)
    tri_margin = d1 + d2 - d3                       # >= 0 for a metric
    tri_viol = tri_margin < -1e-9

    # per-head mean u
    head_u_sum = np.zeros(n)
    head_u_cnt = np.zeros(n)
    flat_heads = T.reshape(-1)
    np.add.at(head_u_sum, flat_heads, np.repeat(u, 3))
    np.add.at(head_u_cnt, flat_heads, 1)
    per_head_u = np.divide(head_u_sum, head_u_cnt,
                           out=np.full(n, np.nan), where=head_u_cnt > 0)

    # layer structure per triple
    tl = layers[T]
    layer_span = tl.max(axis=1) - tl.min(axis=1)
    n_distinct = np.array([len(np.unique(r)) for r in tl]) if len(tl) else np.array([])

    # u vs layer span profile
    max_span = int(layer_span.max()) if len(layer_span) else 0
    span_grid = np.arange(0, max_span + 1)
    span_mean = np.full(span_grid.shape, np.nan)
    span_cnt = np.zeros(span_grid.shape)
    for s in span_grid:
        m = layer_span == s
        span_cnt[s] = m.sum()
        if m.any():
            span_mean[s] = u[m].mean()

    # tight-pair (sibling) co-occurrence among near-ultrametric triples
    cooccur = np.zeros((n, n))
    near = u < tol
    pmin = np.argmin(S, axis=1)
    pair_a = np.where(pmin == 0, a, np.where(pmin == 1, b, a))
    pair_b = np.where(pmin == 0, b, np.where(pmin == 1, c, c))
    sa, sb = pair_a[near], pair_b[near]
    np.add.at(cooccur, (sa, sb), 1.0)
    np.add.at(cooccur, (sb, sa), 1.0)

    u_edges = np.linspace(0.0, 1.0, 51)
    u_counts, _ = np.histogram(u, bins=u_edges)

    return {
        "n_triples": int(len(u)),
        "sampled": bool(sampled),
        "mean_u": float(u.mean()),
        "median_u": float(np.median(u)),
        "frac_ultrametric": float(near.mean()),
        "triangle_violation_frac": float(tri_viol.mean()),
        "u_hist_edges": u_edges,
        "u_hist_counts": u_counts,
        "per_head_u": per_head_u,
        "span_grid": span_grid,
        "span_mean_u": span_mean,
        "span_count": span_cnt,
        "stratum_mean_u": {int(k): float(u[n_distinct == k].mean())
                           for k in (1, 2, 3) if (n_distinct == k).any()},
        "cooccur": cooccur,
    }


# ── Four-point: additivity and hyperbolicity ──────────────────────────

def _quad_indices(n, sample_cap, rng):
    total = n * (n - 1) * (n - 2) * (n - 3) // 24
    if total <= sample_cap:
        return np.array(list(itertools.combinations(range(n), 4)), dtype=np.int64), False
    draw = int(sample_cap * 1.4) + 16
    idx = rng.integers(0, n, size=(draw, 4))
    ok = np.ones(len(idx), bool)
    for p, q in itertools.combinations(range(4), 2):
        ok &= idx[:, p] != idx[:, q]
    return idx[ok][:sample_cap], True


def quadruple_statistics(D, sample_cap=1_000_000, seed=0, tol=0.05):
    """Four-point diagnostics: additive-tree fit and Gromov delta-hyperbolicity.

    For each quadruple the three pair-sums are sorted m1<=m2<=m3.  The additive
    (four-point) condition requires m3==m2 — the natural sibling of the three-
    point ultrametric test.  u4=(m3-m2)/(m3-m1) is the additivity index (0 =
    perfectly additive), and delta=(m3-m2)/2 is the four-point Gromov delta;
    normalized by the diameter it measures how tree-like (hyperbolic) D is.
    """
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    Q, sampled = _quad_indices(n, sample_cap, rng)
    a, b, c, d = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]

    s = np.stack([D[a, b] + D[c, d], D[a, c] + D[b, d], D[a, d] + D[b, c]], axis=1)
    s.sort(axis=1)
    m1, m2, m3 = s[:, 0], s[:, 1], s[:, 2]
    span = m3 - m1
    u4 = np.where(span > 0, (m3 - m2) / span, 0.0)
    delta = 0.5 * (m3 - m2)
    diam = float(D.max())
    delta_rel = delta / diam if diam > 0 else delta

    edges = np.linspace(0.0, 1.0, 51)
    counts, _ = np.histogram(u4, bins=edges)
    return {
        "n_quads": int(len(u4)),
        "sampled": bool(sampled),
        "mean_u4": float(u4.mean()),
        "median_u4": float(np.median(u4)),
        "frac_additive": float((u4 < tol).mean()),
        "u4_hist_edges": edges,
        "u4_hist_counts": counts,
        "delta_rel_mean": float(delta_rel.mean()),
        "delta_rel_p95": float(np.percentile(delta_rel, 95)),
        "diameter": diam,
    }


# ── Hierarchical clustering view ──────────────────────────────────────

def cluster_analysis(D, keys, top_k=50):
    """Average-linkage tree, cophenetic correlation, and tree residuals.

    The cophenetic correlation is a global ultrametricity scalar.  The most
    negative residuals D-C (pairs much closer than the tree predicts) are the
    specific head-head correlations layered on top of the hierarchy.
    """
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    coph_corr, coph = cophenet(Z, condensed)

    resid = condensed - coph
    order = np.argsort(resid)                       # most negative first
    iu, ju = np.triu_indices(D.shape[0], k=1)
    layers = np.array([k[0] for k in keys])
    heads = np.array([k[1] for k in keys])

    residual_pairs = []
    for p in order[:top_k]:
        i, j = int(iu[p]), int(ju[p])
        residual_pairs.append({
            "i": i, "j": j,
            "layer_1": int(layers[i]), "head_1": int(heads[i]),
            "layer_2": int(layers[j]), "head_2": int(heads[j]),
            "delta_layer": int(abs(layers[i] - layers[j])),
            "distance": float(D[i, j]),
            "cophenetic": float(coph[p]),
            "residual": float(resid[p]),
        })

    return {
        "linkage": Z,
        "cophenetic_corr": float(coph_corr),
        "residual_pairs": residual_pairs,
    }


# ── Is it the wrong tree, or extra structure on top? ──────────────────

def _gini(x):
    x = np.sort(np.abs(x))
    s = x.sum()
    if s == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def residual_decomposition(D, Z, top_k=20):
    """Decompose D = C + R against the average-linkage cophenetic distances C.

    Characterizes the residual R: low-rank (a few global factors) vs sparse
    (specific pairwise affinities) vs noise.  Returns the residual eigenvalue
    spectrum, an effective rank, and concentration statistics.
    """
    condensed = squareform(D, checks=False)
    C = squareform(cophenet(Z, condensed)[1])
    R = D - C

    res_energy_frac = float(np.linalg.norm(R) / np.linalg.norm(D))
    w, V = np.linalg.eigh(R)
    w2 = w ** 2
    energy = float(w2.sum())
    eff_rank = float(energy ** 2 / (w2 ** 2).sum()) if energy > 0 else 0.0
    order = np.argsort(-np.abs(w))           # by |eigenvalue|, descending
    eigvals_sorted = w[order]
    top_eigs = eigvals_sorted[:top_k]
    top_eigvecs = V[:, order[:top_k]]        # (n_heads, top_k), head-space modes
    frac_top10 = float(w2[order[:10]].sum() / energy) if energy > 0 else 0.0

    off = R[np.triu_indices(R.shape[0], k=1)]
    gini = _gini(off)
    kurt = float(kurtosis(off))

    n = R.shape[0]
    if eff_rank < max(3.0, 0.05 * n) and frac_top10 > 0.5:
        verdict = "low-rank"
    elif gini > 0.6 or kurt > 5.0:
        verdict = "sparse"
    else:
        verdict = "noise-like"

    return {
        "res_energy_frac": res_energy_frac,
        "eff_rank": eff_rank,
        "frac_top10": frac_top10,
        "top_eigs": top_eigs,
        "top_eigvecs": top_eigvecs,
        "eigvals_sorted": eigvals_sorted,
        "gini": gini,
        "kurtosis": kurt,
        "verdict": verdict,
    }


def embedding_analysis(D, max_dims=12, top_k=20):
    """Classical MDS (PCoA): how well does a flat Euclidean geometry fit D?

    Double-centers -1/2 D^2 and reads its spectrum.  Large negative-eigenvalue
    energy flags a non-Euclidean (tree/hyperbolic) metric; variance captured by
    the top few positive dimensions says whether a low-dim flat embedding works.
    Also returns the leading principal coordinates (head-space modes).
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    wv, V = np.linalg.eigh(B)
    order = np.argsort(wv)[::-1]              # descending
    w = wv[order]
    top_eigvecs = V[:, order[:top_k]]        # (n_heads, top_k) principal coords
    pos = w[w > 0]
    abs_energy = float(np.abs(w).sum())
    neg_energy_frac = float(np.abs(w[w < 0]).sum() / abs_energy) if abs_energy > 0 else 0.0
    pos_sum = float(pos.sum()) if len(pos) else 0.0
    var_cum = (np.cumsum(pos) / pos_sum) if pos_sum > 0 else np.zeros(0)
    eff_dim = float(pos_sum ** 2 / (pos ** 2).sum()) if len(pos) else 0.0
    return {
        "eigenvalues": w[:max_dims],
        "top_eigvecs": top_eigvecs,
        "top_eigs": w[:top_k],
        "neg_energy_frac": neg_energy_frac,
        "var_frac_2d": float(var_cum[1]) if len(var_cum) > 1 else float(var_cum[-1] if len(var_cum) else 0.0),
        "var_frac_3d": float(var_cum[2]) if len(var_cum) > 2 else float(var_cum[-1] if len(var_cum) else 0.0),
        "eff_dim": eff_dim,
    }


def linkage_comparison(D, methods=("single", "average", "complete", "ward")):
    """Cophenetic correlation across linkage rules.

    Average linkage is greedy; comparing methods (single = subdominant
    ultrametric) separates 'the tree is suboptimal' from 'D is not a tree'.
    """
    condensed = squareform(D, checks=False)
    out = {}
    for meth in methods:
        Z = linkage(condensed, method=meth)
        out[meth] = float(cophenet(Z, condensed)[0])
    return out


# ── Untrained null ensemble ───────────────────────────────────────────

def null_distance_samples(d_head, d_model, metric, n_heads=96, seed=0, max_pairs=8000):
    """Off-diagonal distances for random-init heads (the untrained baseline).

    W_Q, W_K ~ iid Gaussian; per-head W_QK = W_Q W_Kᵀ.  The Frobenius peak lands
    at √2 (cos=0): generic high-dim vectors are near-orthogonal.  Used to separate
    the aligned tail of trained models from this null.
    """
    rng = np.random.default_rng(seed)
    V = np.stack([
        (rng.standard_normal((d_head, d_model)) @
         rng.standard_normal((d_head, d_model)).T).ravel()
        for _ in range(n_heads)
    ])
    iu = np.triu_indices(n_heads, k=1)
    if metric == "frob_cosine":
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        return to_distance(Vn @ Vn.T, "frob_cosine")[iu]
    if metric.endswith("jensen_shannon"):
        from transformer_analysis.pair_metrics import histogram_jensen_shannon
        pairs = list(zip(*iu))
        if len(pairs) > max_pairs:
            sel = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in sel]
        return np.array([np.sqrt(max(histogram_jensen_shannon(V[i], V[j]), 0.0))
                         for i, j in pairs])
    return None


# ── Reference ensemble ────────────────────────────────────────────────

def synthetic_ultrametric(levels=4, noise=0.0, seed=0):
    """A balanced binary ultrametric tree for calibration.

    Leaf distance is set by the depth of the lowest common ancestor, so at
    noise=0 the matrix is perfectly ultrametric (cophenetic correlation 1).
    Noise perturbs the distances toward a realistic, imperfect tree.  Returns
    (D, keys) with keys as (block, index) labels.
    """
    n = 2 ** levels
    bits = (np.arange(n)[:, None] >> np.arange(levels - 1, -1, -1)) & 1
    leading = np.cumprod(bits[:, None, :] == bits[None, :, :], axis=2)
    shared = leading.sum(axis=2)                    # shared prefix length per pair
    D = (levels - shared) / levels
    np.fill_diagonal(D, 0.0)

    if noise > 0:
        rng = np.random.default_rng(seed)
        P = rng.uniform(-1.0, 1.0, size=(n, n))
        D = np.clip(D + 0.5 * noise * 0.5 * (P + P.T), 0.0, None)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

    sub = 2 ** (levels // 2)
    keys = [(i // sub, i % sub) for i in range(n)]
    return D, keys
