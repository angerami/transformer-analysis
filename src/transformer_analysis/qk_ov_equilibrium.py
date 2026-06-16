"""QK/OV equilibrium observables.

Treats the QK circuit as an interaction (W_QK, split into a symmetric
Hamiltonian S and an antisymmetric circulating part A) and the OV circuit as a
state (the density matrix rho_OV = W_OV W_OV^T / Tr). Per head, measures how far
the head is from equilibrium via the commutator C = [rho_OV, W_QK], plus the
QK-internal irreversibility (f_A, nu) and the spectra/element statistics of S
and A.

Everything is a property of the weights — no activations. All operators act on
the d_model residual stream so the commutators are well defined:

    W_QK = W_Q^T W_K     (d_model x d_model, rank <= head_dim, not symmetric)
    W_OV = W_O^T W_V     (d_model x d_model, the residual->residual OV map)

See docs/qk_ov_equilibrium_spec.md.
"""

import numpy as np
import torch

from transformer_analysis.head_metrics import (
    stats_config_default,
    singular_value_metrics,
    normality_metrics,
    make_weight_bins,
)

EPS = 1e-12
DEFAULT_W_BINS = make_weight_bins("fixed")


# ── Operator construction (batched over heads) ────────────────────────

def _eigvalsh(M):
    """Symmetric eigenvalues, with CPU fallback (eigh is unimplemented on MPS)."""
    try:
        return torch.linalg.eigvalsh(M)
    except (NotImplementedError, RuntimeError):
        return torch.linalg.eigvalsh(M.cpu()).to(M.device)

def _eigh(M):
    """Symmetric eigendecomposition, with CPU fallback (unimplemented on MPS)."""
    try:
        return torch.linalg.eigh(M)
    except (NotImplementedError, RuntimeError):
        w, V = torch.linalg.eigh(M.cpu())
        return w.to(M.device), V.to(M.device)

def _fro(X):
    """Frobenius norm over the last two dims: (..., d, d) -> (...)."""
    return torch.sqrt((X * X).sum(dim=(-2, -1)))

def _sym_anti(M):
    Mt = M.transpose(-2, -1)
    return 0.5 * (M + Mt), 0.5 * (M - Mt)

def build_operators(W_Q_h, W_K_h, W_V_h, W_O_h):
    """Per-head residual-stream operators from (n_heads, head_dim, d_model) weights.

    Returns W_QK, W_OV, each (n_heads, d_model, d_model).
    """
    W_QK = torch.bmm(W_Q_h.transpose(1, 2), W_K_h)
    W_OV = torch.bmm(W_O_h.transpose(1, 2), W_V_h)
    return W_QK, W_OV

def density_matrix(W_OV):
    """rho_OV = W_OV W_OV^T / Tr(W_OV W_OV^T): symmetric, PSD, unit-trace."""
    G = torch.bmm(W_OV, W_OV.transpose(-2, -1))
    tr = G.diagonal(dim1=-2, dim2=-1).sum(-1)
    return G / tr.clamp(min=EPS)[:, None, None]


# ── Scalar observables ────────────────────────────────────────────────

def _safe_div(num, den):
    out = num / den.clamp(min=EPS)
    return torch.where(den > EPS, out, torch.full_like(out, float("nan")))

def commutator_scalars(W_QK, S, A, rho):
    """Cross-circuit equilibrium scalars (family 1).

    C = [rho, W_QK]; because rho is symmetric the split is Frobenius-orthogonal:
        C_anti = (C - C^T)/2 = [rho, S]   (antisymmetric, energy mismatch)
        C_sym  = (C + C^T)/2 = [rho, A]   (symmetric, current alignment)
    """
    C = torch.bmm(rho, W_QK) - torch.bmm(W_QK, rho)
    C_anti, C_sym = (0.5 * (C - C.transpose(-2, -1)),
                     0.5 * (C + C.transpose(-2, -1)))

    n_rho, n_S, n_A, n_QK = _fro(rho), _fro(S), _fro(A), _fro(W_QK)
    n_C, n_Ca, n_Cs = _fro(C), _fro(C_anti), _fro(C_sym)

    return {
        "Theta": _safe_div(n_Ca, n_rho * n_S),
        "Theta_curr": _safe_div(n_Cs, n_rho * n_A),
        "Theta_full": _safe_div(n_C, n_rho * n_QK),
        "norm_C": n_C, "norm_C_anti": n_Ca, "norm_C_sym": n_Cs,
        "norm_rho_OV": n_rho,
    }

def irreversibility_scalars(W_QK, S, A):
    """QK-internal irreversibility (family 2), weight-only.

    f_A: antisymmetric (non-reciprocal) fraction of the interaction.
    nu:  non-normality ||[W_QK, W_QK^T]||_F; nu=0 iff W_QK is normal.

    The symmetric/antisymmetric parts make nu purely an S-A interaction:
    [W_QK, W_QK^T] = [S+A, S-A] = 2[A, S], so nu = 2||[A, S]||_F (the S^2 and
    A^2 terms cancel). It therefore needs *both* S and A present and
    non-commuting. nu_align = ||[A, S]|| / (||A|| ||S||) factors out the
    magnitudes (0 iff A and S commute) and is the part independent of f_A:
    f_A->1 means S->0, so nu itself -> 0 (the pure-rotation limit) even though
    f_A is maximal.
    """
    n_QK, n_S, n_A = _fro(W_QK), _fro(S), _fro(A)
    WWt = torch.bmm(W_QK, W_QK.transpose(-2, -1))
    WtW = torch.bmm(W_QK.transpose(-2, -1), W_QK)
    nu = _fro(WWt - WtW)
    return {
        "f_A": _safe_div(n_A ** 2, n_QK ** 2),
        "nu": nu,
        "nu_norm": _safe_div(nu, n_QK ** 2),
        "nu_align": _safe_div(nu, 2 * n_A * n_S),
        "norm_W_QK": n_QK, "norm_S": n_S, "norm_A": n_A,
    }


def rank_deficiency_terms(rho, S):
    """Split ||[rho, S]||^2 by rho's support (diagnostic, family 1).

        within = sum_{a,b in support} (lam_a - lam_b)^2 |S_ab|^2
        cross  = 2 sum_{a in support, j in null} lam_a^2 |S_aj|^2

    distinguishing out-of-equilibrium *within* the directions OV uses from
    leakage into OV's unused (null) directions. S_ab is S in rho's eigenbasis.

    Also returns ssp = sum_{a,b in support} |S_ab|^2 = ||P S P||^2, the energy
    of S restricted to rho's support — the normalizer for the support-restricted
    equilibrium Theta_supp. (within = ||[rho_supp, P S P]||^2, since on the
    support block [rho, S]_ab = (lam_a - lam_b) S_ab.)
    """
    lam, U = _eigh(rho)                                   # ascending
    T = torch.bmm(torch.bmm(U.transpose(-2, -1), S), U)   # S in rho eigenbasis
    supp = lam > (1e-6 * lam.max(dim=-1, keepdim=True).values)
    null = ~supp
    supp_block = supp[:, :, None] & supp[:, None, :]

    dlam2 = (lam[:, :, None] - lam[:, None, :]) ** 2
    within = (dlam2 * T ** 2 * supp_block).sum((-2, -1))
    cross = 2.0 * ((lam[:, :, None] ** 2) * T ** 2 *
                   (supp[:, :, None] & null[:, None, :])).sum((-2, -1))
    ssp = (T ** 2 * supp_block).sum((-2, -1))
    return within, cross, ssp


# ── Spectra and element statistics of S and A (family 3) ──────────────

def _spectral_stats(spectrum_desc, prefix):
    """Run the existing SV-metric battery on a descending-sorted spectrum."""
    h = {}
    for fn in singular_value_metrics.values():
        fn(h, spectrum_desc)
    return {f"{prefix}_{k}": float(v) for k, v in h.items()}

def _element_stats(M, prefix, w_bins):
    """Element moments + histogram-based normality, mirroring head_analyzer."""
    x = M.ravel()
    out = {f"{prefix}_{k}": float(fn(x)) for k, fn in stats_config_default.items()}

    P_w, _ = np.histogram(x, bins=w_bins, density=True)
    centers = 0.5 * (w_bins[:-1] + w_bins[1:])
    h = {"P_w": P_w, "mean": out[f"{prefix}_mean"], "std": out[f"{prefix}_std"]}
    for fn in normality_metrics.values():
        fn(h, centers)
    for k in ("entropy", "fit_mu", "fit_sigma", "kl_vs_empirical_normal"):
        if k in h:
            out[f"{prefix}_{k}"] = float(h[k])
    return out, P_w


# ── Top-level: one layer at a time ────────────────────────────────────

def compute_layer_equilibrium(W_Q_h, W_K_h, W_V_h, W_O_h,
                              with_decomp=True, w_bins=None):
    """All equilibrium observables for one layer's heads.

    Args:
        W_Q_h, W_K_h, W_V_h, W_O_h: (n_heads, head_dim, d_model) tensors.
        with_decomp: also compute the rho-support rank-deficiency split.
        w_bins: bin edges for S/A element histograms (default DEFAULT_W_BINS).

    Returns:
        (scalars, arrays) where
          scalars: list of per-head dicts (Theta, f_A, nu, norms, S_*/A_* stats).
          arrays:  dict name -> (n_heads, ...) ndarray
                   (lambda_S signed spectrum, omega_A rotation rates,
                    P_w_S / P_w_A element histograms).
    """
    if w_bins is None:
        w_bins = DEFAULT_W_BINS

    W_QK, W_OV = build_operators(W_Q_h, W_K_h, W_V_h, W_O_h)
    S, A = _sym_anti(W_QK)
    rho = density_matrix(W_OV)

    scal = {}
    scal.update(commutator_scalars(W_QK, S, A, rho))
    scal.update(irreversibility_scalars(W_QK, S, A))
    if with_decomp:
        within, cross, ssp = rank_deficiency_terms(rho, S)
        norm_PSP = ssp.sqrt()
        scal["decomp_within"], scal["decomp_cross"] = within, cross
        scal["norm_PSP"] = norm_PSP
        # support-restricted equilibrium: does rho commute with S inside the
        # subspace OV actually uses? (strips the trivial support<->null leakage)
        scal["Theta_supp"] = _safe_div(within.sqrt(), _fro(rho) * norm_PSP)

    # Spectra: signed eigenvalues of S; rotation rates omega_k of A (= |eig(A)|,
    # equal pairs) via sqrt of the eigenvalues of A A^T.
    lam_S = _eigvalsh(S)
    omega = torch.sqrt(_eigvalsh(
        torch.bmm(A, A.transpose(-2, -1))).clamp(min=0.0))

    scal = {k: v.detach().cpu().numpy() for k, v in scal.items()}
    lam_S_np = lam_S.detach().cpu().numpy()
    omega_np = omega.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    A_np = A.detach().cpu().numpy()

    n_heads = W_QK.shape[0]
    scalars, P_w_S, P_w_A = [], [], []
    for h in range(n_heads):
        row = {k: float(scal[k][h]) for k in scal}
        # |lambda(S)| and omega(A), descending, for the SV-metric battery
        row.update(_spectral_stats(np.sort(np.abs(lam_S_np[h]))[::-1], "S"))
        row.update(_spectral_stats(np.sort(omega_np[h])[::-1], "A"))
        s_stats, ps = _element_stats(S_np[h], "S", w_bins)
        a_stats, pa = _element_stats(A_np[h], "A", w_bins)
        row.update(s_stats)
        row.update(a_stats)
        scalars.append(row)
        P_w_S.append(ps)
        P_w_A.append(pa)

    arrays = {
        "lambda_S": lam_S_np,
        "omega_A": omega_np,
        "P_w_S": np.stack(P_w_S),
        "P_w_A": np.stack(P_w_A),
    }
    return scalars, arrays
