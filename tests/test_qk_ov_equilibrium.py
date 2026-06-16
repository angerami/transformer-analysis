"""Identities and sanity checks for the QK/OV equilibrium observables.

Mirrors the test list in docs/qk_ov_equilibrium_spec.md. Uses synthetic float64
weights so the orthogonality identities hold to tight tolerance.
"""
import numpy as np
import torch

from transformer_analysis.qk_ov_equilibrium import (
    build_operators,
    density_matrix,
    commutator_scalars,
    irreversibility_scalars,
    rank_deficiency_terms,
    compute_layer_equilibrium,
    _sym_anti,
    _fro,
)

N_HEADS, HEAD_DIM, D_MODEL = 3, 4, 10  # d_model > 2*head_dim -> rho has a null space


def _rand_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    def f():
        return torch.randn(N_HEADS, HEAD_DIM, D_MODEL, generator=g, dtype=torch.float64)
    return f(), f(), f(), f()


def _operators(seed=0):
    W_Q, W_K, W_V, W_O = _rand_weights(seed)
    W_QK, W_OV = build_operators(W_Q, W_K, W_V, W_O)
    S, A = _sym_anti(W_QK)
    rho = density_matrix(W_OV)
    return W_QK, W_OV, S, A, rho


# 1. orthogonal symmetric/antisymmetric split of W_QK
def test_orthogonal_split_wqk():
    W_QK, _, S, A, _ = _operators()
    torch.testing.assert_close(_fro(W_QK) ** 2, _fro(S) ** 2 + _fro(A) ** 2)
    # Frobenius-orthogonality Tr(S^T A) = 0
    cross = (S * A).sum(dim=(-2, -1))
    torch.testing.assert_close(cross, torch.zeros_like(cross), atol=1e-9, rtol=0)


# 2. orthogonal commutator split
def test_commutator_orthogonal_split():
    W_QK, _, S, A, rho = _operators()
    sc = commutator_scalars(W_QK, S, A, rho)
    torch.testing.assert_close(
        sc["norm_C"] ** 2, sc["norm_C_anti"] ** 2 + sc["norm_C_sym"] ** 2
    )


# 3. A has zero diagonal and zero element-mean
def test_antisymmetric_part_zero_diag_and_mean():
    _, _, _, A, _ = _operators()
    diag = A.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-12, rtol=0)
    mean = A.mean(dim=(-2, -1))
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-12, rtol=0)


# 4. rho_OV is symmetric, PSD, unit-trace
def test_rho_is_valid_density_matrix():
    _, _, _, _, rho = _operators()
    torch.testing.assert_close(rho, rho.transpose(-2, -1))
    tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    torch.testing.assert_close(tr, torch.ones_like(tr))
    assert torch.linalg.eigvalsh(rho).min() >= -1e-9


# 5. C_anti = [rho, S] and C_sym = [rho, A]
def test_commutator_symmetry_pieces():
    W_QK, _, S, A, rho = _operators()
    C = torch.bmm(rho, W_QK) - torch.bmm(W_QK, rho)
    C_anti = 0.5 * (C - C.transpose(-2, -1))
    C_sym = 0.5 * (C + C.transpose(-2, -1))
    torch.testing.assert_close(C_anti, torch.bmm(rho, S) - torch.bmm(S, rho))
    torch.testing.assert_close(C_sym, torch.bmm(rho, A) - torch.bmm(A, rho))


# 6. a normal-but-antisymmetric head: f_A = 1, nu = 0 (the two are independent)
def test_pure_rotation_head():
    g = torch.Generator().manual_seed(1)
    M = torch.randn(N_HEADS, D_MODEL, D_MODEL, generator=g, dtype=torch.float64)
    W_QK = M - M.transpose(-2, -1)  # antisymmetric -> normal
    S, A = _sym_anti(W_QK)
    sc = irreversibility_scalars(W_QK, S, A)
    torch.testing.assert_close(sc["f_A"], torch.ones(N_HEADS, dtype=torch.float64))
    torch.testing.assert_close(
        sc["nu"], torch.zeros(N_HEADS, dtype=torch.float64), atol=1e-9, rtol=0
    )


# 7. a symmetric head: A = 0, f_A = 0, eig(S) = eig(W_QK)
def test_symmetric_head():
    g = torch.Generator().manual_seed(2)
    M = torch.randn(N_HEADS, D_MODEL, D_MODEL, generator=g, dtype=torch.float64)
    W_QK = M + M.transpose(-2, -1)  # symmetric
    S, A = _sym_anti(W_QK)
    torch.testing.assert_close(A, torch.zeros_like(A), atol=1e-12, rtol=0)
    sc = irreversibility_scalars(W_QK, S, A)
    torch.testing.assert_close(
        sc["f_A"], torch.zeros(N_HEADS, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        torch.linalg.eigvalsh(S), torch.linalg.eigvalsh(W_QK)
    )


# bonus: rank-deficiency terms partition ||[rho, S]||^2 = ||C_anti||^2,
# and within = ||[rho_supp, P S P]||^2 <= ||P S P||^2 * (spectral spread)
def test_rank_deficiency_terms_sum_to_canti():
    W_QK, _, S, A, rho = _operators()
    within, cross, ssp = rank_deficiency_terms(rho, S)
    sc = commutator_scalars(W_QK, S, A, rho)
    torch.testing.assert_close(within + cross, sc["norm_C_anti"] ** 2)
    # support-block S energy is a sub-sum of the full ||S||^2
    assert (ssp <= _fro(S) ** 2 + 1e-9).all()


# guard: undefined Theta emits NaN rather than dividing by ~0
def test_theta_nan_when_norm_zero():
    g = torch.Generator().manual_seed(3)
    M = torch.randn(N_HEADS, D_MODEL, D_MODEL, generator=g, dtype=torch.float64)
    W_QK = M + M.transpose(-2, -1)  # symmetric -> A = 0 -> Theta_curr undefined
    S, A = _sym_anti(W_QK)
    _, _, _, _, rho = _operators(seed=3)
    sc = commutator_scalars(W_QK, S, A, rho)
    assert torch.isnan(sc["Theta_curr"]).all()


# end-to-end: shapes and finiteness of the per-layer driver
def test_compute_layer_equilibrium_outputs():
    W_Q, W_K, W_V, W_O = _rand_weights(7)
    scalars, arrays = compute_layer_equilibrium(W_Q, W_K, W_V, W_O)
    assert len(scalars) == N_HEADS
    for row in scalars:
        for key in ("Theta", "Theta_full", "f_A", "nu", "S_spectral_entropy",
                    "A_stable_rank", "S_mean", "A_skew"):
            assert key in row
        assert np.isfinite(row["Theta_full"])
    assert arrays["lambda_S"].shape == (N_HEADS, D_MODEL)
    assert arrays["omega_A"].shape == (N_HEADS, D_MODEL)
    assert arrays["P_w_S"].shape[0] == N_HEADS
