"""Dense conditional KIV: exact Algorithms 1/2, no low-rank approximation."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist

GRID = (1e-4, 1e-3, 1e-2, .1, 1.)


def kernel(a, b, bandwidth):
    # Explicit product kernel prevents 50 X coordinates from diluting binary T/I.
    return np.exp(-cdist(a[:, 1:], b[:, 1:], 'sqeuclidean')/(2*bandwidth)
                  -np.square(a[:, 0, None]-b[None, :, 0])/2)


def solve(matrix, rhs):
    jitter = 1e-10*max(float(np.mean(np.diag(matrix))), 1.)
    adjusted = (matrix+matrix.T)/2 + jitter*np.eye(len(matrix))
    result = cho_solve(cho_factor(adjusted, lower=True, check_finite=True), rhs)
    residual = np.linalg.norm(adjusted@result-rhs)/max(np.linalg.norm(rhs), 1e-12)
    assert residual < 1e-5, residual
    return result, jitter, residual


def run(arrays, seed):
    x, t, z, y, xq = (np.asarray(arrays[k], float) for k in
                       ('X_context', 'treatment', 'instrument', 'outcome', 'X_query'))
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    xn, xqn = (x-xm)/xs, (xq-xm)/xs
    bandwidth = float(np.median(pdist(xn, 'sqeuclidean')))
    v, q = np.column_stack([t, xn]), np.column_stack([z, xn])
    permutation = np.random.default_rng(seed).permutation(len(x))
    one, two = permutation[:len(x)//2], permutation[len(x)//2:]
    n, m = len(one), len(two)
    kv = kernel(v[one], v[one], bandwidth)
    kv12 = kernel(v[one], v[two], bandwidth)
    kq = kernel(q[one], q[one], bandwidth)
    kq12 = kernel(q[one], q[two], bandwidth)
    gamma_list, stage1 = [], []
    for lam in GRID:
        gamma, jitter, residual = solve(kq+n*lam*np.eye(n), kq12)
        loss = float((m-2*np.sum(kv12*gamma)+np.sum(gamma*(kv@gamma)))/m)
        gamma_list.append(gamma)
        stage1.append(dict(lambda_value=lam, validation_embedding_loss=loss,
                           jitter=jitter, solve_relative_residual=float(residual)))
    selected = int(np.argmin([r['validation_embedding_loss'] for r in stage1]))
    w = kv@gamma_list[selected]
    ym, ys = y[two].mean(), max(y[two].std(), 1e-6)
    alphas, stage2 = [], []
    for xi in GRID:
        matrix = w@w.T + m*xi*kv
        alpha, jitter, residual = solve(matrix, w@((y[two]-ym)/ys))
        loss = float(np.mean(np.square(y[one]-(kv@alpha*ys+ym))))
        alphas.append(alpha)
        stage2.append(dict(xi=xi, validation_outcome_mse=loss, jitter=jitter,
                           solve_relative_residual=float(residual)))
    chosen = int(np.argmin([r['validation_outcome_mse'] for r in stage2]))
    alpha = alphas[chosen]
    h = {arm: kernel(v[one], np.column_stack([np.full(len(xq), arm), xqn]), bandwidth).T@alpha*ys+ym
         for arm in (0, 1)}
    matrix = w@w.T + m*GRID[chosen]*kv
    metadata = dict(algorithm='Singh et al. KIV Algorithms 1/2; conditional V=[T,X], Q=[I,X]',
                    structural_inputs='[T,X]', instrument_inputs='[I,X]', final_query='X',
                    kernel='product Gaussian; standardized 50d X plus binary coordinate bandwidth=1',
                    x_bandwidth_squared=bandwidth, lambda_grid=GRID, xi_grid=GRID,
                    selected_lambda=GRID[selected], selected_xi=GRID[chosen],
                    stage1_selection=stage1, stage2_selection=stage2,
                    stage1_rows=one.tolist(), stage2_rows=two.tolist(),
                    matrix_dimension=n, stage2_matrix_condition=float(np.linalg.cond(matrix)),
                    maximum_dense_matrix_bytes=n*n*8, approximation='none; numerical jitter reported',
                    validation='training-context only; paper Algorithm 2 observed-Y loss, not true tau',
                    cross_fitting_folds=0, split_stages=2)
    return dict(tau_hat=h[1]-h[0], h0=h[0], h1=h[1]), metadata
