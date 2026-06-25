"""
Minimize (1 - xi) using a Kolmogorov-Arnold Network (KAN), where xi is
Chatterjee's correlation coefficient from scipy.stats.chatterjeexi.

WHY THIS APPROACH
------------------
Chatterjee's xi is a *rank-based* statistic: it sorts y by the order induced
by x and counts sign changes in consecutive ranks. That sorting/counting
operation has zero gradient almost everywhere, so you cannot backprop
through `scipy.stats.chatterjeexi` directly. Two complementary strategies
are used here:

  1. TRUE OBJECTIVE, GRADIENT-FREE: optimize the KAN's parameters with CMA-ES
     (via the `cma` package, falling back to scipy's differential_evolution)
     using loss(theta) = 1 - xi(KAN(x; theta), y) computed exactly with scipy.

  2. WARM START, GRADIENT-BASED: first fit the KAN with plain full-batch
     gradient descent on MSE (which *is* differentiable, via analytic
     spline-basis gradients computed by hand -- no autodiff framework
     required). This gives CMA-ES a much better starting point than random
     init, since "minimize 1 - xi" and "minimize MSE" are loosely aligned
     when trying to recover a functional relationship.

KAN IMPLEMENTATION
-------------------
This is a compact, from-scratch KAN (no external KAN library / torch
dependency), following the standard formulation: every edge between an
input node and a hidden/output node carries its own learnable 1-D function

    phi(x) = w_base * silu(x) + w_spline * sum_i c_i * B_i(x)

where B_i are cubic B-spline basis functions on a fixed knot grid, and the
c_i (spline coefficients), w_base, w_spline are all learnable. Node values
in a layer are the sum of the incoming edge functions (this is the standard
KAN layer formulation from Liu et al., 2024, "KAN: Kolmogorov-Arnold
Networks").

USAGE
-----
    python kan_xi_min.py

Edit `build_dataset()` to swap in your own (x, y) data, and edit
`KAN_LAYER_SIZES` to change network width/depth.
"""

import numpy as np
from scipy.stats import chatterjeexi
from scipy.optimize import differential_evolution

try:
    import cma
    HAVE_CMA = True
except ImportError:
    HAVE_CMA = False


# ----------------------------------------------------------------------------
# B-spline basis
# ----------------------------------------------------------------------------

def make_knots(grid_min, grid_max, n_grid, degree):
    """Open uniform knot vector with `degree`-fold repeated boundary knots."""
    step = (grid_max - grid_min) / n_grid
    inner = np.linspace(grid_min, grid_max, n_grid + 1)
    knots = np.r_[
        np.repeat(grid_min, degree),
        inner,
        np.repeat(grid_max, degree),
    ]
    return knots


def bspline_basis(x, knots, degree):
    """
    Cox-de Boor recursion, vectorized over x.
    Returns (len(x), n_basis) matrix where n_basis = len(knots) - degree - 1.
    """
    x = np.clip(x, knots[degree], knots[-degree - 1] - 1e-12)
    M = len(knots)
    n_basis = M - degree - 1
    width = n_basis + degree  # number of degree-0 intervals we track
    B = np.zeros((degree + 1, width, len(x)))

    for i in range(width):
        B[0, i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(float)

    for d in range(1, degree + 1):
        for i in range(width - d):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            term1 = (x - knots[i]) / denom1 * B[d - 1, i] if denom1 > 0 else 0.0
            term2 = (knots[i + d + 1] - x) / denom2 * B[d - 1, i + 1] if denom2 > 0 else 0.0
            B[d, i] = term1 + term2

    return B[degree, :n_basis].T  # (N, n_basis)


def bspline_basis_grad(x, knots, degree):
    """
    Derivative of each basis function w.r.t. x, via the standard
    B-spline derivative identity:
        B'_{i,d}(x) = d/(t_{i+d}-t_i) * B_{i,d-1}(x)
                    - d/(t_{i+d+1}-t_{i+1}) * B_{i+1,d-1}(x)
    """
    Blow = bspline_basis_all_degrees(x, knots, degree - 1)  # (N, n_basis_low)
    M = len(knots)
    n_basis = M - degree - 1
    dB = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]
        t1 = degree / denom1 * Blow[:, i] if denom1 > 0 and i < Blow.shape[1] else 0.0
        t2 = degree / denom2 * Blow[:, i + 1] if denom2 > 0 and (i + 1) < Blow.shape[1] else 0.0
        dB[:, i] = t1 - t2
    return dB


def bspline_basis_all_degrees(x, knots, degree):
    """Helper: basis at a possibly-lower degree (used for derivative calc)."""
    if degree < 0:
        return np.zeros((len(x), len(knots) - 1))
    return bspline_basis(x, knots, degree)


def silu(x):
    return x / (1.0 + np.exp(-x))


def silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s + x * s * (1 - s)


# ----------------------------------------------------------------------------
# KAN layer and network
# ----------------------------------------------------------------------------

class KANLayer:
    """
    One KAN layer: in_dim inputs -> out_dim outputs.
    Each of the in_dim * out_dim edges has its own spline + base function.
    """

    def __init__(self, in_dim, out_dim, degree=3, n_grid=5,
                 grid_min=-2.0, grid_max=2.0, rng=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.n_grid = n_grid
        self.knots = make_knots(grid_min, grid_max, n_grid, degree)
        self.n_basis = len(self.knots) - degree - 1
        rng = rng or np.random.default_rng()

        # Parameters per edge (in_dim x out_dim edges)
        self.coef = rng.normal(scale=0.1, size=(in_dim, out_dim, self.n_basis))
        self.w_base = rng.normal(loc=1.0, scale=0.1, size=(in_dim, out_dim))
        self.w_spline = rng.normal(loc=1.0, scale=0.1, size=(in_dim, out_dim))

    def n_params(self):
        return self.coef.size + self.w_base.size + self.w_spline.size

    def get_params(self):
        return np.concatenate([self.coef.ravel(), self.w_base.ravel(), self.w_spline.ravel()])

    def set_params(self, theta):
        nc = self.coef.size
        nb = self.w_base.size
        self.coef = theta[:nc].reshape(self.coef.shape)
        self.w_base = theta[nc:nc + nb].reshape(self.w_base.shape)
        self.w_spline = theta[nc + nb:nc + nb + self.w_base.size].reshape(self.w_spline.shape)

    def forward(self, x, cache=False):
        """
        x: (N, in_dim)
        returns: (N, out_dim), and optionally a cache dict for backward()
        """
        N = x.shape[0]
        out = np.zeros((N, self.out_dim))
        basis_cache = {}
        for i in range(self.in_dim):
            xi_col = x[:, i]
            B = bspline_basis(xi_col, self.knots, self.degree)       # (N, n_basis)
            base = silu(xi_col)                                     # (N,)
            basis_cache[i] = (B, base)
            for j in range(self.out_dim):
                spline_val = B @ self.coef[i, j]                     # (N,)
                out[:, j] += self.w_base[i, j] * base + self.w_spline[i, j] * spline_val
        if cache:
            return out, basis_cache
        return out

    def backward(self, x, basis_cache, grad_out):
        """
        Compute gradients of loss w.r.t. this layer's params and w.r.t. x,
        given grad_out = dLoss/d(layer_output), shape (N, out_dim).
        Returns (grad_theta, grad_x) where grad_theta matches get_params() order.
        """
        N = x.shape[0]
        grad_coef = np.zeros_like(self.coef)
        grad_w_base = np.zeros_like(self.w_base)
        grad_w_spline = np.zeros_like(self.w_spline)
        grad_x = np.zeros_like(x)

        for i in range(self.in_dim):
            xi_col = x[:, i]
            B, base = basis_cache[i]
            dbase = silu_grad(xi_col)
            dB = bspline_basis_grad(xi_col, self.knots, self.degree)
            for j in range(self.out_dim):
                g = grad_out[:, j]  # (N,)
                spline_val = B @ self.coef[i, j]

                grad_w_base[i, j] = np.sum(g * base)
                grad_w_spline[i, j] = np.sum(g * spline_val)
                grad_coef[i, j] = (g * self.w_spline[i, j]) @ B

                dphi_dx = (self.w_base[i, j] * dbase
                           + self.w_spline[i, j] * (dB @ self.coef[i, j]))
                grad_x[:, i] += g * dphi_dx

        grad_theta = np.concatenate([grad_coef.ravel(), grad_w_base.ravel(), grad_w_spline.ravel()])
        return grad_theta, grad_x


class KAN:
    """Stack of KANLayers, e.g. sizes=[1, 4, 1] for a 1->4->1 network."""

    def __init__(self, sizes, degree=3, n_grid=5, grid_min=-2.0, grid_max=2.0, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [
            KANLayer(sizes[k], sizes[k + 1], degree=degree, n_grid=n_grid,
                     grid_min=grid_min, grid_max=grid_max, rng=rng)
            for k in range(len(sizes) - 1)
        ]
        self._sizes_per_layer = [layer.n_params() for layer in self.layers]

    def n_params(self):
        return sum(self._sizes_per_layer)

    def get_params(self):
        return np.concatenate([layer.get_params() for layer in self.layers])

    def set_params(self, theta):
        idx = 0
        for layer, n in zip(self.layers, self._sizes_per_layer):
            layer.set_params(theta[idx:idx + n])
            idx += n

    def forward(self, x, cache=False):
        """x: (N, in_dim) -> (N, out_dim_last). x is internally normalized per-call to [-1.5, 1.5]-ish range by caller."""
        caches = []
        h = x
        for layer in self.layers:
            if cache:
                h, c = layer.forward(h, cache=True)
                caches.append(c)
            else:
                h = layer.forward(h, cache=False)
        if cache:
            return h, caches
        return h

    def backward(self, x, caches, grad_out):
        """Full backward pass; returns flat grad_theta matching get_params() order."""
        # Recompute intermediate layer inputs (cheap: just re-run forward layer by layer)
        inputs = [x]
        h = x
        for layer in self.layers[:-1]:
            h = layer.forward(h, cache=False)
            inputs.append(h)

        grad_theta_parts = [None] * len(self.layers)
        g = grad_out
        for k in reversed(range(len(self.layers))):
            layer = self.layers[k]
            grad_theta_parts[k], g = layer.backward(inputs[k], caches[k], g)
        return np.concatenate(grad_theta_parts)


# ----------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------

def predict_flat(kan, x_col):
    """x_col: (N,) -> (N,) network output, assuming 1-d in/out KAN."""
    out = kan.forward(x_col.reshape(-1, 1), cache=False)
    return out.ravel()


def xi_loss(theta, kan, x_col, y):
    """loss = 1 - xi(KAN(x), y). Lower is better. Not differentiable (rank-based)."""
    kan.set_params(theta)
    pred = predict_flat(kan, x_col)
    if np.std(pred) < 1e-10:
        return 1.0  # degenerate constant output -> worst case
    xi = chatterjeexi(pred, y).statistic
    return 1.0 - xi


def mse_loss_and_grad(theta, kan, x_col, y):
    """Differentiable warm-start objective: MSE between KAN(x) and y."""
    kan.set_params(theta)
    pred, caches = kan.forward(x_col.reshape(-1, 1), cache=True)
    pred = pred.ravel()
    resid = pred - y
    loss = np.mean(resid ** 2)
    grad_out = (2.0 / len(y) * resid).reshape(-1, 1)
    grad_theta = kan.backward(x_col.reshape(-1, 1), caches, grad_out)
    return loss, grad_theta


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def warm_start_gd(kan, x_col, y, n_steps=300, lr=0.05, verbose=True):
    """Plain full-batch gradient descent on MSE to get a sane initialization."""
    theta = kan.get_params()
    for step in range(n_steps):
        loss, grad = mse_loss_and_grad(theta, kan, x_col, y)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 5.0:  # crude gradient clipping for stability
            grad = grad / grad_norm * 5.0
        theta = theta - lr * grad
        if verbose and (step % 50 == 0 or step == n_steps - 1):
            kan.set_params(theta)
            xi_now = chatterjeexi(predict_flat(kan, x_col), y).statistic
            print(f"  [warm-start GD] step {step:4d}  MSE={loss:.4f}  xi={xi_now:.4f}")
    kan.set_params(theta)
    return theta


def optimize_xi_cma(kan, x_col, y, theta0, sigma0=0.3, max_iter=200, popsize=None, verbose=True):
    """Gradient-free optimization of 1 - xi via CMA-ES."""
    objective = lambda th: xi_loss(th, kan, x_col, y)
    if HAVE_CMA:
        opts = {"maxiter": max_iter, "verbose": -9}
        if popsize is not None:
            opts["popsize"] = popsize
        es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)
        gen = 0
        while not es.stop():
            solutions = es.ask()
            losses = [objective(s) for s in solutions]
            es.tell(solutions, losses)
            gen += 1
            if verbose and gen % 10 == 0:
                print(f"  [CMA-ES] gen {gen:4d}  best (1-xi)={es.result.fbest:.4f}  xi={1 - es.result.fbest:.4f}")
        theta_best = es.result.xbest
    else:
        if verbose:
            print("  'cma' package not found; falling back to scipy.optimize.differential_evolution")
        bounds = [(t - 1.5, t + 1.5) for t in theta0]
        result = differential_evolution(
            objective, bounds, maxiter=max_iter, popsize=15,
            tol=1e-6, seed=0, polish=False, updating="deferred", workers=-1,
        )
        theta_best = result.x
        if verbose:
            print(f"  [DE] best (1-xi)={result.fun:.4f}  xi={1 - result.fun:.4f}")

    kan.set_params(theta_best)
    return theta_best


# ----------------------------------------------------------------------------
# Dataset (replace with your own data)
# ----------------------------------------------------------------------------

def build_dataset(n=300, seed=0):
    """
    Example: a noisy, non-monotonic relationship y = sin(2x) + noise on x in
    [-2, 2]. Pearson/Spearman struggle with non-monotonic relationships; xi
    is designed to detect them, which is why it's an interesting KAN target.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = np.sin(2 * x) + rng.normal(scale=0.15, size=n)
    return x, y


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

KAN_LAYER_SIZES = [1, 6, 1]  # 1 input -> 6 hidden edges-worth -> 1 output

if __name__ == "__main__":
    x, y = build_dataset()

    kan = KAN(KAN_LAYER_SIZES, degree=3, n_grid=5, grid_min=-2.5, grid_max=2.5, seed=0)
    print(f"KAN parameter count: {kan.n_params()}")

    pred0 = predict_flat(kan, x)
    xi0 = chatterjeexi(pred0, y).statistic
    print(f"Initial xi(KAN(x), y) = {xi0:.4f}  (random init)")

    print("\n=== Phase 1: gradient-descent warm start (minimizing MSE) ===")
    theta_warm = warm_start_gd(kan, x, y, n_steps=300, lr=0.05)

    pred_warm = predict_flat(kan, x)
    xi_warm = chatterjeexi(pred_warm, y).statistic
    print(f"\nAfter warm start: xi = {xi_warm:.4f}")

    print("\n=== Phase 2: CMA-ES directly minimizing (1 - xi) ===")
    theta_final = optimize_xi_cma(kan, x, y, theta0=theta_warm, sigma0=0.3, max_iter=150)

    pred_final = predict_flat(kan, x)
    res_final = chatterjeexi(pred_final, y)
    print(f"\nFinal xi(KAN(x), y) = {res_final.statistic:.4f}  (p-value = {res_final.pvalue:.3e})")
    print(f"Final 1 - xi (loss) = {1 - res_final.statistic:.4f}")
