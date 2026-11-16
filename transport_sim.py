"""
transport_sim.py

Simulation study for
  "Causal inference with interference when the exposure mapping is induced
   by an uncertain transport model"

Design (bipartite interference):
  * J pollution sources (facilities) are the intervention units; n receptor
    locations are the outcome units. A_j = 1 means facility j installs a
    control that removes a fraction r of its emissions.
  * Pollution is transported by a steady advection-diffusion-decay PDE,
        -K Lap c + v . grad c + lam c = q,
    solved daily on a 32 x 32 grid with the day's wind. In scenario A the
    wind is spatially uniform (equal to the wind recorded at a single
    meteorological station); in scenario B it veers across the domain, so
    no candidate transport model is correctly specified.
  * The analyst observes daily PM at a sparse monitor network under baseline
    (pre-intervention) emissions, plus daily facility emissions and the
    station wind. Candidate transport models are FITTED to these data.
  * The health outcome at receptor i depends on the intervention-induced
    exposure change e*_i(A) = mean_t [T_t (r s_t o A)]_i.

Candidate transport models (all fitted to the monitor data):
  AD-PDE      advection-diffusion PDE, (K, lam, wind multiplier) estimated
  D-PDE       diffusion-decay PDE without advection, (K, lam) estimated
  Kernel      isotropic exponential source-receptor kernel, range estimated
  LSO         learned spectral operator: a wind-conditioned translation-
              invariant Green's function represented by truncated Fourier
              modes (the linear spectral-convolution layer of a Fourier
              neural operator), estimated by generalized ridge regression

All code is numpy/scipy only.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------
# Grid and PDE
# ----------------------------------------------------------------------------
N = 32
LDOM = 10.0
H = LDOM / N
_xc = (np.arange(N) + 0.5) * H
GX, GY = np.meshgrid(_xc, _xc, indexing="ij")
GRID_XY = np.column_stack([GX.ravel(), GY.ravel()])      # flat index = ix*N + iy
_IDX = np.arange(N * N).reshape(N, N)

TRUE_K, TRUE_LAM = 0.15, 0.25
R_CONTROL = 0.8          # fraction of emissions removed by a control
BACKGROUND = 3.0
SIGMA_OBS = 0.35         # monitor measurement noise (SD)
VEER = 0.6               # scenario B: wind veers by +/- VEER radians across y


def build_L(K, lam, vx, vy):
    """Sparse matrix of -K Lap c + v.grad c + lam c (first-order upwind
    advection, zero Dirichlet boundary). vx, vy: scalars or (N, N) arrays."""
    vx = np.broadcast_to(np.asarray(vx, float), (N, N))
    vy = np.broadcast_to(np.asarray(vy, float), (N, N))
    diag = 4 * K / H**2 + lam + np.abs(vx) / H + np.abs(vy) / H
    rows, cols, vals = [_IDX.ravel()], [_IDX.ravel()], [diag.ravel()]
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ix = np.arange(N)[:, None] + dx + 0 * np.arange(N)[None, :]
        iy = np.arange(N)[None, :] + dy + 0 * np.arange(N)[:, None]
        ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
        c = np.full((N, N), -K / H**2)
        if dx == -1:
            c = c - np.maximum(vx, 0) / H
        if dx == 1:
            c = c - np.maximum(-vx, 0) / H
        if dy == -1:
            c = c - np.maximum(vy, 0) / H
        if dy == 1:
            c = c - np.maximum(-vy, 0) / H
        rows.append(_IDX[ok])
        cols.append(_IDX[ix[ok], iy[ok]])
        vals.append(c[ok])
    return sp.csc_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N * N, N * N))


# ----------------------------------------------------------------------------
# Geography (fixed across replicates) and policies
# ----------------------------------------------------------------------------
def make_geography(seed=2024, J=36, n_mon=40, n_mon_train=30, n_rec=500):
    rng = np.random.default_rng(seed)
    interior = _IDX[2:N - 2, 2:N - 2].ravel()
    fac = rng.choice(interior, J, replace=False)
    rest = np.setdiff1d(np.arange(N * N), fac)
    pick = rng.choice(rest, n_mon + n_rec, replace=False)
    mon, rec = pick[:n_mon], pick[n_mon:]
    s_base = np.exp(rng.normal(0, 0.6, J))
    fxy, rxy = GRID_XY[fac], GRID_XY[rec]
    # receptor covariate (urbanicity proxy) correlated with proximity to sources
    d_rf = np.linalg.norm(rxy[:, None, :] - fxy[None, :, :], axis=2)
    prox = np.log((s_base[None, :] * np.exp(-d_rf / 1.5)).sum(1))
    Xcov = (prox - prox.mean()) / prox.std() + rng.normal(0, 0.5, n_rec)
    Xcov = (Xcov - Xcov.mean()) / Xcov.std()
    # policies (facility masks)
    box = (fxy[:, 0] > 3) & (fxy[:, 0] < 7) & (fxy[:, 1] > 3) & (fxy[:, 1] < 7)
    point = np.zeros(J, bool)
    point[np.where(box)[0][np.argmax(s_base[box])]] = True
    policies = {
        "Regional": box,
        "Point source": point,
        "Upwind": fxy[:, 0] < 3.5,
        "Downwind": fxy[:, 0] > 6.5,
    }
    return dict(fac=fac, mon=mon, mon_train=np.arange(n_mon_train),
                mon_test=np.arange(n_mon_train, n_mon), rec=rec,
                s_base=s_base, fxy=fxy, rxy=rxy, mxy=GRID_XY[mon],
                Xcov=Xcov, policies=policies, J=J)


def draw_days(rng, geo, D=80, theta_bar=0.25, kappa=4.0, lsd=0.3):
    theta = theta_bar + rng.vonmises(0.0, kappa, D)
    speed = np.exp(rng.normal(0, lsd, D))
    ux, uy = speed * np.cos(theta), speed * np.sin(theta)
    s = geo["s_base"][None, :] * np.exp(rng.normal(-0.045, 0.3, (D, geo["J"])))
    return dict(D=D, ux=ux, uy=uy, theta=theta, speed=speed, s=s)


def _rhs(geo, w):
    """Point-source density on the grid for facility weights w (J,)."""
    q = np.zeros(N * N)
    q[geo["fac"]] = w / H**2
    return q


def _policy_weights(geo, s_t):
    return [R_CONTROL * s_t * geo["policies"][p] for p in geo["policies"]]


def solve_days(geo, days, K, lam, ux, uy, extra_weights=True):
    """For each day: baseline field (grid) and policy exposure fields.
    ux, uy: (D,) station winds or (D, N, N) wind fields."""
    D = days["D"]
    base = np.empty((D, N * N))
    npol = len(geo["policies"])
    pol = np.zeros((npol, N * N))
    for t in range(D):
        lu = spla.splu(build_L(K, lam, ux[t], uy[t]))
        cols = [_rhs(geo, days["s"][t])]
        if extra_weights:
            cols += [_rhs(geo, w) for w in _policy_weights(geo, days["s"][t])]
        sol = lu.solve(np.column_stack(cols))
        base[t] = sol[:, 0]
        if extra_weights:
            pol += sol[:, 1:].T
    return base, pol / D


def truth(geo, days, scenario):
    if scenario == "A":
        ux, uy = days["ux"], days["uy"]
    else:
        ang = days["theta"][:, None, None] + VEER * (GY[None] - 5) / 5
        ux = days["speed"][:, None, None] * np.cos(ang)
        uy = days["speed"][:, None, None] * np.sin(ang)
    base, pol = solve_days(geo, days, TRUE_K, TRUE_LAM, ux, uy)
    e_star = {p: pol[i][geo["rec"]] for i, p in enumerate(geo["policies"])}
    return base, e_star


def observe(rng, geo, base):
    return BACKGROUND + base[:, geo["mon"]] + rng.normal(0, SIGMA_OBS, base[:, geo["mon"]].shape)


# ----------------------------------------------------------------------------
# Candidate transport models
# ----------------------------------------------------------------------------
def _profile(f, y):
    """OLS of y on [1, f]; returns (b, gamma, residual)."""
    X = np.column_stack([np.ones_like(f), f])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[0], coef[1], y - X @ coef


class PDEModel:
    """advect=True: AD-PDE; advect=False: D-PDE (no advection). Parameters are
    shape parameters (see _fields); gamma (emission-to-concentration scale) and
    the background b are profiled out by least squares."""

    def __init__(self, advect):
        self.advect = advect
        self.name = "AD-PDE" if advect else "D-PDE"

    def _fields(self, phi, geo, days, policies=False):
        """Scale-free parameterization: dividing the PDE by lam gives
        -ell^2 Lap c + (alpha/lam) v.grad c + c = q/lam; the overall scale 1/lam
        is absorbed by the profiled multiplier gamma. phi = (log ell^2, log alpha/lam)
        for AD-PDE and phi = (log ell^2,) for D-PDE."""
        K = np.exp(phi[0])
        if self.advect:
            a = np.exp(phi[1])
            return solve_days(geo, days, K, 1.0, a * days["ux"], a * days["uy"], policies)
        lu = spla.splu(build_L(K, 1.0, 0.0, 0.0))
        S = np.column_stack([_rhs(geo, days["s"][t]) for t in range(days["D"])])
        base = lu.solve(S).T
        pol = None
        if policies:
            npol = len(geo["policies"])
            W = np.zeros((N * N, npol))
            for t in range(days["D"]):
                for i, w in enumerate(_policy_weights(geo, days["s"][t])):
                    W[:, i] += _rhs(geo, w)
            pol = lu.solve(W / days["D"]).T
        return base, pol

    def fit(self, geo, days, y, train_days):
        mt = geo["mon_train"]
        sub = dict(days, D=len(train_days), ux=days["ux"][train_days],
                   uy=days["uy"][train_days], s=days["s"][train_days])
        yv = y[np.ix_(train_days, mt)].ravel()

        def resid(phi):
            base, _ = self._fields(phi, geo, sub)
            f = base[:, geo["mon"][mt]].ravel()
            return _profile(f, yv)[2]

        phi0 = np.log([1.0, 2.0])[: 2 if self.advect else 1]
        res = least_squares(resid, phi0, method="trf", diff_step=1e-3,
                            max_nfev=80, x_scale=1.0)
        self.phi = res.x
        base, _ = self._fields(self.phi, geo, sub)
        f = base[:, geo["mon"][mt]].ravel()
        self.b, self.gamma, r = _profile(f, yv)
        dof = len(yv) - len(self.phi) - 2
        self.sigma2 = r @ r / dof
        # joint Gauss-Newton covariance of theta = (phi, b, gamma)
        cols = []
        for k in range(len(self.phi)):
            dphi = np.zeros_like(self.phi); dphi[k] = 1e-3
            fk = self._fields(self.phi + dphi, geo, sub)[0][:, geo["mon"][mt]].ravel()
            cols.append(self.gamma * (fk - f) / 1e-3)
        Jf = np.column_stack(cols + [np.ones_like(f), f])
        self.cov_full = self.sigma2 * np.linalg.inv(Jf.T @ Jf)
        self.cov = self.cov_full[: len(self.phi), : len(self.phi)]
        return self

    def predict_and_expose(self, geo, days, phi=None):
        phi = self.phi if phi is None else phi
        base, pol = self._fields(phi, geo, days, policies=True)
        # re-profile gamma for drawn phi on the training data is done by caller
        return base, pol

    def exposures(self, geo, days, phi=None, gamma=None):
        base, pol = self.predict_and_expose(geo, days, phi)
        g = self.gamma if gamma is None else gamma
        e = {p: g * pol[i][geo["rec"]] for i, p in enumerate(geo["policies"])}
        pred = self.b + g * base[:, geo["mon"]]
        return e, pred

    def draws(self, geo, days, y, train_days, S, rng):
        """Approximate posterior draws of (phi, b, gamma) from the joint
        Gauss-Newton normal approximation (within-model uncertainty)."""
        th0 = np.concatenate([self.phi, [self.b, self.gamma]])
        out = []
        for th in rng.multivariate_normal(th0, self.cov_full, S):
            phi, g = th[: len(self.phi)], th[-1]
            _, pol = self._fields(phi, geo, days, policies=True)
            out.append({p: g * pol[i][geo["rec"]] for i, p in enumerate(geo["policies"])})
        return out


class KernelModel:
    """c_t(x) = b + gamma * sum_j s_jt exp(-|x - x_j| / ell)."""
    name = "Kernel"

    def _basis(self, ell, xy, fxy, s):
        Kx = np.exp(-np.linalg.norm(xy[:, None] - fxy[None], axis=2) / ell)  # (nloc, J)
        return s @ Kx.T, Kx                                                  # (D, nloc)

    def fit(self, geo, days, y, train_days):
        mt = geo["mon_train"]
        yv = y[np.ix_(train_days, mt)].ravel()
        st = days["s"][train_days]

        def resid(phi):
            f, _ = self._basis(np.exp(phi[0]), geo["mxy"][mt], geo["fxy"], st)
            return _profile(f.ravel(), yv)[2]

        res = least_squares(resid, [np.log(1.0)], diff_step=1e-4)
        self.phi = res.x
        f, _ = self._basis(np.exp(self.phi[0]), geo["mxy"][mt], geo["fxy"], st)
        f = f.ravel()
        self.b, self.gamma, r = _profile(f, yv)
        self.sigma2 = r @ r / (len(yv) - 3)
        fk, _ = self._basis(np.exp(self.phi[0] + 1e-4), geo["mxy"][mt], geo["fxy"], st)
        Jf = np.column_stack([self.gamma * (fk.ravel() - f) / 1e-4, np.ones_like(f), f])
        self.cov_full = self.sigma2 * np.linalg.inv(Jf.T @ Jf)
        self.cov = self.cov_full[:1, :1]
        return self

    def _expo(self, geo, days, ell, gamma):
        _, Kx = self._basis(ell, geo["rxy"], geo["fxy"], days["s"][:1])
        sbar = days["s"].mean(0)
        return {p: gamma * Kx @ (R_CONTROL * sbar * m) for p, m in geo["policies"].items()}

    def exposures(self, geo, days):
        ell = np.exp(self.phi[0])
        f, _ = self._basis(ell, geo["mxy"], geo["fxy"], days["s"])
        return self._expo(geo, days, ell, self.gamma), self.b + self.gamma * f

    def draws(self, geo, days, y, train_days, S, rng):
        th0 = np.array([self.phi[0], self.b, self.gamma])
        return [self._expo(geo, days, np.exp(th[0]), th[2])
                for th in rng.multivariate_normal(th0, self.cov_full, S)]


class SpectralOperator:
    """Learned spectral operator (LSO).

    c_t(x) = b + sum_j s_jt G(x - x_j; u_t),
    G(r; u) = sum_{k in K} sum_{p in {1, u_x, u_y}} u_p [a_kp cos(k.r) + b_kp sin(k.r)],
    with k on a truncated lattice 2*pi/P * (k1, k2), |k1|, |k2| <= kmax
    (half-plane, P = 2 * domain to avoid wrap-around). This is the linear
    spectral-convolution layer of a Fourier neural operator conditioned on
    wind; coefficients are estimated by generalized ridge regression with a
    roughness penalty and GCV-selected penalty level.
    """
    name = "LSO"

    def __init__(self, kmax=8, period=2 * LDOM, k0=1.0):
        k1, k2 = np.meshgrid(np.arange(0, kmax + 1), np.arange(-kmax, kmax + 1), indexing="ij")
        k1, k2 = k1.ravel(), k2.ravel()
        keep = (k1 > 0) | ((k1 == 0) & (k2 >= 0))
        self.kv = 2 * np.pi / period * np.column_stack([k1[keep], k2[keep]])
        self.nk = len(self.kv)
        kn2 = (self.kv**2).sum(1)
        w = 1 + (kn2 / k0**2) ** 2
        # feature order: [cos x (1,ux,uy), sin x (1,ux,uy)] per mode; drop sin of k=0
        wfull = np.concatenate([np.repeat(w, 3), np.repeat(w, 3)])
        self.keep_cols = np.ones(6 * self.nk, bool)
        z = np.where(kn2 == 0)[0][0]
        self.keep_cols[3 * self.nk + 3 * z: 3 * self.nk + 3 * z + 3] = False
        self.pen = wfull[self.keep_cols]

    def _features(self, xy, fxy, W, ux, uy):
        """xy (n,2), W (D,J) facility weights, ux/uy (D,). Returns (D, n, p)."""
        ph_f = fxy @ self.kv.T                      # (J, nk)
        Cf, Sf = W @ np.cos(ph_f), W @ np.sin(ph_f)  # (D, nk)
        ph_x = xy @ self.kv.T                       # (n, nk)
        cx, sx = np.cos(ph_x), np.sin(ph_x)
        C = cx[None] * Cf[:, None] + sx[None] * Sf[:, None]   # (D, n, nk)
        Sn = sx[None] * Cf[:, None] - cx[None] * Sf[:, None]
        U = np.stack([np.ones_like(ux), ux, uy], 1)            # (D, 3)
        FC = C[..., None] * U[:, None, None, :]                # (D, n, nk, 3)
        FS = Sn[..., None] * U[:, None, None, :]
        F = np.concatenate([FC.reshape(*C.shape[:2], -1), FS.reshape(*C.shape[:2], -1)], 2)
        return F[..., self.keep_cols]

    def fit(self, geo, days, y, train_days):
        mt = geo["mon_train"]
        F = self._features(geo["mxy"][mt], geo["fxy"], days["s"][train_days],
                           days["ux"][train_days], days["uy"][train_days])
        X = F.reshape(-1, F.shape[-1])
        yv = y[np.ix_(train_days, mt)].ravel()
        Xc, yc = X - X.mean(0), yv - yv.mean()
        Ws = 1 / np.sqrt(self.pen)
        U_, d, Vt = np.linalg.svd(Xc * Ws[None], full_matrices=False)
        uty = U_.T @ yc
        n = len(yv)
        best = None
        for lam in 10.0 ** np.arange(-4, 4.01, 0.25):
            f = d**2 / (d**2 + lam)
            rss = ((yc - U_ @ (f * uty)) ** 2).sum()
            gcv = rss / (1 - f.sum() / n) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, lam, f.sum())
        _, lam, edf = best
        self.lam, self.edf = lam, edf
        A = Xc.T @ Xc + lam * np.diag(self.pen)
        self.theta = np.linalg.solve(A, Xc.T @ yc)
        self.b = yv.mean() - X.mean(0) @ self.theta
        r = yv - self.b - X @ self.theta
        self.sigma2 = r @ r / (n - edf - 1)
        self.cov = self.sigma2 * np.linalg.inv(A)
        return self

    def _features_mean(self, xy, fxy, W, ux, uy):
        """Day-averaged features (n, p) without materializing (D, n, p)."""
        ph_f = fxy @ self.kv.T
        Cf, Sf = W @ np.cos(ph_f), W @ np.sin(ph_f)            # (D, nk)
        U = np.stack([np.ones_like(ux), ux, uy], 1)             # (D, 3)
        CU = np.einsum("tk,tp->kp", Cf, U) / len(ux)            # (nk, 3)
        SU = np.einsum("tk,tp->kp", Sf, U) / len(ux)
        ph_x = xy @ self.kv.T
        cx, sx = np.cos(ph_x), np.sin(ph_x)                     # (n, nk)
        FC = cx[:, :, None] * CU[None] + sx[:, :, None] * SU[None]
        FS = sx[:, :, None] * CU[None] - cx[:, :, None] * SU[None]
        F = np.concatenate([FC.reshape(len(xy), -1), FS.reshape(len(xy), -1)], 1)
        return F[:, self.keep_cols]

    def _expo_design(self, geo, days):
        return {p: self._features_mean(geo["rxy"], geo["fxy"], R_CONTROL * days["s"] * m[None],
                                       days["ux"], days["uy"])
                for p, m in geo["policies"].items()}

    def exposures(self, geo, days):
        F = self._features(geo["mxy"], geo["fxy"], days["s"], days["ux"], days["uy"])
        pred = self.b + F @ self.theta
        self._Fe = self._expo_design(geo, days)
        return {p: F_ @ self.theta for p, F_ in self._Fe.items()}, pred

    def draws(self, geo, days, y, train_days, S, rng):
        Th = rng.multivariate_normal(self.theta, self.cov, S)
        Fe = self._Fe if hasattr(self, "_Fe") else self._expo_design(geo, days)
        return [{p: F_ @ th for p, F_ in Fe.items()} for th in Th]


# ----------------------------------------------------------------------------
# Outcome analysis
# ----------------------------------------------------------------------------
BETA, BETA_X = 2.0, 0.5
SIGMA_H = 0.30   # outcome noise SD (oracle SE of beta about 0.03-0.11 across policies)


def residualizer(Z):
    Q, _ = np.linalg.qr(Z)
    return lambda v: v - Q @ (Q.T @ v)


def ols_beta(Mfun, e, Hmat):
    """Regress each row of Hmat (R, n) on [Z, e]; return beta_hat, SE (R,)."""
    me = Mfun(e)
    denom = me @ me
    MH = Mfun(Hmat.T).T                      # (R, n)
    b = MH @ me / denom
    res = MH - b[:, None] * me[None]
    n, p = Hmat.shape[1], 3
    s2 = (res**2).sum(1) / (n - p)
    return b, np.sqrt(s2 / denom)


def rho(Mfun, e_m, e_k):
    a, b = Mfun(e_m), Mfun(e_k)
    return (a @ b) / (a @ a)


def rel_resid_err(Mfun, e_m, e_k):
    a, b = Mfun(e_m), Mfun(e_k)
    return np.linalg.norm(b - a) / np.linalg.norm(a)
