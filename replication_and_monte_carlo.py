"""
Replication + Monte Carlo check of the simulation in
"Causal Inference with Interference under Exposure Mapping Uncertainty".

Reproduces the paper's Table 1 exactly (seed 123, same RNG order as
Spillover_identifiability_operator_learning .py), then:
  1. Repeats the outcome draw R times with COMMON noise across candidate
     operators (the original policy loop drew fresh noise per operator).
  2. Compares Monte Carlo means with the closed-form bias in Proposition 1:
         E[beta2_hat(T_m) | A] = beta2 * <M e_m, M e*> / ||M e_m||^2,
     where M residualizes on (1, A).
  3. Shows that the reported 0.303-0.305 "intervention effects" follow from
     fixing beta at the truth with row-normalized operators.
"""
import numpy as np
from scipy.spatial.distance import cdist

# ---------------- original design (matches repo, seed 123) ----------------
np.random.seed(123)
n, b1, b2 = 500, 1.0, 2.0
coords = np.random.uniform(0, 10, (n, 2))
D = cdist(coords, coords)


def kern(scale):
    T = np.exp(-D / scale)
    np.fill_diagonal(T, 1)
    return T / T.sum(1, keepdims=True)


T_true = kern(1.5)
dist_c = np.linalg.norm(coords - [5, 5], axis=1)
A = (dist_c < 1.5).astype(float)
PM = T_true @ A + np.random.normal(0, 0.01, n)      # NOTE: SD 0.01, not variance
E_true = T_true @ A
H = b1 * A + b2 * E_true + np.random.normal(0, 0.05, n)

T_pde, T_pino = kern(1.7), kern(1.3)
T_fno = 0.5 * T_pde + 0.5 * T_pino
T_fno /= T_fno.sum(1, keepdims=True)
T_geo = 0.7 * T_pino + 0.3 * T_fno + np.random.normal(0, 0.0005, (n, n))
T_geo = np.maximum(T_geo, 0)
T_geo /= T_geo.sum(1, keepdims=True)
models = {"PDE": T_pde, "PINO": T_pino, "FNO": T_fno, "GeoPT": T_geo}


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def resid(v, Z):
    Z = np.column_stack([np.ones(n), Z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


print("Replication of Table 1 (single draw):")
for k, T in models.items():
    print(f"  {k:6s} beta2_hat = {ols(np.column_stack([A, T @ A]), H)[2]:.3f}")

# ---------------- Monte Carlo with common noise ----------------
A_point = np.zeros(n)
A_point[np.argmin(dist_c)] = 1
A_strip = (coords[:, 0] < 3).astype(float)            # the paper's "upwind" policy
policies = {"Regional (disc)": A, "Point source": A_point, "'Upwind' (x<3)": A_strip}

R = 5000
rng = np.random.default_rng(1)
print(f"\nMonte Carlo, R={R}, common outcome noise across operators")
for pname, Ap in policies.items():
    Es = T_true @ Ap
    Hs = b1 * Ap + b2 * Es + rng.normal(0, 0.05, (R, n))
    print(f"\n== {pname}: {int(Ap.sum())} treated units")
    for k, T in models.items():
        Em = T @ Ap
        X = np.column_stack([np.ones(n), Ap, Em])
        bh = (np.linalg.pinv(X) @ Hs.T)[2]
        Mm, Ms = resid(Em, Ap), resid(Es, Ap)
        rho = Mm @ Ms / (Mm @ Mm)
        print(f"  {k:6s} MC mean={bh.mean():.3f}  MC SD={bh.std():.3f}  "
              f"95% MC range=({np.quantile(bh, .025):.2f}, {np.quantile(bh, .975):.2f})  "
              f"Prop.1 prediction={b2 * rho:.3f}")

# ---------------- affine-invariance counterexample ----------------
Hs = b1 * A + b2 * E_true + rng.normal(0, 0.05, (R, n))
Em = 0.8 * E_true
X = np.column_stack([np.ones(n), A, Em])
print(f"\nOperator 0.8*T_true: PM corr^2 = {np.corrcoef(PM, Em)[0, 1]**2:.3f} "
      f"(same as truth), beta2_hat = {(np.linalg.pinv(X) @ Hs.T)[2].mean():.3f}")

# ---------------- why tau = 0.303-0.305 ----------------
r = np.random.default_rng(0)
A1 = np.zeros(n)
A1[r.choice(n, 50, replace=False)] = 1
print("\nFixed-beta intervention effect (50 random treated units, beta held at truth):")
for k, T in models.items():
    tau = A1.mean() + b2 * (T @ A1).mean()
    print(f"  {k:6s} tau={tau:.4f}  mean column sum={T.sum(0).mean():.3f}")
print("  -> row-normalization forces mean column sum = 1, so tau ~ (b1+b2)*mean(A1) "
      "for any operator.")
