import numpy as np
from mosek.fusion import *
import mosek.fusion


def joint_learning_L_gamma(Xs, Xt, lambda_G, lambda_T, C):
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    ds = Xs.shape[1] if len(Xs.shape) == 2 else 1
    dt = Xt.shape[1] if len(Xs.shape) == 2 else 1
    a = np.ones(shape=(ns,)) / ns
    b = np.ones(shape=(nt,)) / nt
    G = np.outer(a, b)
    L = np.eye(ds, dt)

    while True:
        # Learn G_k+1 with fixed L_k
        with mosek.fusion.Model("minG") as M:

            G_k = M.variable("G_k", [ns, ds], Domain.greaterThan(0.0))
            # M.constraint(Expr.hstack(G_k, Expr.constTerm(1.0)), Domain.lessThan())
            M.constraint("c1", np.sum(G_k, axis=0) == a)
            M.constraint("c2", np.sum(G_k, axis=1) == b)

            first_norm = np.linalg.norm(
                x=(np.dot(Xs, L) - ns * np.dot(G_k, Xt)), ord="fro"
            )
            first_term = (1 / (ns * dt)) * (first_norm**2)
            second_term = (
                lambda_G / np.max(C) * np.linalg.norm(x=np.dot(G_k, C), ord="fro")
            )
            third_norm = np.linalg.norm(L - np.eye(ds, dt))
            third_term = (lambda_T / (ds * dt)) * (third_norm**2)
            obj_func = first_term + second_term + third_term

            M.objective("obj", ObjectiveSense.Minimize, obj_func)

            M.solve()

            G = G_k.level()

        # Learn L_k+1 using fixed G_k
        L = np.linalg.solve(
            a=(
                (1 / (ns * dt)) * np.dot(Xs.T, Xs)
                + (lambda_T / (ds * dt) * np.eye(ds, dt))
            ),
            b=(
                1 / (ns * dt) * np.dot(np.dot(Xs.T, ns * G), Xt)
                + (lambda_T / (ds * dt)) * np.eye(ds, dt)
            ),
        )

        if first_term < 1.0:
            break
        else:
            print(f"Loss: {first_term}")

    return G, L
