import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import mlflow
import ot
from Utils.joint_OT_mapping_linear_classreg import compute_joint_OT_mapping
from Utils.SVM_utils import test_svm
import matplotlib.lines as lines


def optimize_hyperparams(
    X_trains, y_trains, n_features, barycenter, yt, device, model, iter
):
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("office_caltech")

    method = "gaussian"
    bias = False
    mus = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]
    etas = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]
    num_itermax = iter
    numinneritermax = int(10)
    stopThr = 1e-7
    stopinnerThr = 1e-7

    for mu in mus:
        for eta in etas:
            for d in range(
                len(X_trains)
            ):  # for every domain, compute the map and coupling
                with mlflow.start_run(run_name=f"dummy run"):
                    mlflow.log_param("domain", d)
                    mlflow.log_param("mu", mu)
                    mlflow.log_param("eta", eta)
                    mlflow.log_param("numitermax", num_itermax)
                    mlflow.log_param("numinneritermax", numinneritermax)
                    mlflow.log_param("stopThr", stopThr)
                    mlflow.log_param("stopinnerThr", stopinnerThr)
                    mlflow.log_param("method", method)
                    mlflow.log_param("bias", bias)

                    G, L, loss = compute_joint_OT_mapping(
                        xs=X_trains[d].to(device),
                        xt=barycenter.to(device),
                        ys=y_trains[d].cpu(),
                        yt=yt.cpu(),
                        mu=mu,
                        eta=eta,
                        numItermax=num_itermax,
                        numInnerItermax=numinneritermax,
                        stopThr=stopThr,
                        stopInnerThr=stopinnerThr,
                        method=method,
                        bias=bias,
                        class_reg=True,
                    )

                    b = torch.zeros(size=(n_features,))

                    ls = [l.cpu() for l in loss["loss"]]
                    plt.plot(ls)
                    plt.title(f"Loss")
                    mlflow.log_figure(plt.gcf(), f"loss.png")
                    plt.close()

                    if method == "gaussian":
                        K = ot.utils.kernel(
                            X_trains[d], X_trains[d], method=method, sigma=1.0
                        )
                        if bias:
                            K = torch.concatenate(
                                [K, torch.ones((Xs.shape[0], 1), dtype=torch.float32)],
                                dim=1,
                            )

                        transp = (
                            G / torch.sum(G, 1)[:, None]
                        )  # standard barycentric mapping
                        transp = torch.nan_to_num(transp, nan=0, posinf=0, neginf=0)
                        mapped_Xtrain = K @ L

                        metrics, disp = test_svm(
                            model, mapped_Xtrain.cpu(), y_trains[d].cpu()
                        )

                        disp.plot()
                        mlflow.log_figure(plt.gcf(), f"confusion_mtx.png")
                        plt.close()

                        mlflow.log_metric("accuracy", metrics["accuracy"])
                        print(metrics["precision"])
                        for i, prec in enumerate(metrics["precision"]):
                            mlflow.log_metric("precision", prec, step=i)
                        for i, rec in enumerate(metrics["recall"]):
                            mlflow.log_metric("recall", rec, step=i)

                        # np.save("artifacts/G.npy", G.cpu().numpy())
                        # np.save("artifacts/L.npy", L.cpu().numpy())
                        # np.save("artifacts/b.npy", b.cpu().numpy())

                        # mlflow.log_artifact("artifacts/G.npy")
                        # mlflow.log_artifact("artifacts/L.npy")
                        # mlflow.log_artifact("artifacts/b.npy")

                        # os.remove("artifacts/G.npy")
                        # os.remove("artifacts/L.npy")
                        # os.remove("artifacts/b.npy")
