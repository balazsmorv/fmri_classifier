import ot
from OT_utils.joint_OT_mapping_linear_classreg import compute_joint_OT_mapping
import torch
from SVM_utils import train_svm, test_svm
from data_utils import setup_MNIST_dataset
import mlflow  # to start a logging server in the terminal: mlflow server --host 127.0.0.1 --port 8080

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from visualization_utils import *


# Hyperparameters
dataset = "MNIST"
n_samples = 50
noise_scale = 0.4
dtype = torch.float32
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
log = True
n_sites = 2


def optimize_hyperparams():
    # X_train_original and X_train_L have shape (n_digits, n_samples, 784)
    X_train_original, ys, X_train_L = setup_MNIST_dataset(
        root="/Users/balazsmorvay/PycharmProjects/fmri_classifier/Data/MNIST/raw",
        digits=digits,
        indices=list(np.arange(start=0, stop=n_samples)),
        device=device,
        noise_scale=noise_scale,
    )

    mlflow.log_figure(plot_images(X_train_original), "original.png")
    mlflow.log_figure(plot_images(X_train_L), "modified.png")

    # All diracs have uniform probability mass of 1 / n_samples
    a_i = 1.0 / n_samples
    w_i = torch.ones(size=(n_samples,), dtype=torch.int, device=device) * a_i

    # Compute the free support barycenter
    barycenter, log_dict = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=torch.concatenate((X_train_original, X_train_L)),
        measures_weights=[w_i] * len(digits),
        X_init=torch.randn(
            (2 * n_samples, X_train_original.shape[2]),
            dtype=torch.float32,
            device=device,
        ),
        reg=25.0,
        b=torch.ones((n_samples * 2,), dtype=torch.float32, device=device)
        / (n_samples * 2),
        numItermax=2e10,
        numInnerItermax=1000,
        verbose=True,
        stopThr=1e-7,
        log=True,
    )

    # mlflow.log_dict([tensor.item() for tensor in log_dict["displacement_square_norms"]], "barycenter_loss.json")
    bary_samples = visualize_barycenter_diracs(barycenter=barycenter, num_images=5)
    mlflow.log_figure(bary_samples, artifact_file="barycenter.png")

    # Compute the Kantorovich coupling and Monge map simultaniously
    G_1, L_1_0 = compute_joint_OT_mapping(
        xs=X_train_original, xt=barycenter, ys=ys, yt=ys, method="linear"
    )
    L1 = L_1_0[0:784]
    b1 = L_1_0[784]

    G_2, L_2_0 = compute_joint_OT_mapping(
        xs=X_train_original, xt=barycenter, ys=ys, yt=ys, method="linear"
    )
    L2 = L_2_0[0:784]
    b2 = L_2_0[784]

    model = train_svm(X=barycenter, y=ys, kernel="rbf", C=1.0)
    assert model.fit_status_ == 0

    # Performance on train and source domains
    s1_metrics, _ = test_svm(model, x=X_train_original, y=ys)
    s2_metrics, _ = test_svm(model, x=X_train_L, y=ys)
    bary_metrics, _ = test_svm(model, x=barycenter, y=ys)

    t1_metrics, _ = test_svm(model, x=X_train_original, y=ys, L=L1, b=b1)
    t2_metrics, _ = test_svm(model, x=X_train_L, y=ys, L=L2, b=b2)

    # Performance on test data
    X_test_original, y_test, X_test_L = setup_MNIST_dataset(
        root="/Users/balazsmorvay/PycharmProjects/fmri_classifier/Data/MNIST/raw",
        digits=digits,
        indices=list(np.arange(start=n_samples, stop=2 * n_samples)),
        device=device,
        noise_scale=noise_scale,
    )
    new1_metrics, _ = test_svm(model, x=X_test_original, y=y_test)
    new2_metrics, _ = test_svm(model, x=X_test_L, y=y_test)
    new1_l_metrics, _ = test_svm(model, x=X_test_original, y=y_test, L=L1, b=b1)
    new2_l_metrics, _ = test_svm(model, x=X_test_L, y=y_test, L=L2, b=b2)

    print(s1_metrics)
    print(s2_metrics)
    print(bary_metrics)
    print(t1_metrics)
    print(t2_metrics)
    print(new1_metrics)
    print(new2_metrics)
    print(new1_l_metrics)
    print(new2_l_metrics)


if __name__ == "__main__":
    print("Using device: {}".format(device))
    if log:
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("MNIST_WTB")
        with mlflow.start_run() as run:
            mlflow.log_param("noise_scale", noise_scale)
            mlflow.log_param("dtype", dtype)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("digits", digits)
            optimize_hyperparams()
    else:
        optimize_hyperparams()
