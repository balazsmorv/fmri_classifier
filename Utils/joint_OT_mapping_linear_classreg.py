# -*- coding: utf-8 -*-
"""
Optimal Transport maps and variants

.. warning::
    Note that by default the module is not imported in :mod:`ot`. In order to
    use it you need to explicitly import :mod:`ot.mapping`
"""
import ot.da
from ot.backend import get_backend, to_numpy
from ot.lp import emd
import numpy as np
import mlflow
from ot.optim import cg
from ot.utils import dist, unif, list_to_array, kernel, dots
import matplotlib.pyplot as plt


def free_support_sinkhorn_barycenter(
    measures_locations,
    measures_weights,
    X_init,
    reg,
    b=None,
    weights=None,
    numItermax=100,
    numInnerItermax=1000,
    stopThr=1e-7,
    verbose=False,
    log=None,
    method="sinkhorn",
    **kwargs
):
    r"""
    Solves the free support (locations of the barycenters are optimized, not the weights) regularized Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Sinkhorn divergence), formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i W_{reg}^2(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

    where :

    - :math:`w \in \mathbb{(0, 1)}^{N}`'s are the barycenter weights and sum to one
    - `measure_weights` denotes the :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}`: empirical measures weights (on simplex)
    - `measures_locations` denotes the :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d}`: empirical measures atoms locations
    - :math:`\mathbf{b} \in \mathbb{R}^{k}` is the desired weights vector of the barycenter

    This problem is considered in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).
    There are two differences with the following codes:

    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. :math:`\theta = 1` in
      :ref:`[20] <references-free-support-barycenter>` (Algorithm 2). This can be seen as a discrete
      implementation of the fixed-point algorithm of
      :ref:`[43] <references-free-support-barycenter>` proposed in the continuous setting.
    - at each iteration, instead of solving an exact OT problem, we use the Sinkhorn algorithm for calculating the
      transport plan in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).

    Parameters
    ----------
    measures_locations : list of N (k_i,d) array-like
        The discrete support of a measure supported on :math:`k_i` locations of a `d`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    measures_weights : list of N (k_i,) array-like
        Numpy arrays where each numpy array has :math:`k_i` non-negatives values summing to one
        representing the weights of each discrete input measure

    X_init : (k,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    reg : float
        Regularization term >0
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations when calculating the transport plans with Sinkhorn
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT solver
    ot.lp.free_support_barycenter : Barycenter solver based on Linear Programming

    .. _references-free-support-barycenter:
    References
    ----------
    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """
    nx = get_backend(*measures_locations, *measures_weights, X_init)

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = nx.ones((k,), type_as=X_init) / k
    if weights is None:
        weights = nx.ones((N,), type_as=X_init) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1.0

    while displacement_square_norm > stopThr and iter_count < numItermax:

        T_sum = nx.zeros((k, d), type_as=X_init)

        for measure_locations_i, measure_weights_i, weight_i in zip(
            measures_locations, measures_weights, weights
        ):
            M_i = dist(X, measure_locations_i)
            T_i = ot.bregman.sinkhorn(
                b,
                measure_weights_i,
                M_i,
                reg=reg,
                numItermax=numInnerItermax,
                method=method,
                **kwargs
            )
            T_sum = T_sum + weight_i * 1.0 / b[:, None] * nx.dot(
                T_i, measure_locations_i
            )

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print(
                "iteration %d, displacement_square_norm=%f\n",
                iter_count,
                displacement_square_norm,
            )

        iter_count += 1

    if log:
        log_dict["displacement_square_norms"] = displacement_square_norms
        return X, log_dict
    else:
        return X


def joint_OT_mapping_linear(
    xs,
    xt,
    ys,
    yt,
    mu=1,
    eta=0.001,
    bias=False,
    verbose=False,
    verbose2=False,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
    class_reg=True,
    **kwargs
):
    r"""Joint OT and linear mapping estimation as proposed in
    :ref:`[8] <references-joint-OT-mapping-linear>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L}\quad \|L(\mathbf{X_s}) - n_s\gamma \mathbf{X_t} \|^2_F +
          \mu \langle \gamma, \mathbf{M} \rangle_F + \eta \|L - \mathbf{I}\|^2_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`d\times d` linear operator that approximates the barycentric
      mapping
    - :math:`\mathbf{I}` is the identity matrix (neutral linear mapping)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and a linear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-linear>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditional gradient)
    and the update of :math:`\mathbf{L}` using a classical least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    bias : bool,optional
        Estimate linear mapping with constant bias
    numItermax : int, optional
        Max number of BCD iterations
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (d, d) array-like
        Linear mapping matrix ((:math:`d+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-linear:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
        "Mapping estimation for discrete optimal transport",
        Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    Original Authors: Eloi Tanguy <eloi.tanguy@u-paris.fr>
                    Remi Flamary <remi.flamary@unice.fr>
    License: MIT License

    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt, d = xs.shape[0], xt.shape[0], xt.shape[1]

    if bias:
        xs1 = nx.concatenate((xs, nx.ones((ns, 1), type_as=xs)), axis=1)
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d + 1, type_as=xs)
        Id[-1] = 0
        I0 = Id[:, :-1]

        def sel(x):
            return x[:-1, :]

    else:
        xs1 = xs
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d, type_as=xs)
        I0 = Id

        def sel(x):
            return x

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    M_ = M.clone().detach()
    # Ötlet 2: legyenek a különböző classú minták nagyon távol egymástól
    # Ez elvileg nem rontja el a konvexitást, mert nem függ sem gammától, sem L-től.
    if class_reg:
        for c in [2, 4]:
            idx_s = np.where((ys != c) & (ys != -1))[0]
            idx_t = np.where(yt == c)[0]

            for j in idx_t:
                M_[idx_s, j] = (
                    M.max() * 1.0001
                )  # Needed for numerical reasons (see: https://github.com/PythonOT/POT/issues/229#issuecomment-824616912)
    M = M_
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        # Ötlet 1: If the coupling mtx assigns weight to non-matching class target sample, penalize
        # Ha valamilyen thresholdnál több masst akarna rossz labelű mintába vinni akkor penalty
        return (
            nx.sum((nx.dot(xs1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.sum(sel(L - I0) ** 2)
        )

    def solve_L(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(xstxs + eta * Id, nx.dot(xs1.T, xst) + eta * I0)

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(xs1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

        G = cg(
            a,
            b,
            M,
            1.0 / mu,
            f,
            df,
            G0=G0,
            numItermax=numInnerItermax,
            stopThr=stopInnerThr,
        )
        return G

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:

        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print(
                "{:5d}|{:8e}|{:8e}".format(
                    it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])
                )
            )
    if log:
        log["loss"] = vloss
        return G, L, log
    else:
        return G, L


def compute_joint_OT_mapping(
    xs,
    xt,
    ys,
    yt,
    method="linear",
    mu=1.0,
    eta=1.0,
    bias=True,
    verbose=True,
    numItermax=100000,
    numInnerItermax=100000,
    stopInnerThr=1e-10,
    stopThr=1e-10,
    log=False,
    class_reg=True,
):
    G, L, loss = joint_OT_mapping_linear(
        xs=xs,
        xt=xt,
        ys=ys,
        yt=ys,
        mu=mu,
        eta=eta,
        bias=bias,
        verbose=verbose,
        numItermax=numItermax,
        numInnerItermax=numInnerItermax,
        stopInnerThr=stopInnerThr,
        stopThr=stopThr,
        log=True,
        class_reg=class_reg,
    )

    if log:
        for i, entry in enumerate(loss["loss"]):
            mlflow.log_metric("joint_ot_loss", value=entry, step=i)
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(G)
        axarr[0].set_title("Coupling matrix")
        axarr[1].imshow(L)
        axarr[1].set_title("Linear transformation mtx")
        mlflow.log_figure(fig, artifact_file="coupling_map.png")

    return G, L, loss
