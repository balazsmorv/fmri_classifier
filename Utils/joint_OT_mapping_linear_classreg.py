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
from ot import mapping, sinkhorn
from ot.utils import dist, unif, list_to_array, kernel, dots
import matplotlib.pyplot as plt


def dist_classreg(xs, xt, ys, yt):
    M = ot.dist(xs, xt)
    M_max = M.max() * 1.0001
    for c in np.unique(ys):
        idx_s = np.where((ys != c) & (ys != -1))[0]
        idx_t = np.where(yt == c)[0]

        for j in idx_t:
            M[idx_s, j] = (
                M_max  # Needed for numerical reasons (see: https://github.com/PythonOT/POT/issues/229#issuecomment-824616912)
            )
    return M


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
    measure_labels=None,
    bary_labels=None,
    **kwargs,
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

        for i, (measure_locations_i, measure_weights_i, weight_i) in enumerate(
            zip(measures_locations, measures_weights, weights)
        ):
            if measure_labels is not None and bary_labels is not None:
                measure_labels_i = measure_labels[i]
                M_i = dist_classreg(
                    X, measure_locations_i, ys=bary_labels, yt=measure_labels_i
                )
            else:
                M_i = dist(X, measure_locations_i)
            T_i = ot.bregman.sinkhorn(
                b,
                measure_weights_i,
                M_i,
                reg=reg,
                numItermax=numInnerItermax,
                method=method,
                log=False,
                verbose=False,
                **kwargs,
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


class MappingTransport(ot.utils.BaseEstimator):
    """MappingTransport: DA methods that aims at jointly estimating a optimal
    transport coupling and the associated mapping

    Parameters
    ----------
    mu : float, optional (default=1)
        Weight for the linear OT loss (>0)
    eta : float, optional (default=0.001)
        Regularization term for the linear mapping `L` (>0)
    bias : bool, optional (default=False)
        Estimate linear mapping with constant bias
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    kernel : string, optional (default="linear")
        The kernel to use either linear or gaussian
    sigma : float, optional (default=1)
        The gaussian kernel parameter
    max_iter : int, optional (default=100)
        Max number of BCD iterations
    tol : float, optional (default=1e-5)
        Stop threshold on relative loss decrease (>0)
    max_inner_iter : int, optional (default=10)
        Max number of iterations (inner CG solver)
    inner_tol : float, optional (default=1e-6)
        Stop threshold on error (inner CG solver) (>0)
    log : bool, optional (default=False)
        record log if True
    verbose : bool, optional (default=False)
        Print information along iterations
    verbose2 : bool, optional (default=False)
        Print information along iterations

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    mapping_ :
        The associated mapping

        - array-like, shape (`n_features` (+ 1), `n_features`),
          (if bias) for kernel == linear

        - array-like, shape (`n_source_samples` (+ 1), `n_features`),
          (if bias) for kernel == gaussian
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
            "Mapping estimation for discrete optimal transport",
            Neural Information Processing Systems (NIPS), 2016.

    """

    def __init__(
        self,
        mu=1,
        eta=0.001,
        bias=False,
        metric="sqeuclidean",
        norm=None,
        kernel="linear",
        sigma=1,
        max_iter=100,
        tol=1e-5,
        max_inner_iter=10,
        inner_tol=1e-6,
        log=False,
        verbose=False,
        verbose2=False,
    ):
        self.metric = metric
        self.norm = norm
        self.mu = mu
        self.eta = eta
        self.bias = bias
        self.kernel = kernel
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.verbose2 = verbose2

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        r"""Builds an optimal coupling and estimates the associated mapping
        from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self
        """
        self._get_backend(Xs, ys, Xt, yt)

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs, Xt=Xt):

            self.xs_ = Xs
            self.xt_ = Xt
            self.ys_ = ys
            self.yt_ = yt

            if self.kernel == "linear":
                returned_ = joint_OT_mapping_linear(
                    Xs,
                    Xt,
                    ys=ys,
                    yt=yt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    verbose=self.verbose,
                    verbose2=self.verbose2,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopThr=self.tol,
                    stopInnerThr=self.inner_tol,
                    log=self.log,
                    class_reg=True,
                )

            elif self.kernel == "gaussian":
                returned_ = joint_OT_mapping_kernel(
                    Xs,
                    Xt,
                    ys=ys,
                    yt=yt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    sigma=self.sigma,
                    verbose=self.verbose,
                    verbose2=self.verbose,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopInnerThr=self.inner_tol,
                    stopThr=self.tol,
                    log=self.log,
                    class_reg=True,
                )

            # deal with the value of log
            if self.log:
                self.coupling_, self.mapping_, self.log_ = returned_
            else:
                self.coupling_, self.mapping_ = returned_
                self.log_ = dict()

        return self

    def transform(self, Xs):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):
                # perform standard barycentric mapping
                print("Barycentric mapping")
                transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                if self.kernel == "gaussian":
                    print("Gaussian mapping")
                    K = kernel(Xs, self.xs_, method=self.kernel, sigma=self.sigma)
                elif self.kernel == "linear":
                    print("Linear mapping")
                    K = Xs
                if self.bias:
                    K = nx.concatenate(
                        [K, nx.ones((Xs.shape[0], 1), type_as=K)], axis=1
                    )
                transp_Xs = nx.dot(K, self.mapping_)

            return transp_Xs


def joint_OT_mapping_linear(
    xs,
    xt,
    ys,
    yt,
    mu=1,
    eta=0.001,
    bias=True,
    verbose=False,
    verbose2=False,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
    class_reg=True,
    **kwargs,
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
    M_max = M.max() * 1.0001
    # Ötlet 2: legyenek a különböző classú minták nagyon távol egymástól
    # Ez elvileg nem rontja el a konvexitást, mert nem függ sem gammától, sem L-től.
    if class_reg:
        for c in nx.unique(ys):
            idx_s = np.where((ys != c) & (ys != -1))[0]
            idx_t = np.where(yt == c)[0]

            for j in idx_t:
                M[idx_s, j] = (
                    M_max
                    # Needed for numerical reasons (see: https://github.com/PythonOT/POT/issues/229#issuecomment-824616912)
                )
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


def joint_OT_mapping_kernel(
    xs,
    xt,
    ys,
    yt,
    mu=1,
    eta=0.001,
    kerneltype="gaussian",
    sigma=1,
    bias=True,
    verbose=False,
    verbose2=False,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
    class_reg=True,
    classes=None,
    **kwargs,
):
    r"""Joint OT and nonlinear mapping estimation with kernels as proposed in
    :ref:`[8] <references-joint-OT-mapping-kernel>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma, L\in\mathcal{H}}\quad \|L(\mathbf{X_s}) -
        n_s\gamma \mathbf{X_t}\|^2_F + \mu \langle \gamma, \mathbf{M} \rangle_F +
        \eta \|L\|^2_\mathcal{H}

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`n_s \times d` linear operator on a kernel matrix that
      approximates the barycentric mapping
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and the nonlinear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-kernel>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditional gradient)
    and the update of :math:`\mathbf{L}` using a classical kernel least square solver.


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
    kerneltype : str,optional
        kernel used by calling function :py:func:`ot.utils.kernel` (gaussian by default)
    sigma : float, optional
        Gaussian kernel bandwidth.
    bias : bool,optional
        Estimate linear mapping with constant bias
    verbose : bool, optional
        Print information along iterations
    verbose2 : bool, optional
        Print information along iterations
    numItermax : int, optional
        Max number of BCD iterations
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (ns, d) array-like
        Nonlinear mapping matrix ((:math:`n_s+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-kernel:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
       "Mapping estimation for discrete optimal transport",
       Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    if classes is None:
        classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt = xs.shape[0], xt.shape[0]

    K = kernel(xs, xs, method=kerneltype, sigma=sigma)
    if bias:
        K1 = nx.concatenate((K, nx.ones((ns, 1), type_as=xs)), axis=1)
        Id = nx.eye(ns + 1, type_as=xs)
        Id[-1] = 0
        Kp = nx.eye(ns + 1, type_as=xs)
        Kp[:ns, :ns] = K

        # ls regu
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # RKHS regul
        K0 = nx.dot(K1.T, K1) + eta * Kp
        Kreg = Kp

    else:
        K1 = K
        Id = nx.eye(ns, type_as=xs)

        # ls regul
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # proper kernel ridge
        K0 = K + eta * Id
        Kreg = K

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    M_max = M.max() * 1.0001
    # Ötlet 2: legyenek a különböző classú minták nagyon távol egymástól
    # Ez elvileg nem rontja el a konvexitást, mert nem függ sem gammától, sem L-től.
    if class_reg:
        for c in classes:
            idx_s = np.where((ys != c) & (ys != -1))[0]
            idx_t = np.where(yt == c)[0]

            for j in idx_t:
                M[idx_s, j] = (
                    M_max  # Needed for numerical reasons (see: https://github.com/PythonOT/POT/issues/229#issuecomment-824616912)
                )

    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(K1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.trace(dots(L.T, Kreg, L))
        )

    def solve_L_nobias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, xst)

    def solve_L_bias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, nx.dot(K1.T, xst))

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(K1, L)

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
        # G = ot.sinkhorn(
        #     a,
        #     b,
        #     M,
        #     mu,
        #     "sinkhorn_log",
        #     numitermax=numInnerItermax,
        #     stopThr=stopInnerThr,
        #     verbose=False,
        # )

        return G

    if bias:
        solve_L = solve_L_bias
    else:
        solve_L = solve_L_nobias

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
    classes=None,
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

    if method == "linear":
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
            classes=classes,
        )
    elif method == "gaussian":
        G, L, loss = joint_OT_mapping_kernel(
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
            classes=classes,
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


class OTDA(object):
    """Class for domain adaptation with optimal transport as proposed in [5]


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    """

    def __init__(self, metric="sqeuclidean"):
        """Class initialization"""
        self.xs = 0
        self.xt = 0
        self.G = 0
        self.metric = metric
        self.computed = False

    def fit(self, xs, xt, ws=None, wt=None):
        """Fit domain adaptation between samples is xs and xt (with optional weights)"""
        self.xs = xs
        self.xt = xt

        if wt is None:
            wt = unif(xt.shape[0])
        if ws is None:
            ws = unif(xs.shape[0])

        self.ws = ws
        self.wt = wt

        self.M = dist(xs, xt, metric=self.metric)
        self.G = emd(ws, wt, self.M)
        self.computed = True

    def interp(self, direction=1):
        """Barycentric interpolation for the source (1) or target (-1) samples

        This Barycentric interpolation solves for each source (resp target)
        sample xs (resp xt) the following optimization problem:

        .. math::
            arg\min_x \sum_i \gamma_{k,i} c(x,x_i^t)

        where k is the index of the sample in xs

        For the moment only squared euclidean distance is provided but more
        metric  could be used in the future.

        """
        if direction > 0:  # >0 then source to target
            G = self.G
            w = self.ws.reshape((self.xs.shape[0], 1))
            x = self.xt
        else:
            G = self.G.T
            w = self.wt.reshape((self.xt.shape[0], 1))
            x = self.xs

        if self.computed:
            if self.metric == "sqeuclidean":
                return np.dot(G / w, x)  # weighted mean
            else:
                print("Warning, metric not handled yet, using weighted average")
                return np.dot(G / w, x)  # weighted mean
                return None
        else:
            print("Warning, model not fitted yet, returning None")
            return None

    def predict(self, x, direction=1):
        """Out of sample mapping using the formulation from [6]

        For each sample x to map, it finds the nearest source sample xs and
        map the samle x to the position xst+(x-xs) wher xst is the barycentric
        interpolation of source sample xs.

        References
        ----------

        .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

        """
        if direction > 0:  # >0 then source to target
            xf = self.xt
            x0 = self.xs
        else:
            xf = self.xs
            x0 = self.xt

        D0 = dist(x, x0)  # dist netween new samples an source
        idx = np.argmin(D0, 1)  # closest one
        xf = self.interp(direction)  # interp the source samples
        return xf[idx, :] + x - x0[idx, :]  # aply the delta to the interpolation


class OTDA_mapping_linear(OTDA):
    """Class for optimal transport with joint linear mapping estimation as in [8]"""

    def __init__(self):
        """Class initialization"""

        self.xs = 0
        self.xt = 0
        self.G = 0
        self.L = 0
        self.bias = False
        self.computed = False
        self.metric = "sqeuclidean"

    def fit(self, xs, xt, mu=1, eta=1, bias=False, **kwargs):
        """Fit domain adaptation between samples is xs and xt (with optional
        weights)"""
        self.xs = xs
        self.xt = xt
        self.bias = bias

        self.ws = unif(xs.shape[0])
        self.wt = unif(xt.shape[0])

        self.G, self.L = joint_OT_mapping_linear(
            xs, xt, mu=mu, eta=eta, bias=bias, **kwargs
        )
        self.computed = True

    def mapping(self):
        return lambda x: self.predict(x)


class OTDA_mapping_kernel(OTDA_mapping_linear):
    """Class for optimal transport with joint nonlinear mapping estimation as in [8]"""

    def fit(
        self, xs, xt, mu=1, eta=1, bias=False, kerneltype="gaussian", sigma=1, **kwargs
    ):
        """Fit domain adaptation between samples is xs and xt"""
        self.xs = xs
        self.xt = xt
        self.bias = bias

        self.ws = unif(xs.shape[0])
        self.wt = unif(xt.shape[0])
        self.kernel = kerneltype
        self.sigma = sigma
        self.kwargs = kwargs

        self.G, self.L = joint_OT_mapping_kernel(
            xs, xt, mu=mu, eta=eta, bias=bias, **kwargs
        )
        self.computed = True

    def predict(self, x):
        """Out of sample mapping estimated during the call to fit"""

        if self.computed:
            K = kernel(x, self.xs, method=self.kernel, sigma=self.sigma, **self.kwargs)
            if self.bias:
                K = np.hstack((K, np.ones((x.shape[0], 1))))
            return K.dot(self.L)
        else:
            print("Warning, model not fitted yet, returning None")
            return None
