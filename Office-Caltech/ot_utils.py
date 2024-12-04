import numpy as np
from collections import defaultdict
import torch
import ot


def barycentric_mapping(xs, xt, xnew, coupling, batch_size=10):

    if torch.equal(xs, xnew):
        # perform standard barycentric mapping
        print("barycentric mapping")
        transp = coupling / torch.sum(coupling, axis=1)[:, None]

        # set nans to 0
        transp = torch.nan_to_num(transp, nan=0, posinf=0, neginf=0)

        # compute transported samples
        transp_Xs = transp @ xt
    else:
        # perform out of sample mapping
        print("out of sample mapping")
        indices = torch.arange(xnew.shape[0])
        batch_ind = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        transp_Xs = []
        for bi in batch_ind:
            # get the nearest neighbor in the source domain
            D0 = ot.dist(xnew[bi], xs)
            idx = torch.argmin(D0, dim=1)

            # transport the source samples
            transp = coupling / torch.sum(coupling, axis=1)[:, None]
            transp = torch.nan_to_num(transp, nan=0, posinf=0, neginf=0)
            transp_Xs_ = transp @ xt

            # define the transported points
            transp_Xs_ = transp_Xs_[idx, :] + xnew[bi] - xs[idx, :]

            transp_Xs.append(transp_Xs_)

        transp_Xs = torch.concatenate(transp_Xs, axis=0)

    return transp_Xs


def dist_classreg(xs, xt, ys, yt, device="cpu"):
    M = ot.dist(xs, xt)

    M_max = M.max() * 1.0001

    for c in torch.unique(ys):
        idx_s = torch.where((ys != c) & (ys != -1))[0]
        idx_t = torch.where(yt == c)[0]

        for j in idx_t:
            M[idx_s, j] = (
                M_max  # Needed for numerical reasons (see: https://github.com/PythonOT/POT/issues/229#issuecomment-824616912)
            )
    return M
