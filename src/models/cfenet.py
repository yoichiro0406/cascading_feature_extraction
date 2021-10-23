"""
Modified from:
    RPM-Net: Robust Point Matching using Learned Features
    https://github.com/yewzijian/RPMNet
"""
import argparse
import logging

import torch
import torch.nn as nn
from common.math_torch import se3
from common.torch import to_numpy
from models.feature_nets import (
    FeatExtractionEarlyFusion_ball,
    FeatExtractionEarlyFusion_xyz,
    ParameterPredictionNet,
)
from models.pointnet_util import angle_difference, square_distance

_logger = logging.getLogger(__name__)

_EPS = 1e-5  # To prevent division by zero
_EPS_SVD = 1e-9


def match_features(feat_src, feat_ref, metric="l2"):
    """Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == "l2":
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == "angle":
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


class SinkhornNorm:
    def __init__(self, sk_type, const_iters=5):
        self.sk_type = sk_type
        if sk_type == "sum" or sk_type == "sum_approx":
            self.one_pad = nn.ConstantPad2d((0, 1, 0, 1), 1.0)
        self.const_iters = const_iters

    # @profile
    def __call__(self, a, var_iters=None):
        if self.sk_type == "sum" or self.sk_type == "sum_approx":
            if self.sk_type == "sum":
                n_iters = self.const_iters
            else:
                n_iters = var_iters
            a = torch.exp(a)
            a = self.one_pad(a)
            for i in range(n_iters):
                a[:, :-1] /= torch.sum(a[:, :-1], dim=2, keepdim=True)
                a[:, :, :-1] /= torch.sum(a[:, :, :-1], dim=1, keepdim=True)
            return a[:, :-1, :-1]

        elif self.sk_type == "log":
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(a[:, None, :, :])

            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(self.const_iters):
                # Row normalization
                log_alpha_padded = torch.cat(
                    (
                        log_alpha_padded[:, :-1, :]
                        - (
                            torch.logsumexp(
                                log_alpha_padded[:, :-1, :], dim=2, keepdim=True
                            )
                        ),
                        log_alpha_padded[:, -1, None, :],
                    ),  # Don't normalize last row
                    dim=1,
                )

                # Column normalization
                log_alpha_padded = torch.cat(
                    (
                        log_alpha_padded[:, :, :-1]
                        - (
                            torch.logsumexp(
                                log_alpha_padded[:, :, :-1], dim=1, keepdim=True
                            )
                        ),
                        log_alpha_padded[:, :, -1, None],
                    ),  # Don't normalize last column
                    dim=2,
                )

            log_alpha = log_alpha_padded[:, :-1, :-1]
            return torch.exp(log_alpha)

        else:
            raise NotImplementedError


def compute_rigid_transform(
    a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor, via_cpu=True
):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    device = a.device

    weights_normalized = weights[..., None] / (
        torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS
    )
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    if via_cpu:
        centroid_a = centroid_a.to("cpu", non_blocking=True)
        centroid_b = centroid_b.to("cpu", non_blocking=True)
        cov = cov.cpu()

    # SVD is numerically unstable.
    try:
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
    except:
        return None
    if torch.any(torch.sum(s, dim=1) < _EPS_SVD):
        return None

    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(
        torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg
    )

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    if via_cpu:
        transform = transform.to(device)

    return transform


class CFENet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.via_cpu = args.via_cpu
        self.num_sk_max_iter = args.num_sk_iter
        self.sinkhorn_norm = SinkhornNorm(args.sk_type, const_iters=args.num_sk_iter)

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (
                feat_distance - alpha[:, None, None]
            )
        return hybrid_affinity

    # @profile
    def forward(self, data, num_iter: int = 5):
        """Forward pass for CFENet
        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        xyz_ref, norm_ref = data["points_ref"][:, :, :3], data["points_ref"][:, :, 3:6]
        xyz_src, norm_src = data["points_src"][:, :, :3], data["points_src"][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []

        betas, alphas = self.weights_net([xyz_src, xyz_ref])

        feat_ref_xyz = self.feat_extractor_ball(xyz_ref, norm_ref)
        feat_src_xyz = self.feat_extractor_ball(xyz_src_t, norm_src_t)

        for i in range(num_iter):

            beta = betas[:, i]
            alpha = alphas[:, i]

            feat_src_xyz = self.feat_extractor_xyz(
                xyz_src_t,
                feat_src_xyz,
                i,
            )
            feat_ref_xyz = self.feat_extractor_xyz(
                xyz_ref,
                feat_ref_xyz,
                i,
            )

            feat_distance = match_features(feat_src_xyz, feat_ref_xyz)
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            perm_matrix = self.sinkhorn_norm(
                affinity, int(self.num_sk_max_iter * i // num_iter) + 1
            )

            weighted_ref = (
                perm_matrix
                @ xyz_ref
                / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)
            )

            # transform is None if svd result seems to be ill.
            transform = compute_rigid_transform(
                xyz_src,
                weighted_ref,
                weights=torch.sum(perm_matrix, dim=2),
                via_cpu=self.via_cpu,
            )
            # avoid Nan backprop
            if transform is None:
                return None

            xyz_src_t, norm_src_t = se3.transform(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(to_numpy(beta))
            all_alpha.append(to_numpy(alpha))

        endpoints["perm_matrices"] = torch.stack(all_perm_matrices)
        endpoints["transforms"] = transforms

        return endpoints


class CFENetEarlyFusion(CFENet):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.weights_net = ParameterPredictionNet(args.num_reg_iter, weights_dim=[0])
        self.feat_extractor_ball = FeatExtractionEarlyFusion_ball(
            features=["ppf", "dxyz"],
            feature_dim=args.feat_dim,
            num_neighbors=args.num_neighbors,
            ppf_type=args.ppf_type,
            knn=args.knn,
            radius=args.radius,
        )
        self.feat_extractor_xyz = FeatExtractionEarlyFusion_xyz(
            args.feat_dim, args.num_reg_iter
        )


def get_model(args: argparse.Namespace) -> CFENet:
    return CFENetEarlyFusion(args)
