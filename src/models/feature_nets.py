"""
Feature Extraction and Parameter Prediction networks

Modified from:
    RPM-Net: Robust Point Matching using Learned Features
    https://github.com/yewzijian/RPMNet

"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import sample_and_group_multi

_raw_features_sizes = {"xyz": 3, "dxyz": 3, "ppf": 4}
_raw_features_order = {"xyz": 0, "dxyz": 1, "ppf": 2}


class ParameterPredictionNet(nn.Module):
    def __init__(self, n_iter, weights_dim, gammas_dim=3):
        """PointNet based Parameter prediction network"""

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.weights_dim = weights_dim
        self.n_iter = n_iter
        self.gammas_dim = gammas_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.n_iter + np.prod(weights_dim)),
        )

        self._logger.info("Predicting weights with dim {}.".format(self.weights_dim))

    def forward(self, x):
        """Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode="constant", value=0)
        ref_padded = F.pad(x[1], (0, 1), mode="constant", value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        betas = F.softplus(raw_weights[:, : self.n_iter])
        alphas = F.softplus(raw_weights[:, self.n_iter : self.n_iter * 2])

        return betas, alphas


class FeatExtractionEarlyFusion_ball(nn.Module):
    """Feature extraction Module that extracts hybrid features"""

    def __init__(
        self,
        features,
        feature_dim,
        num_neighbors,
        radius,
        ppf_type,
        knn,
    ):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("Using early fusion, feature dim = {}".format(feature_dim))
        self.n_sample = num_neighbors
        self.radius = radius
        self.ppf_type = ppf_type
        self.knn = knn

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info(
            "Feature extraction using features {}".format(", ".join(self.features))
        )

        # Layers
        raw_dim = np.sum(
            [_raw_features_sizes[f] for f in self.features]
        )  # number of channels after concat

        self.prepool = nn.Sequential(
            nn.Conv2d(raw_dim, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(),
        )

    # @profile
    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        features = sample_and_group_multi(
            -1,
            self.radius,
            self.n_sample,
            xyz,
            normals,
            ppf_type=self.ppf_type,
            knn=self.knn,
        )
        features["xyz"] = features["xyz"][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, feat, k, N]
        new_feat = self.prepool(new_feat).permute(0, 3, 2, 1)

        pooled_feat = torch.max(new_feat, dim=2)[0]  # [B, feat, N]

        cluster_feat = pooled_feat

        return cluster_feat  # (B, N, C)


class FeatExtractionEarlyFusion_xyz(nn.Module):
    "this method will be called once when prediction"
    "xyz_dim = out dim"

    def __init__(self, feat_dim, n_iter=5):
        super().__init__()
        nets_a = []
        nets_b = []

        for _ in range(n_iter):
            nets_a.append(
                nn.Sequential(
                    nn.GroupNorm(8, feat_dim),
                    nn.ReLU(),
                    nn.Conv1d(feat_dim, feat_dim, 1),
                )
            )
            nets_b.append(nn.Sequential(nn.Conv1d(3, feat_dim, 1, bias=False)))

        self.nets_a = nn.ModuleList(nets_a)
        self.nets_b = nn.ModuleList(nets_b)

    # @profile
    def forward(self, xyz, pre_feat, i):
        xyz = xyz.permute(0, 2, 1)
        xyz_feat = self.nets_b[i](xyz)

        xyz_feat = xyz_feat.permute(0, 2, 1)
        new_feat = pre_feat + xyz_feat

        new_feat = new_feat.permute(0, 2, 1)
        new_feat = self.nets_a[i](new_feat)
        new_feat = new_feat.permute(0, 2, 1)

        new_feat = new_feat / torch.norm(new_feat, dim=-1, keepdim=True)
        return new_feat
