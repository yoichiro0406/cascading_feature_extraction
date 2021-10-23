import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt


def save_plot(pcds, dimension, save_path):
    """
    Input should be either numpy.ndarray of torch.tensor
    Dimension should be specified with str ('2d', '3d')
    """
    if pcds[0].dtype == torch.float:
        for (i, pcd) in enumerate(pcds):
            pcds[i] = pcd.detach().cpu().numpy()

    fig = plt.figure(figsize=(50, 50))

    if dimension == "2d":
        ax = fig.add_subplot()
        for pcd in pcds:
            ax.scatter(pcd[..., 0], pcd[..., 1], s=20)

    elif dimension == "3d":
        ax = fig.add_subplot(111, projection="3d")
        for pcd in pcds:
            ax.scatter(pcd[..., 0], pcd[..., 1], pcd[..., 2])

    else:
        raise NotImplementedError('Dimension must be specified with "2d" or "3d"')

    plt.savefig(save_path, transparent=True)
    plt.close()
