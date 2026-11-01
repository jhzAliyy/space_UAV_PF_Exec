# -*- coding: utf-8 -*-
"""Lloyd 覆盖控制（开源标准算法的离散点形态）。

正典：Cortés, Martínez, Bullo 2004《Coverage control for mobile sensing
networks》——智能体向自身 Voronoi 胞元内的密度加权质心移动。
密度为带权点集时，Voronoi 分配=逐密度点找最近智能体（精确等价，
Lloyd-Max 量化形态），胞元质心=被分配点集的加权均值。scipy.spatial
的 Voronoi 对连续区域给出同一划分的顶点积分形态；此处离散点形态
无凸包边界情形、确定性可复现（金样闸要求）。
"""


import numpy as np


def lloyd_targets(pos: np.ndarray, pts: np.ndarray, w: np.ndarray,
                  stay_eps: float = 0.5) -> np.ndarray:
    """标准 Lloyd 步：返回每架无人机的胞元密度加权质心。

    pos (N,2) 无人机位置；pts (M,2) 密度点（用户位置）；w (M,) 密度权重。
    空胞元（该无人机最近的用户为零）→ 目标=原地（stay）。
    """
    d2 = ((pts[:, None, :] - pos[None, :, :]) ** 2).sum(-1)  # (M,N)
    assign = d2.argmin(1)                                     # Voronoi 分配
    targets = pos.copy()
    for i in range(len(pos)):
        m = assign == i
        if m.any() and w[m].sum() > 1e-12:
            targets[i] = (pts[m] * w[m][:, None]).sum(0) / w[m].sum()
    return targets
