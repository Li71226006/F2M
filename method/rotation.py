from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation


def matrix_to_euler(matrix):
    device = matrix.device
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler("XYZ")
    return torch.tensor(euler, dtype=torch.float32, device=device)


def euler_to_matrix(euler):
    device = euler.device
    matrix = Rotation.from_euler("XYZ", euler.cpu().numpy()).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def matrix_to_rot6d(matrix):
    return matrix.T.reshape(9)[:6]


def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)


def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    return torch.stack([x, y, z], dim=-2).mT


def euler_to_rot6d(euler):
    return matrix_to_rot6d(euler_to_matrix(euler))


def rot6d_to_euler(rot6d):
    return matrix_to_euler(rot6d_to_matrix(rot6d))


def axisangle_to_matrix(axis, angle):
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor(
        [
            [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
            [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
            [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c],
        ]
    )


def q_euler_to_q_rot6d(q_euler):
    return torch.cat([q_euler[..., :3], euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)


def q_rot6d_to_q_euler(q_rot6d):
    return torch.cat([q_rot6d[..., :3], rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)
