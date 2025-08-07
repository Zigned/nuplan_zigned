import os
import numpy as np
import pandas as pd
import torch
import math
from pathlib import Path
import cv2
import pprint
import scipy

from typing import Any, List, Optional, Union

from nuplan.planning.simulation.simulation_log import SimulationLog


def interpolate_polynomial(deg: int,
                           x_1: Union[float, np.ndarray],
                           y_0: Union[float, np.ndarray, List[np.ndarray]],
                           y_prime_0: Union[float, np.ndarray, List[np.ndarray]],
                           y_pprime_0: Union[float, np.ndarray, List[np.ndarray]],
                           y_prime_1: Union[float, np.ndarray],
                           y_pprime_1: Union[float, np.ndarray],
                           x_0: Union[float, np.ndarray]=0.,
                           y_1: Optional[Union[float, np.ndarray]]=None,
                           ) -> Union[np.matrix, List[np.matrix]]:
    """
    Polynomial function: ``y(x) = params[deg, k] * x**deg + ... + params[0, k]``, k = 0, 2, ..., K-1
    A * P = Q
    P = inv(A) * Q
    :param deg: degree of polynomials
    :param x_1: x horizon
    :param y_0: y(x=x_0)
    :param y_prime_0: y'(x=x_0)
    :param y_pprime_0: y"(x=x_0)
    :param y_prime_1: y'(x=x_1)
    :param y_pprime_1: y"(x=x_1)
    :param x_0: start of x
    :param y_1: y(x=x_1)
    :return: polynomial parameters
    """
    assert deg == 4 or deg == 5, f'degree must be 4 or 5, but got degree: {deg}'
    if deg == 4:
        # 4th-order Polynomial
        if isinstance(x_0, np.ndarray):
            if isinstance(y_0, np.ndarray):
                A = [[np.zeros((deg + 1, deg + 1)) for _ in range(x_0.shape[0])] for _ in range(x_0.shape[0])]
                for i, (start, end) in enumerate(zip(x_0, x_1)):
                    a = np.matrix([[1., start, start ** 2, start ** 3, start ** 4],
                                   [0., 1., 2 * start, 3 * start ** 2, 4 * start ** 3],
                                   [0., 0., 2., 6 * start, 12 * start ** 2],
                                   [0., 1., 2. * end, 3 * end ** 2, 4 * end ** 3],
                                   [0., 0., 2., 6 * end, 12 * end ** 2]])
                    A[i][i] = a
                A = np.block(A)
                q = np.array([
                    y_0,
                    y_prime_0,
                    y_pprime_0,
                    y_prime_1,
                    y_pprime_1
                ])
                Q = [[np.zeros_like(q) for _ in range(x_0.shape[0])] for _ in range(x_0.shape[0])]
                for i in range(x_0.shape[0]):
                    Q[i][i] = q
                Q = np.block(Q)
                P = A.I * Q
                P_list = []
                i = 0
                j = 0
                while i < P.shape[0]:
                    P_list.append(P[i:i + q.shape[0], j:j + q.shape[1]])
                    i += q.shape[0]
                    j += q.shape[1]
                P = P_list

            elif isinstance(y_0, list):
                P = []
                for i, (start, end) in enumerate(zip(x_0, x_1)):
                    a = np.matrix([[1., start, start ** 2, start ** 3, start ** 4],
                                   [0., 1., 2 * start, 3 * start ** 2, 4 * start ** 3],
                                   [0., 0., 2., 6 * start, 12 * start ** 2],
                                   [0., 1., 2. * end, 3 * end ** 2, 4 * end ** 3],
                                   [0., 0., 2., 6 * end, 12 * end ** 2]])
                    q = np.array([
                        y_0[i].repeat(y_pprime_1.shape[0]),
                        y_prime_0[i].repeat(y_pprime_1.shape[0]),
                        y_pprime_0[i].repeat(y_pprime_1.shape[0]),
                        np.tile(y_prime_1, y_0[i].shape[0]),
                        np.tile(y_pprime_1, y_0[i].shape[0])
                    ])
                    p = a.I * q
                    P.append(p)

        else:
            A = np.matrix([[1., x_0, x_0 ** 2, x_0 ** 3, x_0 ** 4],
                           [0., 1., 2 * x_0, 3 * x_0 ** 2, 4 * x_0 ** 3],
                           [0., 0., 2., 6 * x_0, 12 * x_0 ** 2],
                           [0., 1., 2. * x_1, 3 * x_1 ** 2, 4 * x_1 ** 3],
                           [0., 0., 2., 6 * x_1, 12 * x_1 ** 2]])
            Q = np.array([
                y_0,
                y_prime_0,
                y_pprime_0,
                y_prime_1,
                y_pprime_1
            ])
            P = A.I * Q

    else:
        # 5th-order Polynomial
        if isinstance(x_0, np.ndarray):
            if isinstance(y_0, np.ndarray):
                A = [[np.zeros((deg + 1, deg + 1)) for _ in range(x_0.shape[0])] for _ in range(x_0.shape[0])]
                for i, (start, end) in enumerate(zip(x_0, x_1)):
                    a = np.matrix([[1., start, start ** 2, start ** 3, start ** 4, start ** 5],
                                   [0., 1., 2 * start, 3 * start ** 2, 4 * start ** 3, 5 * start ** 4],
                                   [0., 0., 2., 6 * start, 12 * start ** 2, 20 * start ** 3],
                                   [1., end, end ** 2, end ** 3, end ** 4, end ** 5],
                                   [0., 1., 2 * end, 3 * end ** 2, 4 * end ** 3, 5 * end ** 4],
                                   [0., 0., 2., 6 * end, 12 * end ** 2, 20 * end ** 3]])
                    A[i][i] = a
                A = np.block(A)
                q = np.array([
                    y_0,
                    y_prime_0,
                    y_pprime_0,
                    y_1,
                    y_prime_1,
                    y_pprime_1
                ])
                Q = [[np.zeros_like(q) for _ in range(x_0.shape[0])] for _ in range(x_0.shape[0])]
                for i in range(x_0.shape[0]):
                    Q[i][i] = q
                Q = np.block(Q)
                P = A.I * Q
                P_list = []
                i = 0
                j = 0
                while i < P.shape[0]:
                    P_list.append(P[i:i + q.shape[0], j:j + q.shape[1]])
                    i += q.shape[0]
                    j += q.shape[1]
                P = P_list

            elif isinstance(y_0, list):
                P = []
                for i, (start, end) in enumerate(zip(x_0, x_1)):
                    a = np.matrix([[1., start, start ** 2, start ** 3, start ** 4, start ** 5],
                                   [0., 1., 2 * start, 3 * start ** 2, 4 * start ** 3, 5 * start ** 4],
                                   [0., 0., 2., 6 * start, 12 * start ** 2, 20 * start ** 3],
                                   [1., end, end ** 2, end ** 3, end ** 4, end ** 5],
                                   [0., 1., 2 * end, 3 * end ** 2, 4 * end ** 3, 5 * end ** 4],
                                   [0., 0., 2., 6 * end, 12 * end ** 2, 20 * end ** 3]])
                    q = np.array([
                        y_0[i].repeat(y_pprime_1.shape[0]),
                        y_prime_0[i].repeat(y_pprime_1.shape[0]),
                        y_pprime_0[i].repeat(y_pprime_1.shape[0]),
                        np.tile(y_1, y_0[i].shape[0]),
                        np.tile(y_prime_1, y_0[i].shape[0]),
                        np.tile(y_pprime_1, y_0[i].shape[0])
                    ])
                    p = a.I * q
                    P.append(p)

        else:
            A = np.matrix([[1., x_0, x_0 ** 2, x_0 ** 3, x_0 ** 4, x_0 ** 5],
                           [0., 1., 2 * x_0, 3 * x_0 ** 2, 4 * x_0 ** 3, 5 * x_0 ** 4],
                           [0., 0., 2., 6 * x_0, 12 * x_0 ** 2, 20 * x_0 ** 3],
                           [1., x_1, x_1 ** 2, x_1 ** 3, x_1 ** 4, x_1 ** 5],
                           [0., 1., 2 * x_1, 3 * x_1 ** 2, 4 * x_1 ** 3, 5 * x_1 ** 4],
                           [0., 0., 2., 6 * x_1, 12 * x_1 ** 2, 20 * x_1 ** 3]])
            Q = np.array([
                y_0,
                y_prime_0,
                y_pprime_0,
                y_1,
                y_prime_1,
                y_pprime_1
            ])
            P = A.I * Q

    return P


def polynomial(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial function: ``y(x) = params[deg, k] * x**deg + ... + params[0, k]``, k = 0, 2, ..., K-1
    :param x: shape (M,), x-coordinates of the M sample points ``(x[i], y[i])``.
    :param params: shape (deg + 1, K) or (deg + 1,), coefficients of `K` polynomials of degree `deg`.
    :return: shape (K, M) or (1, M), y-coordinates of the sample points.
    """
    deg = params.shape[0] - 1
    if deg == 3:
        return np.array([np.ones_like(x), x, x ** 2, x ** 3]).T.dot(params).T
    elif deg == 4:
        return np.array([np.ones_like(x), x, x ** 2, x ** 3, x ** 4]).T.dot(params).T
    elif deg == 5:
        return np.array([np.ones_like(x), x, x ** 2, x ** 3, x ** 4, x ** 5]).T.dot(params).T
    elif deg == 6:
        return np.array([np.ones_like(x), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6]).T.dot(params).T


def polynomial_derivative(x: np.ndarray, params: np.ndarray, order: int) -> np.ndarray:
    """
    Compute [order]th derivative of polynomial of degree [deg].
    :param x: x.
    :param params: polynomial coefficients, p[0] * x**deg + ... + p[deg].
    :param order: order of derivative.
    :return: derivatives.
    """
    deg = params.shape[0] - 1  # degree of polynomial
    if deg == 3:
        if order == 1:
            return np.array([np.zeros_like(x), np.ones_like(x), 2 * x, 3 * x ** 2]).T.dot(params).T
        elif order == 2:
            return np.array([np.zeros_like(x), np.zeros_like(x), 2 * np.ones_like(x), 6 * x]).T.dot(params).T
    elif deg == 4:
        if order == 1:
            return np.array([np.zeros_like(x), np.ones_like(x), 2 * x, 3 * x ** 2, 4 * x ** 3]).T.dot(params).T
        elif order == 2:
            return np.array([np.zeros_like(x), np.zeros_like(x), 2 * np.ones_like(x), 6 * x, 12 * x ** 2]).T.dot(params).T
    elif deg == 5:
        if order == 1:
            return np.array([np.zeros_like(x), np.ones_like(x), 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4]).T.dot(params).T
        elif order == 2:
            return np.array([np.zeros_like(x), np.zeros_like(x), 2 * np.ones_like(x), 6 * x, 12 * x ** 2, 20 * x ** 3]).T.dot(params).T
    elif deg == 6:
        if order == 1:
            return np.array([np.zeros_like(x), np.ones_like(x), 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4, 6 * x ** 5]).T.dot(params).T
        elif order == 2:
            return np.array([np.zeros_like(x), np.zeros_like(x), 2 * np.ones_like(x), 6 * x, 12 * x ** 2, 20 * x ** 3, 30 * x ** 4]).T.dot(params).T


def inner_product(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]


def outter_product(vector1, vector2):
    return vector1[0] * vector2[1] - vector2[0] * vector1[1]


def list_outter_product(vector1, vector2):
    return [v1[0] * v2[1] - v2[0] * v1[1] for v1, v2 in zip(vector1, vector2)]


def point_to_line_distance(pt: np.ndarray, pt1_of_line: np.ndarray, pt2_of_line: np.ndarray) -> np.ndarray:
    """
    Compute point to line distances.
    :param pt: shape (2,), coords of query point.
    :param pt1_of_line: shape (num_lines, 2), point 1 used to define lines.
    :param pt2_of_line: shape (num_lines, 2), point 2 used to define lines.
    :return: shape (num_lines,), point to line distances.
    """
    # Ax+By+C=0
    A = pt2_of_line[:, 1] - pt1_of_line[:, 1]
    B = pt1_of_line[:, 0] - pt2_of_line[:, 0]
    C = (pt2_of_line[:, 0] * pt1_of_line[:, 1] - pt1_of_line[:, 0] * pt2_of_line[:, 1])
    distance = np.abs(A * pt[0] + B * pt[1] + C) / np.sqrt(A ** 2 + B ** 2)
    return distance


def point_to_point_distance(pt: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Compute point to points distances.
    :param pt: shape (2,), coords of query point.
    :param pts: shape (num_points, 2), coords of key points.
    :return: shape (num_points,), point to points distances.
    """
    return np.linalg.norm(pt - pts, ord=2, axis=1)


def wrap_angle(
        angle: Union[torch.Tensor, np.ndarray],
        min_val: float = -math.pi,
        max_val: float = math.pi) -> Union[torch.Tensor, np.ndarray]:
    return min_val + (angle + max_val) % (max_val - min_val)


def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def angle_between_2d_vectors_no_atan2(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    idx = torch.where(x == 0)
    x[idx] = 1e-6
    angle = torch.atan(y / x)
    mask_yp = (y >= 0) & (x < 0)
    mask_yn = (y < 0) & (x < 0)
    idx_yp = torch.where(mask_yp)
    idx_yn = torch.where(mask_yn)
    angle[idx_yp] += math.pi
    angle[idx_yn] -= math.pi
    return angle


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def stack_first_2_dims(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    if len(shape) == 2:
        tensor = tensor.reshape((-1))
    elif len(shape) == 3:
        tensor = tensor.reshape((-1, shape[2]))
    elif len(shape) == 4:
        tensor = tensor.reshape((-1, shape[2], shape[3]))
    elif len(shape) == 5:
        tensor = tensor.reshape((-1, shape[2], shape[3], shape[4]))

    return tensor


def efficient_relative_to_absolute_poses(anchor_poses: Union[np.ndarray, torch.Tensor],
                                         relative_poses: Union[np.ndarray, torch.Tensor]):
    """
    efficient version of nuplan.common.geometry.convert.relative_to_absolute_poses
    :param anchor_poses: shape (num_agents, 3)
    :param relative_poses: shape (num_agents, num_modes, num_poses, 3)
    :return:
    """
    num_modes, num_poses = relative_poses.shape[1:3]
    if isinstance(anchor_poses, np.ndarray):
        relative_transforms = np.array(
            [
                [np.cos(relative_poses[:, :, :, 2]), -np.sin(relative_poses[:, :, :, 2]), relative_poses[:, :, :, 0]],
                [np.sin(relative_poses[:, :, :, 2]), np.cos(relative_poses[:, :, :, 2]), relative_poses[:, :, :, 1]],
                [np.zeros_like(relative_poses[:, :, :, 2]), np.zeros_like(relative_poses[:, :, :, 2]), np.ones_like(relative_poses[:, :, :, 2])],
            ]
        ).transpose(2, 3, 4, 0, 1)
        origin_transform = np.array(
            [
                [np.cos(anchor_poses[:, 2]), -np.sin(anchor_poses[:, 2]), anchor_poses[:, 0]],
                [np.sin(anchor_poses[:, 2]), np.cos(anchor_poses[:, 2]), anchor_poses[:, 1]],
                [np.zeros_like(anchor_poses[:, 2]), np.zeros_like(anchor_poses[:, 2]), np.ones_like(anchor_poses[:, 2])],
            ]
        ).transpose(2, 0, 1)
        origin_transform = np.expand_dims(origin_transform, axis=(1, 2))
        origin_transform = np.broadcast_to(origin_transform, (origin_transform.shape[0],
                                                              num_modes,
                                                              num_poses,
                                                              origin_transform.shape[3],
                                                              origin_transform.shape[4]))
        absolute_transforms = origin_transform @ relative_transforms
        heading = np.arctan2(absolute_transforms[:, :, :, 1, 0], absolute_transforms[:, :, :, 0, 0])
        absolute_poses = np.stack([absolute_transforms[:, :, :, 0, 2], absolute_transforms[:, :, :, 1, 2], heading], axis=-1)

    elif isinstance(anchor_poses, torch.Tensor):
        relative_transforms = torch.stack(
            [
                torch.stack([relative_poses[:, :, :, 2].cos(), -relative_poses[:, :, :, 2].sin(), relative_poses[:, :, :, 0]]),
                torch.stack([relative_poses[:, :, :, 2].sin(), relative_poses[:, :, :, 2].cos(), relative_poses[:, :, :, 1]]),
                torch.stack([torch.zeros_like(relative_poses[:, :, :, 2]), torch.zeros_like(relative_poses[:, :, :, 2]), torch.ones_like(relative_poses[:, :, :, 2])]),
            ]
        ).permute(2, 3, 4, 0, 1)
        origin_transform = torch.stack(
            [
                torch.stack([anchor_poses[:, 2].cos(), -anchor_poses[:, 2].sin(), anchor_poses[:, 0]]),
                torch.stack([anchor_poses[:, 2].sin(), anchor_poses[:, 2].cos(), anchor_poses[:, 1]]),
                torch.stack([torch.zeros_like(anchor_poses[:, 2]), torch.zeros_like(anchor_poses[:, 2]), torch.ones_like(anchor_poses[:, 2])]),
            ]
        ).permute(2, 0, 1)
        origin_transform = origin_transform.unsqueeze(1).unsqueeze(2)
        absolute_transforms = origin_transform @ relative_transforms
        heading = torch.atan2(absolute_transforms[:, :, :, 1, 0], absolute_transforms[:, :, :, 0, 0])
        absolute_poses = torch.stack([absolute_transforms[:, :, :, 0, 2], absolute_transforms[:, :, :, 1, 2], heading], dim=-1)

    return absolute_poses


def efficient_absolute_to_relative_poses(anchor_poses: Optional[Union[np.ndarray, torch.Tensor]], absolute_poses: Optional[Union[np.ndarray, torch.Tensor]]):
    """
    efficient version of nuplan.common.geometry.convert.relative_to_absolute_poses
    :param anchor_poses: shape (num_agents, 3)
    :param absolute_poses: shape (num_agents, num_modes, num_poses, 3)
    :return:
    """
    num_modes, num_poses = absolute_poses.shape[1:3]

    if isinstance(anchor_poses, np.ndarray):
        absolute_transforms = np.array(
            [
                [np.cos(absolute_poses[:, :, :, 2]), -np.sin(absolute_poses[:, :, :, 2]), absolute_poses[:, :, :, 0]],
                [np.sin(absolute_poses[:, :, :, 2]), np.cos(absolute_poses[:, :, :, 2]), absolute_poses[:, :, :, 1]],
                [np.zeros_like(absolute_poses[:, :, :, 2]), np.zeros_like(absolute_poses[:, :, :, 2]), np.ones_like(absolute_poses[:, :, :, 2])],
            ]
        ).transpose(2, 3, 4, 0, 1)
        origin_transform = np.array(
            [
                [np.cos(anchor_poses[:, 2]), -np.sin(anchor_poses[:, 2]), anchor_poses[:, 0]],
                [np.sin(anchor_poses[:, 2]), np.cos(anchor_poses[:, 2]), anchor_poses[:, 1]],
                [np.zeros_like(anchor_poses[:, 2]), np.zeros_like(anchor_poses[:, 2]), np.ones_like(anchor_poses[:, 2])],
            ]
        ).transpose(2, 0, 1)
        origin_transform = np.linalg.inv(origin_transform)
        origin_transform = np.expand_dims(origin_transform, axis=(1, 2))
        # origin_transform = np.broadcast_to(origin_transform, (origin_transform.shape[0],
        #                                                       num_modes,
        #                                                       num_poses,
        #                                                       origin_transform.shape[3],
        #                                                       origin_transform.shape[4]))
        absolute_transforms = origin_transform @ absolute_transforms
        heading = np.arctan2(absolute_transforms[:, :, :, 1, 0], absolute_transforms[:, :, :, 0, 0])
        relative_poses = np.stack([absolute_transforms[:, :, :, 0, 2], absolute_transforms[:, :, :, 1, 2], heading], axis=-1)

    elif isinstance(anchor_poses, torch.Tensor):
        assert anchor_poses.dtype == torch.float64, \
            'use torch.float64 instead of torch.float32 because torch.float32 cause wrong results'
        absolute_transforms = torch.stack(
            [
                torch.stack([absolute_poses[:, :, :, 2].cos(), -absolute_poses[:, :, :, 2].sin(), absolute_poses[:, :, :, 0]]),
                torch.stack([absolute_poses[:, :, :, 2].sin(), absolute_poses[:, :, :, 2].cos(), absolute_poses[:, :, :, 1]]),
                torch.stack([torch.zeros_like(absolute_poses[:, :, :, 2]), torch.zeros_like(absolute_poses[:, :, :, 2]), torch.ones_like(absolute_poses[:, :, :, 2])]),
            ]
        ).permute(2, 3, 4, 0, 1)
        origin_transform = torch.stack(
            [
                torch.stack([anchor_poses[:, 2].cos(), -anchor_poses[:, 2].sin(), anchor_poses[:, 0]]),
                torch.stack([anchor_poses[:, 2].sin(), anchor_poses[:, 2].cos(), anchor_poses[:, 1]]),
                torch.stack([torch.zeros_like(anchor_poses[:, 2]), torch.zeros_like(anchor_poses[:, 2]), torch.ones_like(anchor_poses[:, 2])]),
            ]
        ).permute(2, 0, 1)
        origin_transform = torch.linalg.inv(origin_transform.cpu()).to(origin_transform.device)
        origin_transform = origin_transform.unsqueeze(1).unsqueeze(2)
        absolute_transforms = origin_transform @ absolute_transforms
        heading = torch.atan2(absolute_transforms[:, :, :, 1, 0], absolute_transforms[:, :, :, 0, 0])
        relative_poses = torch.stack([absolute_transforms[:, :, :, 0, 2], absolute_transforms[:, :, :, 1, 2], heading], dim=-1)

    return relative_poses


def split_list(input_list, chunk_sizes):
    result = []
    index = 0

    for size in chunk_sizes:
        chunk = input_list[index:index + size]
        result.append(chunk)
        index += size

    return result


def read_parquets(dirs: List[str]):
    files = [os.listdir(dir)[0] for dir in dirs]
    dfs = [pd.read_parquet(os.path.join(dir, file)) for dir, file in zip(dirs, files)]
    metadata_dfs = [df[df['log_name'].notna()] for df in dfs]
    if len(dfs) > 1:
        summary_dfs = [df.iloc[-1:] for df in dfs]
        metadata_df = pd.concat(metadata_dfs).reset_index(drop=True)
        summaries_df = pd.concat(summary_dfs).reset_index(drop=True)
        summary_df = summaries_df.iloc[:1]
        summary_df.loc[0, 'num_scenarios'] = summaries_df['num_scenarios'].sum()
        summary_df.loc[0, 'drivable_area_compliance'] = summaries_df['drivable_area_compliance'].mean()
        summary_df.loc[0, 'driving_direction_compliance'] = summaries_df['driving_direction_compliance'].mean()
        summary_df.loc[0, 'ego_is_comfortable'] = summaries_df['ego_is_comfortable'].mean()
        summary_df.loc[0, 'ego_is_making_progress'] = summaries_df['ego_is_making_progress'].mean()
        summary_df.loc[0, 'ego_progress_along_expert_route'] = summaries_df['ego_progress_along_expert_route'].mean()
        summary_df.loc[0, 'no_ego_at_fault_collisions'] = summaries_df['no_ego_at_fault_collisions'].mean()
        summary_df.loc[0, 'speed_limit_compliance'] = summaries_df['speed_limit_compliance'].mean()
        summary_df.loc[0, 'time_to_collision_within_bound'] = summaries_df['time_to_collision_within_bound'].mean()
        summary_df.loc[0, 'score'] = summaries_df['score'].mean()
    else:
        summary_dfs = [df.iloc[-1:] for df in dfs]
        metadata_df = pd.concat(metadata_dfs).reset_index(drop=True)
        summaries_df = pd.concat(summary_dfs).reset_index(drop=True)
        summary_df = summaries_df.iloc[:1]
    return {
        'metadata': metadata_df,
        'summary': summary_df
    }


def compute_ade_with_simulation_log(dirs: List[str]):
    ade = []
    for dir in dirs:
        scenario_types = os.listdir(dir)
        for scenario_type in scenario_types:
            log_names = os.listdir(os.path.join(dir, scenario_type))
            for log_name in log_names:
                scenario_tokens = os.listdir(os.path.join(dir, scenario_type, log_name))
                for scenario_token in scenario_tokens:
                    path = Path(os.path.join(dir, scenario_type, log_name, scenario_token, scenario_token + '.msgpack.xz'))
                    log_data = SimulationLog.load_data(path)
                    expert_ego_state_trajectory = list(log_data.scenario.get_expert_ego_trajectory())
                    expert_ego_trajectory = [state.center.array for state in expert_ego_state_trajectory]
                    expert_ego_trajectory = np.array(expert_ego_trajectory)
                    simulated_ego_state_trajectory = expert_ego_state_trajectory[:1] + log_data.simulation_history.extract_ego_state
                    simulated_ego_trajectory = [state.center.array for state in simulated_ego_state_trajectory]
                    simulated_ego_trajectory = np.array(simulated_ego_trajectory)
                    ade.append(np.linalg.norm(expert_ego_trajectory - simulated_ego_trajectory, ord=2, axis=1).mean())

    return np.mean(ade)


def load_simulation_log_given_scenario_token(dirs: List[str], token: str):
    for dir in dirs:
        scenario_types = os.listdir(dir)
        for scenario_type in scenario_types:
            log_names = os.listdir(os.path.join(dir, scenario_type))
            for log_name in log_names:
                scenario_tokens = os.listdir(os.path.join(dir, scenario_type, log_name))
                for scenario_token in scenario_tokens:
                    if token not in scenario_token:
                        continue
                    path = Path(os.path.join(dir, scenario_type, log_name, scenario_token, scenario_token + '.msgpack.xz'))
                    log_data = SimulationLog.load_data(path)
                    expert_ego_state_trajectory = list(log_data.scenario.get_expert_ego_trajectory())
                    # expert_ego_trajectory = [state.center.array for state in expert_ego_state_trajectory]
                    # expert_ego_heading = [state.center.heading for state in expert_ego_state_trajectory]
                    # expert_ego_velocity_x = [state.dynamic_car_state.center_velocity_2d.x for state in expert_ego_state_trajectory]
                    # expert_ego_velocity_y = [state.dynamic_car_state.center_velocity_2d.y for state in expert_ego_state_trajectory]
                    # expert_ego_acceleration_x = [state.dynamic_car_state.center_acceleration_2d.x for state in expert_ego_state_trajectory]
                    # expert_ego_acceleration_y = [state.dynamic_car_state.center_acceleration_2d.y for state in expert_ego_state_trajectory]
                    expert_ego_trajectory = [state.rear_axle.array for state in expert_ego_state_trajectory]
                    expert_ego_heading = [state.rear_axle.heading for state in expert_ego_state_trajectory]
                    expert_ego_velocity_x = [state.dynamic_car_state.rear_axle_velocity_2d.x for state in expert_ego_state_trajectory]
                    expert_ego_velocity_y = [state.dynamic_car_state.rear_axle_velocity_2d.y for state in expert_ego_state_trajectory]
                    expert_ego_acceleration_x = [state.dynamic_car_state.rear_axle_acceleration_2d.x for state in expert_ego_state_trajectory]
                    expert_ego_acceleration_y = [state.dynamic_car_state.rear_axle_acceleration_2d.y for state in expert_ego_state_trajectory]
                    expert_ego_trajectory = np.array(expert_ego_trajectory)
                    expert_ego_heading = np.array(expert_ego_heading)
                    expert_ego_velocity_x = np.array(expert_ego_velocity_x)
                    expert_ego_velocity_y = np.array(expert_ego_velocity_y)
                    expert_ego_speed = np.sqrt(expert_ego_velocity_x ** 2 + expert_ego_velocity_y ** 2)
                    expert_ego_acceleration_x = np.array(expert_ego_acceleration_x)
                    expert_ego_acceleration_y = np.array(expert_ego_acceleration_y)
                    expert_ego_acceleration = np.sqrt(expert_ego_acceleration_x ** 2 + expert_ego_acceleration_y ** 2)
                    simulated_ego_state_trajectory = expert_ego_state_trajectory[:1] + log_data.simulation_history.extract_ego_state
                    # simulated_ego_trajectory = [state.center.array for state in simulated_ego_state_trajectory]
                    # simulated_ego_heading = [state.center.heading for state in simulated_ego_state_trajectory]
                    # simulated_ego_velocity_x = [state.dynamic_car_state.center_velocity_2d.x for state in simulated_ego_state_trajectory]
                    # simulated_ego_velocity_y = [state.dynamic_car_state.center_velocity_2d.y for state in simulated_ego_state_trajectory]
                    # simulated_ego_acceleration_x = [state.dynamic_car_state.center_acceleration_2d.x for state in simulated_ego_state_trajectory]
                    # simulated_ego_acceleration_y = [state.dynamic_car_state.center_acceleration_2d.y for state in simulated_ego_state_trajectory]
                    simulated_ego_trajectory = [state.rear_axle.array for state in simulated_ego_state_trajectory]
                    simulated_ego_heading = [state.rear_axle.heading for state in simulated_ego_state_trajectory]
                    simulated_ego_velocity_x = [state.dynamic_car_state.rear_axle_velocity_2d.x for state in simulated_ego_state_trajectory]
                    simulated_ego_velocity_y = [state.dynamic_car_state.rear_axle_velocity_2d.y for state in simulated_ego_state_trajectory]
                    simulated_ego_acceleration_x = [state.dynamic_car_state.rear_axle_acceleration_2d.x for state in simulated_ego_state_trajectory]
                    simulated_ego_acceleration_y = [state.dynamic_car_state.rear_axle_acceleration_2d.y for state in simulated_ego_state_trajectory]
                    simulated_ego_heading = np.array(simulated_ego_heading)
                    simulated_ego_velocity_x = np.array(simulated_ego_velocity_x)
                    simulated_ego_velocity_y = np.array(simulated_ego_velocity_y)
                    simulated_ego_speed = np.sqrt(simulated_ego_velocity_x ** 2 + simulated_ego_velocity_y ** 2)
                    simulated_ego_acceleration_x = np.array(simulated_ego_acceleration_x)
                    simulated_ego_acceleration_y = np.array(simulated_ego_acceleration_y)
                    simulated_ego_acceleration = np.sqrt(simulated_ego_acceleration_x ** 2 + simulated_ego_acceleration_y ** 2)
                    simulated_ego_trajectory = np.array(simulated_ego_trajectory)

                    return {
                        'expert': {
                            'x [m]': expert_ego_trajectory[:, 0],
                            'y [m]': expert_ego_trajectory[:, 1],
                            'heading [rad]': expert_ego_heading,
                            'velocity_x [m/s]': expert_ego_velocity_x,
                            'velocity_y [m/s]': expert_ego_velocity_y,
                            'speed [m/s]': expert_ego_speed,
                            'acceleration_x [m/s^2]': expert_ego_acceleration_x,
                            'acceleration_y [m/s^2]': expert_ego_acceleration_y,
                            'acceleration [m/s^2]': expert_ego_acceleration,
                        },
                        'ego': {
                            'x [m]': simulated_ego_trajectory[:, 0],
                            'y [m]': simulated_ego_trajectory[:, 1],
                            'heading [rad]': simulated_ego_heading,
                            'velocity_x [m/s]': simulated_ego_velocity_x,
                            'velocity_y [m/s]': simulated_ego_velocity_y,
                            'speed [m/s]': simulated_ego_speed,
                            'acceleration_x [m/s^2]': simulated_ego_acceleration_x,
                            'acceleration_y [m/s^2]': simulated_ego_acceleration_y,
                            'acceleration [m/s^2]': simulated_ego_acceleration,
                        },
                    }


def to_tensor(data):
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            return torch.from_numpy(data).float()
        else:
            return torch.from_numpy(data)
    elif isinstance(data, np.number):
        return torch.tensor(data).float()
    elif isinstance(data, list):
        return data
    elif isinstance(data, int):
        return torch.tensor(data)
    elif isinstance(data, tuple):
        return to_tensor(data[0])
    else:
        print(type(data), data)
        raise NotImplementedError


def to_numpy(data):
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            return data.detach().cpu().numpy()
        else:
            return data.cpu().numpy()
    else:
        print(type(data), data)
        raise NotImplementedError


def enable_grad(data):
    if isinstance(data, dict):
        return {k: enable_grad(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.float32:
            data.requires_grad = True
    else:
        raise NotImplementedError


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise NotImplementedError


def print_dict_tensor(data, prefix=""):
    for k, v in data.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"{prefix}{k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict_tensor(v, "    ")


def print_simulation_results(file=None):
    if file is not None:
        df = pd.read_parquet(file)
    else:
        root = Path(os.getcwd()) / "aggregator_metric"
        result = list(root.glob("*.parquet"))
        result = max(result, key=lambda item: item.stat().st_ctime)
        df = pd.read_parquet(result)
    final_score = df[df["scenario"] == "final_score"]
    final_score = final_score.to_dict(orient="records")[0]
    pprint.PrettyPrinter(indent=4).pprint(final_score)


def load_checkpoint(checkpoint):
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    return state_dict


def safe_index(ls, value):
    try:
        return ls.index(value)
    except ValueError:
        return None


def shift_and_rotate_img(img, shift, angle, resolution, cval=-200):
    """
    img: (H, W, C)
    shift: (H_shift, W_shift, 0)
    resolution: float
    angle: float
    """
    rows, cols = img.shape[:2]
    shift = shift / resolution
    translation_matrix = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    translated_img = cv2.warpAffine(
        img, translation_matrix, (cols, rows), borderValue=cval
    )
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
    rotated_img = cv2.warpAffine(translated_img, M, (cols, rows), borderValue=cval)
    if len(img.shape) == 3 and len(rotated_img.shape) == 2:
        rotated_img = rotated_img[..., np.newaxis]
    return rotated_img.astype(np.float32)


def crop_img_from_center(img, crop_size):
    h, w = img.shape[:2]
    h_crop, w_crop = crop_size
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    return img[h_start : h_start + h_crop, w_start : w_start + w_crop].astype(
        np.float32
    )


def smooth_headings(trajs, use_savgol_filter=False, window_length=11, polyorder=2):
    """
    smooth headings
    :param trajs: shape (num_trajs, num_poses, 3), num_poses=1+num_future_poses
    :return: smoothed trajs
    """
    # deal with slow speed
    slow_mask = np.sqrt((trajs[:, 1:, 0] - trajs[:, :-1, 0]) ** 2) < 0.2
    slow_mask = np.concatenate([slow_mask, slow_mask[:, -1:]], axis=1)
    slow_mask[:, 0] = False
    false_to_true = slow_mask[:, 1:] & ~slow_mask[:, :-1]
    heading = trajs[:, :, 2]
    for i in range(trajs.shape[0]):
        heading[i, 1:][slow_mask[i, 1:]] = 0.
        f2t = np.where(false_to_true[i])[0]
        for idx in f2t:
            heading[i, idx + 1:][slow_mask[i, idx + 1:]] = heading[i, idx]

    # use savgol filter
    if use_savgol_filter:
        heading_ori = trajs[:, :, 2]
        smoothed_heading = scipy.signal.savgol_filter(heading_ori, window_length=window_length, polyorder=polyorder, axis=1)
        trajs[:, :, 2] = smoothed_heading

    return trajs