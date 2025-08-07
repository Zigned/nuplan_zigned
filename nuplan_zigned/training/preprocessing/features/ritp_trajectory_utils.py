from typing import List, Union
import torch

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2


def _convert_absolute_to_relative_states(
    origin_absolute_state: Union[StateSE2, torch.Tensor],
    absolute_states: Union[List[StateSE2], torch.Tensor],
) -> Union[List[StateSE2], torch.Tensor]:
    """
    Computes the relative states from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative states.
    """
    if isinstance(origin_absolute_state, StateSE2):
        origin_absolute_transform = origin_absolute_state.as_matrix()
        origin_transform = np.linalg.inv(origin_absolute_transform)

        absolute_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in absolute_states])
        relative_transforms = origin_transform @ absolute_transforms.reshape(-1, 3, 3)

        relative_states = [StateSE2.from_matrix(transform) for transform in relative_transforms]
    else:
        origin_absolute_transform = as_matrix(origin_absolute_state,
                                              dtype=torch.float,
                                              device=torch.device('cpu'))
        origin_transform = torch.linalg.inv(origin_absolute_transform)

        absolute_transforms = torch.stack([as_matrix(pose, dtype=torch.float, device=torch.device('cpu')) for pose in absolute_states])
        relative_transforms = origin_transform @ absolute_transforms.reshape(-1, 3, 3)

        relative_states = torch.stack([from_matrix(transform) for transform in relative_transforms]).to(origin_absolute_state.device)

    return relative_states


def _convert_relative_to_absolute_states(
    origin_absolute_state: StateSE2,
    relative_states: List[StateSE2],
) -> List[StateSE2]:
    """
    Computes the absolute states from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of relative poses.
    :return: list of absolute states.
    """
    origin_transform = origin_absolute_state.as_matrix()

    relative_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in relative_states])
    absolute_transforms = origin_transform @ relative_transforms.reshape(-1, 3, 3)

    absolute_states = [StateSE2.from_matrix(transform) for transform in absolute_transforms]

    return absolute_states


def convert_absolute_to_relative_poses(
        origin_absolute_state: Union[StateSE2, torch.Tensor],
        absolute_states: Union[List[StateSE2], torch.Tensor]
) -> Union[npt.NDArray[np.float32], torch.Tensor]:
    """
    Computes the relative poses from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_state, absolute_states)

    if isinstance(origin_absolute_state, StateSE2):
        relative_poses: npt.NDArray[np.float32] = np.asarray([state.serialize() for state in relative_states]).astype(
            np.float32
        )
    else:
        relative_poses = relative_states

    return relative_poses


def convert_relative_to_absolute_poses(
    origin_absolute_state: StateSE2, relative_states: List[StateSE2]
) -> npt.NDArray[np.float64]:
    """
    Computes the absolute poses from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    absolute_states = _convert_relative_to_absolute_states(origin_absolute_state, relative_states)
    absolute_poses: npt.NDArray[np.float64] = np.asarray([state.serialize() for state in absolute_states]).astype(
        np.float64
    )

    return absolute_poses


def convert_absolute_to_relative_velocities(
    origin_absolute_velocity: StateSE2, absolute_velocities: List[StateSE2]
) -> npt.NDArray[np.float32]:
    """
    Computes the relative velocities from a list of absolute velocities using an origin (anchor) velocity.

    :param origin_absolute_velocity: absolute velocities to be used as origin.
    :param absolute_velocities: list of absolute velocities.
    :return: list of relative velocities as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_velocity, absolute_velocities)
    relative_velocities: npt.NDArray[np.float32] = np.asarray([[state.x, state.y] for state in relative_states]).astype(
        np.float32
    )

    return relative_velocities


def as_matrix(pose: torch.Tensor, dtype=torch.float, device=torch.device('cpu')) -> torch.Tensor:
    return torch.tensor(
        [
                [pose[2].cos(), -pose[2].sin(), pose[0]],
                [pose[2].sin(), pose[2].cos(), pose[1]],
                [0.0, 0.0, 1.0],
            ],
        dtype=dtype,
        device=device
    )


def from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    :param matrix: 3x3 2D transformation matrix
    :return: StateSE2 object
    """
    assert matrix.shape == (3, 3), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

    vector = matrix.new_tensor([matrix[0, 2], matrix[1, 2], torch.atan2(matrix[1, 0], matrix[0, 0])])
    return vector