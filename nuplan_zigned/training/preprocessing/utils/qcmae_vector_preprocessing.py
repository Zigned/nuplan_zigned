from typing import List, Optional, Tuple, Dict

import torch


def interpolate_points(coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
    """
    Interpolate points within map element to maintain fixed size.
    :param coords: Sequence of coordinate points representing map element. <torch.Tensor: num_points, 2>
    :param max_points: Desired size to interpolate to.
    :param interpolation: Torch interpolation mode. Available options: 'linear' and 'area'.
    :return: Coordinate points interpolated to max_points size.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, 2)")

    x_coords = coords[:, 0].unsqueeze(0).unsqueeze(0)
    y_coords = coords[:, 1].unsqueeze(0).unsqueeze(0)
    align_corners = True if interpolation == 'linear' else None
    x_coords = torch.nn.functional.interpolate(x_coords, max_points, mode=interpolation, align_corners=align_corners)
    y_coords = torch.nn.functional.interpolate(y_coords, max_points, mode=interpolation, align_corners=align_corners)
    coords = torch.stack((x_coords, y_coords), dim=-1).squeeze()

    return coords


def convert_lane_layers_to_consistent_size(
    centerline_coords: List[Dict[str, torch.Tensor]],
    left_boundary_coords: List[Dict[str, torch.Tensor]],
    right_boundary_coords: List[Dict[str, torch.Tensor]],
    max_points: Dict[str, int],
    interpolation: Optional[str],
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Converts variable sized map features to the max size tensors.
        Points per boundary are interpolated to maintain the same size as the centerline.
    :param centerline_coords: Vector set of centerline coordinates for collection of lane elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param left_boundary_coords: Vector set of left boundary coordinates for collection of lane elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param right_boundary_coords: Vector set of right boundary coordinates for collection of lane elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param max_points: maximum number of points per feature to extract per feature layer.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options:
        'linear' and 'area'.
    :return
        centerline_coords_output: The converted coords.
        left_boundary_coords_output: The converted coords.
        right_boundary_coords_output: The converted coords.
    :raise ValueError: If centerline and boundary coordinates size do not match.
    """
    centerline_coords_output = {}
    left_boundary_coords_output = {}
    right_boundary_coords_output = {}

    for lane_id in centerline_coords[0].keys():
        num_points = centerline_coords[0][lane_id].shape[0]
        max_num_points = min(num_points, max_points['LANE'])
        center_coords = interpolate_points(centerline_coords[0][lane_id], max_num_points, interpolation=interpolation)
        left_coords = interpolate_points(left_boundary_coords[0][lane_id], max_num_points, interpolation=interpolation)
        right_coords = interpolate_points(right_boundary_coords[0][lane_id], max_num_points, interpolation=interpolation)

        centerline_coords_output[lane_id] = center_coords
        left_boundary_coords_output[lane_id] = left_coords
        right_boundary_coords_output[lane_id] = right_coords

    return [centerline_coords_output], [left_boundary_coords_output], [right_boundary_coords_output]
