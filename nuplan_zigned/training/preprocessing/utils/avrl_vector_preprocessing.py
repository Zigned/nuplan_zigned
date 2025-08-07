from typing import List, Dict, Optional, Tuple

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


def convert_feature_layer_to_fixed_size(
    feature_coords: List[torch.Tensor],
    feature_tl_data_over_time: Optional[List[List[torch.Tensor]]],
    max_elements: int,
    max_points: int,
    traffic_light_encoding_dim: int,
    interpolation: Optional[str],
    neighbor_data: Optional[Dict],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[str], Dict[str, List]]:
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data_over_time: Optional traffic light status corresponding to map elements at given index in coords.
        [num_frames, num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options:
        'linear' and 'area'.
    :param neighbor_data: Neighbor data.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
        map_obj_ids: Map object ids.
        neighbor_ids: Lane ids of the lane's left neighbor lane, right neighbor lane, predecessor, and successor.
    :raise ValueError: If coordinates and traffic light data size do not match.
    """
    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float64)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    map_obj_ids = []
    if neighbor_data is not None:
        neighbor_ids = {
            'LANE_TYPE': [],
            'LEFT_NEIGHBOR_LANE_ID': [],
            'RIGHT_NEIGHBOR_LANE_ID': [],
            'PREDECESSOR_ID': [],
            'SUCCESSOR_ID': [],
        }
    else:
        neighbor_ids = None
    tl_data_tensor = (
        torch.zeros(
            (len(feature_tl_data_over_time), max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32
        )
        if feature_tl_data_over_time is not None
        else None
    )
    # tl_data_tensor: Tensor<num_frames, max_elements, max_points, traffic_light_encoding_dim>

    feature_coords_list = list(feature_coords.items())
    for element_idx in range(min(len(feature_coords), max_elements)):
        element_id = feature_coords_list[element_idx][0]
        element_coords = feature_coords_list[element_idx][1]

        # interpolate to maintain fixed size according to specified interpolation method if specified
        if interpolation is not None:
            num_points = max_points
            element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        # otherwise trim/zero-pad points to maintain fixed size
        else:
            num_points = min(len(element_coords), max_points)
            element_coords = element_coords[:num_points]

        coords_tensor[element_idx, :num_points] = element_coords
        avails_tensor[element_idx, :num_points] = True  # specify real vs zero-padded data
        map_obj_ids.append(element_id)
        if neighbor_data is not None:
            neighbor_ids['LANE_TYPE'].append(neighbor_data['LANE_TYPE'][element_idx])
            neighbor_ids['LEFT_NEIGHBOR_LANE_ID'].append(neighbor_data['LEFT_NEIGHBOR_LANE_ID'][element_idx])
            neighbor_ids['RIGHT_NEIGHBOR_LANE_ID'].append(neighbor_data['RIGHT_NEIGHBOR_LANE_ID'][element_idx])
            neighbor_ids['PREDECESSOR_ID'].append(neighbor_data['PREDECESSOR_ID'][element_idx])
            neighbor_ids['SUCCESSOR_ID'].append(neighbor_data['SUCCESSOR_ID'][element_idx])

        if (feature_tl_data_over_time is not None) and (tl_data_tensor is not None):
            for time_ind in range(len(feature_tl_data_over_time)):
                if len(feature_coords) != len(feature_tl_data_over_time[time_ind]):
                    raise ValueError(
                        f"num_elements between feature_coords and feature_tl_data_over_time inconsistent: {len(feature_coords)}, {len(feature_tl_data_over_time[time_ind])}"
                    )
                tl_data_tensor[time_ind, element_idx, :num_points] = list(feature_tl_data_over_time[time_ind].values())[element_idx]

    return coords_tensor, tl_data_tensor, avails_tensor, map_obj_ids, neighbor_ids
