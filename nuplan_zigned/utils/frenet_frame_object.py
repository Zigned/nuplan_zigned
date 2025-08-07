import numpy as np
import math
from functools import cached_property
from typing import Tuple, List, cast, Union, Optional, Dict
from shapely.geometry import LineString, Point

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.nuplan_map.utils import estimate_curvature_along_path, extract_discrete_polyline
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives
from sympy import Point3D

from nuplan_zigned.utils.utils import inner_product, outter_product, list_outter_product
from nuplan_zigned.utils.utils import wrap_angle



def _get_heading(pt1: Union[Point, np.ndarray, List[Point]],
                 pt2: Union[Point, np.ndarray, List[Point]]) -> Union[float, np.ndarray, List[float]]:
    """
    Computes the angle two points makes to the x-axis.
    :param pt1: origin point.
    :param pt2: end point.
    :return: [rad] resulting angle.
    """
    if isinstance(pt1, Point):
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y
        return math.atan2(y_diff, x_diff)
    elif isinstance(pt1, np.ndarray):
        if len(pt1.shape) == 3:
            x_diff = pt2[:, 0, :] - pt1[:, 0, :]
            y_diff = pt2[:, 1, :] - pt1[:, 1, :]
        elif len(pt1.shape) == 2 or len(pt1.shape) == 1:
            x_diff = pt2[0] - pt1[0]
            y_diff = pt2[1] - pt1[1]
        return np.arctan2(y_diff, x_diff)
    elif isinstance(pt1, list) and isinstance(pt1[0], Point):
        x_diff = [p2.x - p1.x for p1, p2 in zip(pt1, pt2)]
        y_diff = [p2.y - p1.y for p1, p2 in zip(pt1, pt2)]
        return [math.atan2(y_d, x_d) for x_d, y_d in zip(x_diff, y_diff)]


def _get_sin(pt1: Union[Point, np.ndarray], pt2: Union[Point, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(pt1, Point):
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y
        return y_diff / math.sqrt(x_diff ** 2 + y_diff ** 2 + 1e-8)
    elif isinstance(pt1, np.ndarray):
        if len(pt1.shape) == 3:
            x_diff = pt2[:, 0, :] - pt1[:, 0, :]
            y_diff = pt2[:, 1, :] - pt1[:, 1, :]
        elif len(pt1.shape) == 2 or len(pt1.shape) == 1:
            x_diff = pt2[0] - pt1[0]
            y_diff = pt2[1] - pt1[1]
        return y_diff / np.sqrt(x_diff ** 2 + y_diff ** 2 + 1e-8)


def _get_cos(pt1: Union[Point, np.ndarray], pt2: Union[Point, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(pt1, Point):
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y
        return x_diff / math.sqrt(x_diff ** 2 + y_diff ** 2 + 1e-8)
    elif isinstance(pt1, np.ndarray):
        if len(pt1.shape) == 3:
            x_diff = pt2[:, 0, :] - pt1[:, 0, :]
            y_diff = pt2[:, 1, :] - pt1[:, 1, :]
        elif len(pt1.shape) == 2 or len(pt1.shape) == 1:
            x_diff = pt2[0] - pt1[0]
            y_diff = pt2[1] - pt1[1]
        return x_diff / np.sqrt(x_diff ** 2 + y_diff ** 2 + 1e-8)


class FrenetFrame:
    """
    Frenet frame object
    """

    def __init__(
        self,
        reference_line_coords: np.ndarray,
        distance_for_curvature_estimation: float = 2.0,
        distance_for_heading_estimation: float = 0.5,
    ):
        """
        Constructor of Frenet frame.
        :param reference_line_coords: coordinates in global frame.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the reference_line to calculate
                                                    the relative heading.
        """
        self._reference_line: LineString = LineString(reference_line_coords)
        assert self.reference_line.length > 0.0, "The length of the reference_line has to be greater than 0!"

        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def reference_line(self) -> LineString:
        """
        Returns the reference line as a Linestring.
        :return: The reference line as a Linestring.
        """
        return self._reference_line

    @property
    def distance_for_heading_estimation(self):
        return self._distance_for_heading_estimation

    @property
    def length(self) -> float:
        """
        Returns the length of the reference line [m].
        :return: the length of the reference line.
        """
        return float(self.reference_line.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """
        Gets a discretized representation of the reference line.
        :return: a list of StateSE2.
        """
        return cast(List[StateSE2], extract_discrete_polyline(self._reference_line))

    def get_nearest_station_from_position(self, point: Union[Point2D, Point, List[Point2D]]) -> Union[float, List[float]]:
        """
        Returns the station along the reference line where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: [m] station along the reference line.
        """
        if isinstance(point, Point2D):
            return self._reference_line.project(Point(point.x, point.y))
        elif isinstance(point, Point):
            return self._reference_line.project(point)
        elif isinstance(point, list) and isinstance(point[0], Point2D):
            return self._reference_line.project([Point(pt.x, pt.y) for pt in point])

    def get_lateral_from_position(self,
                                  point: Union[Point2D, List[Point2D]],
                                  station: Optional[Union[float, List[float]]]=None) -> Union[float, np.ndarray]:
        """
        Returns the lateral in Frenet frame of the give point.
        :param point: [m] x, y coordinates in global frame.
        :param station: [m] station coordinate in Frenet frame.
        :return: [m] lateral coordinate in Frenet frame.
        """
        if station is None:
            station = self.get_nearest_station_from_position(point)
        state1 = self._reference_line.interpolate(station)
        if isinstance(station, float):
            state2 = self._reference_line.interpolate(station + self._distance_for_heading_estimation)
            vector_ref = [state2.x - state1.x, state2.y - state1.y]
            vector_p = [point.x - state1.x, point.y - state1.y]
            if state1 == state2:
                # Handle the case where the queried position (state1) is at the end of the baseline path
                state2 = self._reference_line.interpolate(station - self._distance_for_heading_estimation)
                vector_ref = [state1.x - state2.x, state1.y - state2.y]
                vector_p = [point.x - state2.x, point.y - state2.y]
            outter_product_res = outter_product(vector_ref, vector_p)
            lateral = np.sign(outter_product_res) * np.sqrt((state1.x - point.x) ** 2 + (state1.y - point.y) ** 2)

        elif isinstance(station, list) or isinstance(station, np.ndarray):
            state2 = self._reference_line.interpolate([s + self._distance_for_heading_estimation for s in station])
            vector_ref = [[s2.x - s1.x, s2.y - s1.y] for s1, s2 in zip(state1, state2)]
            vector_p = [[p.x - s1.x, p.y - s1.y] for p, s1 in zip(point, state1)]
            if np.any(state1 == state2):
                # Handle the case where the queried position (state1) is at the end of the baseline path
                state2 = self._reference_line.interpolate([s - self._distance_for_heading_estimation for s in station])
                vector_ref = [[s1.x - s2.x, s1.y - s2.y] for s1, s2 in zip(state1, state2)]
                vector_p = [[p.x - s2.x, p.y - s2.y] for p, s2 in zip(point, state2)]
            outter_product_res = list_outter_product(vector_ref, vector_p)
            outter_product_res = np.array(outter_product_res)
            state1 = np.array([[s1.x, s1.y] for s1 in state1])
            point = np.array([[p.x, p.y] for p in point])
            lateral = np.sign(outter_product_res) * \
                      np.sqrt((state1[:, 0] - point[:, 0]) ** 2 + (state1[:, 1] - point[:, 1]) ** 2)

        return lateral

    def get_nearest_pose_from_position(self,
                                       point: Union[Point2D, List[Point2D]],
                                       station: Optional[Union[float, List[float]]]=None) -> Union[StateSE2, List[StateSE2]]:
        """
        Returns the pose along the reference line where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :param station: [m] station coordinate in Frenet frame.
        :return: nearest pose along the reference line as StateSE2.
        """
        if station is None:
            station = self.get_nearest_station_from_position(point)
        state1 = self._reference_line.interpolate(station)
        if isinstance(station, float):
            state2 = self._reference_line.interpolate(station + self._distance_for_heading_estimation)
            if state1 == state2:
                # Handle the case where the queried position (state1) is at the end of the baseline path
                state2 = self._reference_line.interpolate(station - self._distance_for_heading_estimation)
                heading = _get_heading(state2, state1)
            else:
                heading = _get_heading(state1, state2)
            return StateSE2(state1.x, state1.y, heading)

        elif isinstance(station, list) or isinstance(station, np.ndarray):
            state2 = self._reference_line.interpolate([s + self._distance_for_heading_estimation for s in station])
            if np.any(state1 == state2):
                # Handle the case where the queried position (state1) is at the end of the baseline path
                state2 = self._reference_line.interpolate([s - self._distance_for_heading_estimation for s in station])
                heading = _get_heading(list(state2), list(state1))
            else:
                heading = _get_heading(list(state1), list(state2))
            return [StateSE2(s.x, s.y, h) for s, h in zip(state1, heading)]

    def get_curvature_at_station(self, station: float) -> float:
        """
        Return curvature at a station along the reference line.
        :param station: [m] arc length along the reference line. It has to be 0<= station <=length.
        :return: [1/m] curvature along a reference line.
        """
        curvature = estimate_curvature_along_path(self._reference_line, station, self._distance_for_curvature_estimation)

        return float(curvature)

    def frenet_to_cartesian(self,
                            pose: np.ndarray,
                            t: np.ndarray=None,
                            v_s_t: np.ndarray=None,
                            v_l_t: np.ndarray=None,
                            a_s_t: np.ndarray=None,
                            a_l_t: np.ndarray=None,
                            pt1: np.ndarray=None,
                            pt2: np.ndarray=None,
                            xy1: np.ndarray=None,
                            xy2: np.ndarray=None,
                            compute_heading: bool=True) -> Dict[str, np.ndarray]:
        """
        :param pose: (num_trajs, 3, num_poses)
        Returns the dynamic states in global frame (Cartesian) given the pose/poses (station, lateral, heading), velocity, and acceleration in Frenet frame.
        """
        if len(pose.shape) == 3:
            station = pose[:, 0, :]
            lateral = pose[:, 1, :]
            heading = pose[:, 2, :]
        elif len(pose.shape) == 2:
            station = pose[0]
            lateral = pose[1]
            heading = pose[2]
        elif len(pose.shape) == 1:
            station = pose[0]
            lateral = pose[1]
            heading = pose[2]
        if pt1 is None:
            point_on_ref_line_cartesian = self._reference_line.interpolate(station)
        else:
            point_on_ref_line_cartesian = pt1
        if pt2 is None:
            point2_on_ref_line_cartesian = self._reference_line.interpolate(station + self._distance_for_heading_estimation)
        else:
            point2_on_ref_line_cartesian = pt2
        num_trajs, num_poses = point_on_ref_line_cartesian.shape
        if xy1 is None:
            xy_on_ref_line_cartesian = [point.xy for point in point_on_ref_line_cartesian.reshape((-1))]
            xy_on_ref_line_cartesian = np.squeeze(np.array(xy_on_ref_line_cartesian), axis=-1).reshape((num_trajs, num_poses, 2)).transpose(0, 2, 1)
        else:
            xy_on_ref_line_cartesian = xy1
        if xy2 is None:
            xy2_on_ref_line_cartesian = [point.xy for point in point2_on_ref_line_cartesian.reshape((-1))]
            xy2_on_ref_line_cartesian = np.squeeze(np.array(xy2_on_ref_line_cartesian), axis=-1).reshape((num_trajs, num_poses, 2)).transpose(0, 2, 1)
        else:
            xy2_on_ref_line_cartesian = xy2
        heading_on_ref_line_cartesian = _get_heading(xy_on_ref_line_cartesian, xy2_on_ref_line_cartesian)
        x_cartesian = xy_on_ref_line_cartesian[:, 0, :] - lateral * np.sin(heading_on_ref_line_cartesian)
        y_cartesian = xy_on_ref_line_cartesian[:, 1, :] + lateral * np.cos(heading_on_ref_line_cartesian)

        if compute_heading:
            # handle zero heading_on_ref_line_cartesian
            pt1_to_pt2 = []
            for pt1, pt2 in zip(point_on_ref_line_cartesian.reshape((-1,)), point2_on_ref_line_cartesian.reshape((-1,))):
                pt1_to_pt2.append(pt1.distance(pt2))
            pt1_to_pt2 = np.array(pt1_to_pt2)
            pt1_to_pt2 = pt1_to_pt2.reshape(point_on_ref_line_cartesian.shape)
            end_of_ref_line_index = np.where(pt1_to_pt2 == 0.)
            end_of_ref_line_traj_index = np.unique(end_of_ref_line_index[0])
            for traj_index in end_of_ref_line_traj_index:
                mask = end_of_ref_line_index[0] == traj_index
                min_pose_index = min(end_of_ref_line_index[1][mask])
                heading_on_ref_line_cartesian[traj_index, min_pose_index:] = heading_on_ref_line_cartesian[traj_index, min_pose_index - 1]

            # handle heading jump
            if np.amax(heading_on_ref_line_cartesian) - np.amin(heading_on_ref_line_cartesian) > math.pi:
                # [-pi, pi] -> [0, 2*pi]
                heading_on_ref_line_cartesian = wrap_angle(heading_on_ref_line_cartesian, min_val=0., max_val=2*math.pi)

            heading_cartesian = heading_on_ref_line_cartesian + heading
        else:
            heading_cartesian = np.zeros_like(heading)

        if t is not None:
            # note that the velocity and acceleration are in the local frame (vehicle coordinate system).
            rear_axle_velocity_x = v_s_t * np.cos(heading) + v_l_t * np.sin(heading)
            rear_axle_velocity_y = - v_s_t * np.sin(heading) + v_l_t * np.cos(heading)
            rear_axle_acceleration_x = a_s_t * np.cos(heading) + a_l_t * np.sin(heading)
            rear_axle_acceleration_y = - a_s_t * np.sin(heading) + a_l_t * np.cos(heading)
            angular_velocity = approximate_derivatives(heading_cartesian, t, window_length=3)
        else:
            rear_axle_velocity_x = None
            rear_axle_velocity_y = None
            rear_axle_acceleration_x = None
            rear_axle_acceleration_y = None
            angular_velocity = None

        # [0, 2*pi] -> [-pi, pi]
        heading_cartesian = wrap_angle(heading_cartesian)

        return {
            'pose_cartesian': np.stack((x_cartesian, y_cartesian, heading_cartesian)).transpose(1, 0, 2),
            'rear_axle_velocity_x': rear_axle_velocity_x,
            'rear_axle_velocity_y': rear_axle_velocity_y,
            'rear_axle_acceleration_x': rear_axle_acceleration_x,
            'rear_axle_acceleration_y': rear_axle_acceleration_y,
            'angular_velocity': angular_velocity,
        }
