import numpy as np
from typing import Dict

from nuplan_zigned.utils.utils import polynomial, polynomial_derivative


class TrajectorySampler:
    def __init__(self, frenet_frame, ego_state, lane_width, delta_l, time_horizon, num_poses):
        self.frenet_frame = frenet_frame
        self.ego_state = ego_state
        self.a_min = -3.
        self.a_max = 2.
        self.lane_width = lane_width
        self.delta_l = delta_l
        self.time_horizon = time_horizon
        self.num_poses = num_poses
        self.t = np.linspace(0., self.time_horizon, self.num_poses + 1)
        self.v_min = max(0., ego_state.dynamic_car_state.speed + self.a_min * self.time_horizon)
        self.v_max = min(90 / 3.6, ego_state.dynamic_car_state.speed + self.a_max * self.time_horizon)


        self.sparse_idx = np.array([0, 2, 3, 5, 6, 8,
                                    10, 13, 16,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26,
                                    28, 31, 34,
                                    36, 38, 39, 41, 42, 44])

    def get_trajectory_samples(self, sparse: bool=True, caching_avrl_features: bool=False) -> Dict[str, np.ndarray]:
        """
        Returns rear-axle sparse pose, velocity, and acceleration samples in Frenet frame.
        :return: shape (num_trajs, ..., self.num_poses + 1), rear-axle sparse pose, velocity, and acceleration samples in Frenet frame.
        """
        v_target = np.array([self.v_min,
                             self.v_min + (self.ego_state.dynamic_car_state.speed - self.v_min) / 2,
                             self.ego_state.dynamic_car_state.speed,
                             self.v_max - (self.v_max - self.ego_state.dynamic_car_state.speed) / 2,
                             self.v_max])
        lateral_left = np.linspace(self.delta_l * 5, self.delta_l * 3, 3)
        lateral_mid = np.linspace(self.delta_l, -self.delta_l, 3)
        lateral_right = np.linspace(-self.delta_l * 3, -self.delta_l * 5, 3)
        lateral_target = np.hstack((lateral_left, lateral_mid, lateral_right))
        current_ego_station = self.frenet_frame.get_nearest_station_from_position(self.ego_state.rear_axle.point)
        reference_line_pose = self.frenet_frame.get_nearest_pose_from_position(self.ego_state.rear_axle.point)
        current_ego_lateral = self.frenet_frame.get_lateral_from_position(self.ego_state.rear_axle.point)
        delta_theta = self.ego_state.rear_axle.heading - reference_line_pose.heading
        cos_delta_theta = np.cos(delta_theta)
        sin_delta_theta = np.sin(delta_theta)
        ego_velocity_local = self.ego_state.dynamic_car_state.rear_axle_velocity_2d
        ego_acceleration_local = self.ego_state.dynamic_car_state.rear_axle_acceleration_2d
        if ego_velocity_local.x * cos_delta_theta + ego_velocity_local.y * sin_delta_theta < 0.5 \
            and ego_acceleration_local.x * cos_delta_theta + ego_acceleration_local.y * sin_delta_theta < 0.:
            coefficient = 0.
        else:
            coefficient = 1.
        current_speed_is_small = self.ego_state.dynamic_car_state.speed < 0.5
        target_speed_is_small = v_target < 0.1
        # force lane keeping when speed is small to avoid generating wrong trajectory
        if current_speed_is_small and not caching_avrl_features:
            lateral_target = np.ones_like(lateral_target) * current_ego_lateral

        # 4th-order Polynomial s(t)
        A_s = np.matrix([[1., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0.],
                         [0., 0., 2., 0., 0.],
                         [0., 1., 2. * self.time_horizon, 3 * self.time_horizon ** 2, 4 * self.time_horizon ** 3],
                         [0., 0., 2., 6 * self.time_horizon, 12 * self.time_horizon ** 2]])
        Q_s = np.array([current_ego_station * np.ones_like(v_target),
                        (ego_velocity_local.x * cos_delta_theta + ego_velocity_local.y * sin_delta_theta) * np.ones_like(v_target),
                        coefficient * (ego_acceleration_local.x * cos_delta_theta + ego_acceleration_local.y * sin_delta_theta) * np.ones_like(v_target),
                        v_target,
                        np.zeros_like(v_target)
                        ])
        P_s = A_s.I * Q_s
        # 5th-order Polynomial l(t)
        A_l = np.matrix([[1., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0.],
                         [0., 0., 2., 0., 0., 0.],
                         [1., self.time_horizon, self.time_horizon ** 2, self.time_horizon ** 3, self.time_horizon ** 4, self.time_horizon ** 5],
                         [0., 1., 2 * self.time_horizon, 3 * self.time_horizon ** 2, 4 * self.time_horizon ** 3, 5 * self.time_horizon ** 4],
                         [0., 0., 2., 6 * self.time_horizon, 12 * self.time_horizon ** 2, 20 * self.time_horizon ** 3]])
        Q_l = np.array([current_ego_lateral * np.ones_like(lateral_target),
                        (ego_velocity_local.x * sin_delta_theta + ego_velocity_local.y * cos_delta_theta) * np.ones_like(lateral_target),
                        (ego_acceleration_local.x * sin_delta_theta + ego_acceleration_local.y * cos_delta_theta) * np.ones_like(lateral_target),
                        lateral_target,
                        np.zeros_like(lateral_target),
                        np.zeros_like(lateral_target)
                        ])
        P_l = A_l.I * Q_l

        s_t = polynomial(self.t, P_s.A)
        l_t = polynomial(self.t, P_l.A)
        vs_t = polynomial_derivative(self.t, P_s.A, order=1)
        vl_t = polynomial_derivative(self.t, P_l.A, order=1)
        as_t = polynomial_derivative(self.t, P_s.A, order=2)
        al_t = polynomial_derivative(self.t, P_l.A, order=2)

        trajs = np.zeros((s_t.shape[0] * l_t.shape[0], 7, self.num_poses + 1))
        for i in range(s_t.shape[0]):
            for j in range(l_t.shape[0]):
                if current_speed_is_small and not caching_avrl_features:
                    if target_speed_is_small[i]:
                        # when both speed and target speed are small, let ego vehicle be stationary
                        trajs[i * l_t.shape[0] + j, 0, :] = np.ones((self.num_poses + 1,)) * s_t[0]
                        trajs[i * l_t.shape[0] + j, 1, :] = np.ones((self.num_poses + 1,)) * l_t[0]
                        trajs[i * l_t.shape[0] + j, 2, :] = np.ones((self.num_poses + 1,)) * delta_theta
                        trajs[i * l_t.shape[0] + j, 3, :] = np.zeros((self.num_poses + 1,))
                        trajs[i * l_t.shape[0] + j, 4, :] = np.zeros((self.num_poses + 1,))
                        trajs[i * l_t.shape[0] + j, 5, :] = np.zeros((self.num_poses + 1,))
                        trajs[i * l_t.shape[0] + j, 6, :] = np.zeros((self.num_poses + 1,))
                    else:
                        # when speed is small whereas target speed is not small, fix heading calculation problem
                        trajs[i * l_t.shape[0] + j, 0, :] = s_t[i]
                        trajs[i * l_t.shape[0] + j, 1, :] = l_t[j]
                        trajs[i * l_t.shape[0] + j, 2, :] = np.ones((self.num_poses + 1,)) * delta_theta
                        trajs[i * l_t.shape[0] + j, 3, :] = vs_t[i]
                        trajs[i * l_t.shape[0] + j, 4, :] = vl_t[j]
                        trajs[i * l_t.shape[0] + j, 5, :] = as_t[i]
                        trajs[i * l_t.shape[0] + j, 6, :] = al_t[j]
                else:
                    theta_t = np.arctan2(
                        polynomial_derivative(self.t, P_l[:, j].A, order=1),
                        polynomial_derivative(self.t, P_s[:, i].A, order=1)
                    )
                    if target_speed_is_small[i]:
                        # fix heading calculation problem
                        theta_t[:, -1] = theta_t[:, -2] + (theta_t[:, -2] - theta_t[:, -3])
                    trajs[i * l_t.shape[0] + j, 0, :] = s_t[i]
                    trajs[i * l_t.shape[0] + j, 1, :] = l_t[j]
                    trajs[i * l_t.shape[0] + j, 2, :] = theta_t
                    trajs[i * l_t.shape[0] + j, 3, :] = vs_t[i]
                    trajs[i * l_t.shape[0] + j, 4, :] = vl_t[j]
                    trajs[i * l_t.shape[0] + j, 5, :] = as_t[i]
                    trajs[i * l_t.shape[0] + j, 6, :] = al_t[j]

        if sparse:
            trajs = trajs[(self.sparse_idx,)]

        return {
            't': self.t,
            'poses_frenet': trajs[:, 0:3],
            'vs_frenet': trajs[:, 3],
            'vl_frenet': trajs[:, 4],
            'as_frenet': trajs[:, 5],
            'al_frenet': trajs[:, 6],
        }