import os
import logging
import numpy as np
import torch
import gc
from pytorch_lightning.utilities import device_parser

logger = logging.getLogger(__name__)


class ReplayBuffer:
	def __init__(self, max_size=int(1e6), batch_size=4, load_dir=None, save_dir=None):
		self.max_size = max_size
		self.batch_size = batch_size
		self.load_dir = load_dir
		self.save_dir = os.path.join(save_dir, 'in_epoch_checkpoints')
		self.ptr = 0
		self.size = 0

		self.state = [None] * max_size
		self.action = [None] * max_size
		self.action_samples = [None] * max_size
		self.target = [None] * max_size
		self.next_state = [None] * max_size
		self.reward = [None] * max_size
		self.not_done = [None] * max_size  # if next_state is not_done
		self.next_target = [None] * max_size
		self.sampled_state = None
		self.sampled_unpacked_state = None
		self.sampled_action = None
		self.sampled_target = None
		self.sampled_action_samples = None
		if self.load_dir is not None:
			self.load(self.load_dir)

	def add(self, state, action, action_samples, target, reward, next_state, done, next_target, add_mask):
		batch_size = len(action)

		# do not add data to replay buffer when no valid action samples
		for sample_idx in range(batch_size):
			a_sample = action_samples[sample_idx]
			if a_sample is None:
				continue
			traj_left, traj_keep, traj_right = a_sample['traj_left_dedup'], a_sample['traj_keep_dedup'], a_sample['traj_right_dedup']
			left_is_valid = [s_t_0.shape[0] > 0 and l_t_0.shape[0] > 0 and s_t_1.shape[0] > 0 and l_t_1.shape[0] > 0
							 for s_t_0, l_t_0, s_t_1, l_t_1 in zip(traj_left['s_t_0'], traj_left['l_t_0'], traj_left['s_t_1'], traj_left['l_t_1'])]
			left_is_valid = any(left_is_valid)
			keep_is_valid = [s_t_0.shape[0] > 0 and l_t_0.shape[0] > 0 and s_t_1.shape[0] > 0
							 and l_t_1.shape[0] > 0 and s_t_2.shape[0] > 0 and l_t_2.shape[0] > 0
							 for s_t_0, l_t_0, s_t_1, l_t_1, s_t_2, l_t_2 in zip(traj_keep['s_t_0'],
																   traj_keep['l_t_0'],
																   traj_keep['s_t_1'],
																   traj_keep['l_t_1'],
																   traj_keep['s_t_2'],
																   traj_keep['l_t_2'])]
			keep_is_valid = any(keep_is_valid)
			right_is_valid = [s_t_0.shape[0] > 0 and l_t_0.shape[0] > 0 and s_t_1.shape[0] > 0 and l_t_1.shape[0] > 0
							 for s_t_0, l_t_0, s_t_1, l_t_1 in zip(traj_right['s_t_0'], traj_right['l_t_0'], traj_right['s_t_1'], traj_right['l_t_1'])]
			right_is_valid = any(right_is_valid)
			if any([left_is_valid, keep_is_valid, right_is_valid]):
				add_mask[sample_idx] = add_mask[sample_idx]
			else:
				add_mask[sample_idx] = False
		# add_mask & valid imagine mask (r is None if the imagination is not valid, see env.step())
		add_mask = [m and r is not None for m, r in zip(add_mask, reward)]

		valid_batch_size = sum(add_mask)
		self.state[self.ptr:self.ptr + valid_batch_size] = [
			{feature_name: feature[sample_idx] for feature_name, feature in state.items()}
			for sample_idx in range(batch_size) if add_mask[sample_idx]
		]
		self.action[self.ptr:self.ptr + valid_batch_size] = [a for a, mask in zip(action, add_mask) if mask]
		self.action_samples[self.ptr:self.ptr + valid_batch_size] = [sample for sample, mask in zip(action_samples, add_mask) if mask]
		if target is not None:
			self.target[self.ptr:self.ptr + valid_batch_size] = [
				{feature_name: feature[sample_idx] for feature_name, feature in target.items()}
				for sample_idx in range(batch_size) if add_mask[sample_idx]
			]
		self.reward[self.ptr:self.ptr + valid_batch_size] = [r for r, mask in zip(reward, add_mask) if mask]
		if next_state['imagined_state'] is not None:
			imagined_state = [
				{
					name: []
					for name in state.keys()
				}
				for _ in range(len(add_mask))
			]
			for img_s in next_state['imagined_state']:
				for i in range(batch_size):
					for name, feature in img_s.items():
						imagined_state[i][name].append(feature[i])
			next_state['imagined_state'] = imagined_state
		self.next_state[self.ptr:self.ptr + valid_batch_size] = [
			{feature_name: feature[sample_idx] for feature_name, feature in next_state.items()}
			for sample_idx in range(batch_size) if add_mask[sample_idx]
		]
		self.not_done[self.ptr:self.ptr + valid_batch_size] = [1. - d for d, mask in zip(done, add_mask) if mask]
		if next_target is not None:
			self.next_target[self.ptr:self.ptr + valid_batch_size] = [
				{target_name: target[sample_idx] for target_name, target in next_target.items()}
				for sample_idx in range(batch_size) if add_mask[sample_idx]
			]

		self.ptr = (self.ptr + valid_batch_size) % self.max_size
		self.size = min(self.size + valid_batch_size, self.max_size)

	def sample(self, device):
		indices = np.random.randint(0, self.size, size=self.batch_size)

		return (
			# state
			[
				{key: value.to_device(device) for key, value in self.state[index].items()}
				for index in indices
			],
			# action
			[torch.FloatTensor(self.action[idx]).to(device) for idx in indices],
			# action_samples
			[self.action_samples[idx] for idx in indices],
			# target
			[
				{key: value.to_device(device) for key, value in self.target[index].items()}
				for index in indices if self.target[index] is not None
			],
			# reward
			[torch.FloatTensor(self.reward[idx]).to(device) for idx in indices],
			# next state
			[
				{
					key: value.to_device(device)
					if not isinstance(value, dict)
					else {k: [vv.to_device(device) for vv in v] for k, v in value.items()}
					for key, value in self.next_state[index].items()
				}
				for index in indices
			],
			# not done
			[torch.FloatTensor([self.not_done[idx]]).to(device) for idx in indices],
			# next target
			[
				{key: value.to_device(device) for key, value in self.next_target[index].items()}
				for index in indices if self.next_target[index] is not None
			]
		)

	def save(self, save_dir: str=None) -> None:
		data = {
			'state': self.state,
			'action': self.action,
			'action_samples': self.action_samples,
			'target': self.target,
			'next_state': self.next_state,
			'reward': self.reward,
			'not_done': self.not_done,
			'next_target': self.next_target,
			'buffer_size': self.size,
		}
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		if save_dir is None:
			save_dir = os.path.join(self.save_dir, 'replay_buffer.pth')
		torch.save(data, save_dir)
		logger.info(f'replay buffer saved: {save_dir}')

	def load(self, file_dir):
		del self.state, self.action, self.action_samples, self.target, self.next_state, self.reward, self.not_done, self.next_target
		gc.collect()
		data = torch.load(file_dir)
		self.state = data['state']
		self.action = data['action']
		self.action_samples = data['action_samples']
		self.target = data['target']
		self.next_state = data['next_state']
		self.reward = data['reward']
		self.not_done = data['not_done']
		self.next_target = data['next_target']
		self.size = data['buffer_size']
		self.ptr = self.size

		logger.info(f'replay buffer loaded:{file_dir}')
