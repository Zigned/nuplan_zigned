from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Union, Any, Optional
import logging

import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence

from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.common.actor_state.ego_state import EgoState

from nuplan_zigned.training.modeling.modules.qcmae_encoder import QCMAEEncoder
from nuplan_zigned.training.modeling.modules.qcmae_decoder import QCMAEDecoder
from nuplan_zigned.utils.weight_init import weight_init
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents
from nuplan_zigned.utils.utils import split_list

logger = logging.getLogger(__name__)


class Actor(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_features: List[str],
                 agent_features: List[str],
                 output_dim: int,
                 output_head: bool,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_decoder_layers: int,
                 radius: float,
                 max_agents: Dict[str, int],
                 pretrained_model_dir: str,
                 map_location: str,
                 finetune_range: str,
                 ) -> None:
        super(Actor, self).__init__()
        self.qcmae_encoder = QCMAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_features=map_features,
            agent_features=agent_features,
            pretrain=False,
            pretrain_map_encoder=False,
            pretrain_agent_encoder=False,
            prob_pretrain_mask=[0., 0.],
            prob_pretrain_mask_mask=0.,
            prob_pretrain_mask_random=0.,
            prob_pretrain_mask_unchanged=0.,
        )
        self.qcmae_decoder = QCMAEDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            finetune_range=finetune_range,
        )
        self.radius = radius
        self.max_agents = max_agents
        self._agent_types = agent_features
        self._polygon_types = map_features

        self.apply(weight_init)

        self.qcmae_encoder.to(device=torch.device(map_location))
        self.qcmae_decoder.to(device=torch.device(map_location))

        if pretrained_model_dir is not None:
            pretrained_model_dict = torch.load(pretrained_model_dir, map_location=map_location)
            encoder_dict = self.qcmae_encoder.state_dict()
            model_dict = {}
            for k, v in pretrained_model_dict['state_dict'].items():
                if k.replace('model.encoder.', '', 1) in list(encoder_dict.keys()):
                    model_dict[k.replace('model.encoder.', '', 1)] = v
            assert len(model_dict) > 0
            encoder_dict.update(model_dict)
            self.qcmae_encoder.load_state_dict(encoder_dict)

            decoder_dict = self.qcmae_decoder.state_dict()
            model_dict = {}
            for k, v in pretrained_model_dict['state_dict'].items():
                if k.replace('model.decoder.', '', 1) in list(decoder_dict.keys()):
                    model_dict[k.replace('model.decoder.', '', 1)] = v
            assert len(model_dict) > 0
            decoder_dict.update(model_dict)
            self.qcmae_decoder.load_state_dict(decoder_dict)

            logger.info(f'\npretrained QCMAE loaded: {pretrained_model_dir}\n')

    def forward(self, features: FeaturesType) -> Dict[str, Any]:

        scene_enc = self.qcmae_encoder(features)
        pred = self.qcmae_decoder(features, scene_enc)

        return pred

    def process_features(self,
                         features: FeaturesType,
                         anchor_ego_state: List[EgoState]) -> FeaturesType:
        """extract features within radius of interest"""
        batch_size = features['vector_set_map'].batch_size
        # process map data
        map_data = copy.deepcopy(features['vector_set_map'].map_data)
        map_data['map_point']['position'] = torch.cat(map_data['map_point']['position'], dim=0)
        position = map_data['map_point']['position']
        anchor_position = [state.center.array for state in anchor_ego_state]
        anchor_position = position.new_tensor(anchor_position)
        anchor_position = [pos.expand(size, 2) for pos, size in zip(anchor_position, map_data['map_point']['num_nodes'])]
        anchor_position = torch.cat(anchor_position, dim=0)
        pt_within_radius = torch.norm(position - anchor_position, p=2, dim=1) < self.radius

        # concatenate edge index
        pt2pl_edge_index = [edge_index.long() for edge_index in map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
        pl2pl_edge_index = [edge_index.long() for edge_index in map_data[('map_polygon', 'to', 'map_polygon')]['edge_index']]
        for sample_idx in range(1, batch_size):
            pt2pl_edge_index[sample_idx][0] += \
                sum(map_data['map_point']['num_nodes'][0:sample_idx])
            pt2pl_edge_index[sample_idx][1] += \
                sum(map_data['map_polygon']['num_nodes'][0:sample_idx])
            pl2pl_edge_index[sample_idx] += \
                sum(map_data['map_polygon']['num_nodes'][0:sample_idx])
        pt2pl_edge_index = torch.cat(pt2pl_edge_index, dim=1)
        pl2pl_edge_index = torch.cat(pl2pl_edge_index, dim=1)

        unique_polygons = torch.unique(pt2pl_edge_index[1, :], return_counts=True)
        pl_within_radius = torch.split(pt_within_radius, unique_polygons[1].cpu().tolist())
        # pl_within_radius = torch.stack([torch.any(whether) for whether in pl_within_radius])
        padded_pl_within_radius = pad_sequence(pl_within_radius, batch_first=True, padding_value=0)
        pl_within_radius = padded_pl_within_radius.any(dim=1)

        pl2pl_type = torch.cat(map_data[('map_polygon', 'to', 'map_polygon')]['type']).long()
        bool1 = torch.any(pl2pl_edge_index[0:1] - unique_polygons[0][pl_within_radius].unsqueeze(-1) == 0., dim=0)
        bool2 = torch.any(pl2pl_edge_index[1:] - unique_polygons[0][pl_within_radius].unsqueeze(-1) == 0., dim=0)
        pl2pl_edge_index_tmp1 = pl2pl_edge_index[:, bool1 & bool2]
        pl2pl_edge_index_tmp2 = pl2pl_edge_index[:, bool2]
        if pl2pl_edge_index_tmp1.shape[1] > 0:
            pl2pl_edge_index = pl2pl_edge_index_tmp1
            pl2pl_type = pl2pl_type[bool1 & bool2]
        else:
            pl2pl_edge_index = pl2pl_edge_index_tmp2
            pl2pl_type = pl2pl_type[bool2]
            pl_within_radius[pl2pl_edge_index_tmp2[0].long()] = True

        repeated_whether_within_radius = pl_within_radius.repeat_interleave(unique_polygons[1])
        map_data[('map_point', 'to', 'map_polygon')]['edge_index'] = \
            pt2pl_edge_index[:, repeated_whether_within_radius]

        map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'] = [pl2pl_edge_index]
        map_data[('map_polygon', 'to', 'map_polygon')]['type'] = [pl2pl_type]
        map_data[('map_point', 'to', 'map_polygon')]['edge_index'] = [map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
        map_data['map_polygon']['position'] = [torch.cat(map_data['map_polygon']['position'], dim=0)]
        map_data['map_polygon']['orientation'] = [torch.cat(map_data['map_polygon']['orientation'])]
        map_data['map_polygon']['type'] = [torch.cat(map_data['map_polygon']['type'])]
        map_data['map_polygon']['is_intersection'] = [torch.cat(map_data['map_polygon']['is_intersection'])]
        map_data['map_polygon']['tl_statuses'] = [torch.cat(map_data['map_polygon']['tl_statuses'])]
        map_data['map_point']['position'] = [map_data['map_point']['position']]
        map_data['map_point']['orientation'] = [torch.cat(map_data['map_point']['orientation'])]
        map_data['map_point']['magnitude'] = [torch.cat(map_data['map_point']['magnitude'])]
        map_data['map_point']['type'] = [torch.cat(map_data['map_point']['type'])]
        map_data['map_point']['side'] = [torch.cat(map_data['map_point']['side'])]
        map_data['map_point']['tl_statuses'] = [torch.cat(map_data['map_point']['tl_statuses'])]

        # process agent data
        agent_data = copy.deepcopy(features['generic_agents'].agent_data)
        position = torch.cat(agent_data['position'], dim=0)
        heading = torch.cat(agent_data['heading'], dim=0)
        velocity = torch.cat(agent_data['velocity'], dim=0)
        length = torch.cat(agent_data['length'], dim=0)
        width = torch.cat(agent_data['width'], dim=0)
        valid_mask = torch.cat(agent_data['valid_mask'], dim=0)
        predict_mask = torch.cat(agent_data['predict_mask'], dim=0)
        id = sum(agent_data['id'], [])
        type = torch.cat(agent_data['type'], dim=0)
        anchor_position = [state.center.array for state in anchor_ego_state]
        anchor_position = position.new_tensor(anchor_position)
        anchor_position = [pos.expand(size, 2) for pos, size in zip(anchor_position, agent_data['num_nodes'])]
        anchor_position = torch.cat(anchor_position, dim=0)
        distance_to_agents = torch.norm(position[:, -1, :] - anchor_position, p=2, dim=1)
        distance_to_agents[agent_data['av_index']] = 0.  # make sure including ego
        within_radius_mask = distance_to_agents < self.radius
        generic_object_mask = type == self._agent_types.index('GENERIC_OBJECT')
        argsort = torch.argsort(distance_to_agents, descending=False)
        sorted_generic_object_mask = generic_object_mask[argsort]
        sorted_generic_object_indices = argsort[sorted_generic_object_mask]

        num_nodes = []
        av_index = []
        for sample_idx in range(batch_size):
            if sample_idx == 0:
                start = 0
                end = agent_data['num_nodes'][0]
            else:
                start = end
                end = start + agent_data['num_nodes'][sample_idx]
            num_nodes.append(sum(within_radius_mask[start:end]).item())
            if sample_idx == 0:
                av_index.append(0)
            else:
                av_index.append(av_index[-1] + num_nodes[sample_idx - 1])

            batch_mask = (start <= sorted_generic_object_indices) & (sorted_generic_object_indices < end)
            ignore_index = sorted_generic_object_indices[batch_mask][self.max_agents['GENERIC_OBJECT']:]
            within_radius_mask.index_fill_(dim=0, index=ignore_index, value=False)

        agent_data['num_nodes'] = num_nodes
        agent_data['av_index'] = av_index
        agent_data['id'] = [[id for id, mask in zip(id, within_radius_mask) if mask]]
        agent_data['position'] = [position[within_radius_mask]]
        agent_data['heading'] = [heading[within_radius_mask]]
        agent_data['velocity'] = [velocity[within_radius_mask]]
        agent_data['type'] = [type[within_radius_mask]]
        agent_data['valid_mask'] = [valid_mask[within_radius_mask]]
        agent_data['predict_mask'] = [predict_mask[within_radius_mask]]
        agent_data['length'] = [length[within_radius_mask]]
        agent_data['width'] = [width[within_radius_mask]]

        return {
            'vector_set_map': VectorSetMap(map_data=map_data),
            'generic_agents': GenericAgents(agent_data=agent_data)
        }

    def unpack_features(self,
                        features: FeaturesType) -> FeaturesType:
        map_data = features['vector_set_map'].map_data
        agent_data = features['generic_agents'].agent_data
        device = map_data['map_polygon']['position'][0].device

        # map feature
        pl_num_nodes = map_data['map_polygon']['num_nodes']
        pl_position = map_data['map_polygon']['position']
        pl_orientation = map_data['map_polygon']['orientation']
        pl_type = map_data['map_polygon']['type']
        pl_is_intersection = map_data['map_polygon']['is_intersection']
        pl_tl_statuses = map_data['map_polygon']['tl_statuses']
        pt_num_nodes = map_data['map_point']['num_nodes']
        pt_position = map_data['map_point']['position']
        pt_orientation = map_data['map_point']['orientation']
        pt_magnitude = map_data['map_point']['magnitude']
        pt_type = map_data['map_point']['type']
        pt_side = map_data['map_point']['side']
        pt_tl_statuses = map_data['map_point']['tl_statuses']
        pt2pl_edge_index = map_data[('map_point', 'to', 'map_polygon')]['edge_index']
        pl2pl_edge_index = map_data[('map_polygon', 'to', 'map_polygon')]['edge_index']
        pl2pl_type = map_data[('map_polygon', 'to', 'map_polygon')]['type']

        map_data['map_polygon']['position'] = list(torch.split(pl_position[0], pl_num_nodes))
        map_data['map_polygon']['orientation'] = list(torch.split(pl_orientation[0], pl_num_nodes))
        map_data['map_polygon']['type'] = list(torch.split(pl_type[0], pl_num_nodes))
        map_data['map_polygon']['is_intersection'] = list(torch.split(pl_is_intersection[0], pl_num_nodes))
        map_data['map_polygon']['tl_statuses'] = list(torch.split(pl_tl_statuses[0], pl_num_nodes))
        map_data['map_point']['position'] = list(torch.split(pt_position[0], pt_num_nodes))
        map_data['map_point']['orientation'] = list(torch.split(pt_orientation[0], pt_num_nodes))
        map_data['map_point']['magnitude'] = list(torch.split(pt_magnitude[0], pt_num_nodes))
        map_data['map_point']['type'] = list(torch.split(pt_type[0], pt_num_nodes))
        map_data['map_point']['side'] = list(torch.split(pt_side[0], pt_num_nodes))
        map_data['map_point']['tl_statuses'] = list(torch.split(pt_tl_statuses[0], pt_num_nodes))

        pt2pl_edge_index_tmp = []
        pl2pl_edge_index_tmp = []
        pl2pl_type_tmp = []
        for sample_idx in range(len(pl_num_nodes)):
            if sample_idx == 0:
                start = 0
                end = pl_num_nodes[sample_idx]
            else:
                start = end
                end = start + pl_num_nodes[sample_idx]
            mask = (start <= pt2pl_edge_index[0][1]) & (pt2pl_edge_index[0][1] < end)
            pt2pl_edge_index_tmp.append(
                pt2pl_edge_index[0][:, mask] - torch.tensor([sum(pt_num_nodes[:sample_idx]),
                                                             sum(pl_num_nodes[:sample_idx])], device=device).unsqueeze(-1)
            )
            mask = (start <= pl2pl_edge_index[0][1]) & (pl2pl_edge_index[0][1] < end)
            pl2pl_edge_index_tmp.append(
                pl2pl_edge_index[0][:, mask] - torch.tensor([sum(pl_num_nodes[:sample_idx])], device=device)
            )
            pl2pl_type_tmp.append(pl2pl_type[0][mask])

        map_data[('map_point', 'to', 'map_polygon')]['edge_index'] = pt2pl_edge_index_tmp
        map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'] = pl2pl_edge_index_tmp
        map_data[('map_polygon', 'to', 'map_polygon')]['type'] = pl2pl_type_tmp

        # agent feature
        a_num_nodes = agent_data['num_nodes']
        av_index = agent_data['av_index']
        valid_mask = agent_data['valid_mask']
        predict_mask = agent_data['predict_mask']
        id = agent_data['id']
        type = agent_data['type']
        position = agent_data['position']
        heading = agent_data['heading']
        velocity = agent_data['velocity']
        length = agent_data['length']
        width = agent_data['width']

        agent_data['av_index'] = [0] * len(av_index)
        agent_data['valid_mask'] = list(torch.split(valid_mask[0], a_num_nodes))
        agent_data['predict_mask'] = list(torch.split(predict_mask[0], a_num_nodes))
        agent_data['id'] = split_list(id[0], a_num_nodes)
        agent_data['type'] = list(torch.split(type[0], a_num_nodes))
        agent_data['position'] = list(torch.split(position[0], a_num_nodes))
        agent_data['heading'] = list(torch.split(heading[0], a_num_nodes))
        agent_data['velocity'] = list(torch.split(velocity[0], a_num_nodes))
        agent_data['length'] = list(torch.split(length[0], a_num_nodes))
        agent_data['width'] = list(torch.split(width[0], a_num_nodes))

        return {
            'vector_set_map': VectorSetMap(map_data=map_data),
            'generic_agents': GenericAgents(agent_data=agent_data)
        }
