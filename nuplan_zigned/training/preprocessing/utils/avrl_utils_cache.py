from __future__ import annotations

import logging
import pathlib
import textwrap
from typing import Optional, Tuple, Union, Dict, List, Any
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCache

from nuplan_zigned.training.preprocessing.features.avrl_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.avrl_generic_agents import GenericAgents

logger = logging.getLogger(__name__)


def compute_or_load_feature(
    scenario: AbstractScenario,
    cache_path: Optional[pathlib.Path],
    builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder],
    storing_mechanism: FeatureCache,
    force_feature_computation: bool,
    data_across_builders: Optional[Any]=None,
) -> Tuple[AbstractModelFeature, Optional[CacheMetadataEntry], Optional[Any]]:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param cache_path: location of cached features
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :param force_feature_computation: if true, even if cache exists, it will be overwritten
    :return features computed with builder and the metadata entry for the computed feature if feature is valid.
    """
    cache_path_available = cache_path is not None

    # Filename of the cached features/targets
    file_name = (
        cache_path / scenario.log_name / scenario.scenario_type / scenario.token / builder.get_feature_unique_name()
        if cache_path_available
        else None
    )

    # If feature recomputation is desired or cached file doesnt exists, compute the feature
    need_to_compute_feature = (
        force_feature_computation or not cache_path_available or not storing_mechanism.exists_feature_cache(file_name)
    )
    feature_stored_sucessfully = False
    if need_to_compute_feature:
        logger.debug("Computing feature...")
        if isinstance(scenario, CachedScenario):
            raise ValueError(
                textwrap.dedent(
                    f"""
                Attempting to recompute scenario with CachedScenario.
                This should typically never happen, and usually means that the scenario is missing from the cache.
                Check the cache to ensure that the scenario is present.

                If it was intended to re-compute the feature on the fly, re-run with `cache.use_cache_without_dataset=False`.

                Debug information:
                Scenario type: {scenario.scenario_type}. Scenario log name: {scenario.log_name}. Scenario token: {scenario.token}.
                """
                )
            )
        if isinstance(builder, AbstractFeatureBuilder):
            # transfer data to agent feature builder
            if builder.get_feature_unique_name() == 'generic_agents':
                scenario.data_across_builders = data_across_builders

            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
        else:
            raise ValueError(f"Unknown builder type: {type(builder)}")

        # If caching is enabled, store the feature
        if feature.is_valid and cache_path_available:
            logger.debug(f"Saving feature: {file_name} to a file...")
            file_name.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(feature, VectorSetMap):
                # data to be transferred to agent feature builder
                data_across_builders = {
                    'trajectory_samples': feature.trajectory_samples,
                }
                # compress feature
                feature = feature.compress()
            if isinstance(feature, GenericAgents):
                # compress feature
                feature = feature.compress()
            feature_stored_sucessfully = storing_mechanism.store_computed_feature_to_folder(file_name, feature)

    else:
        # In case the feature exists in the cache, load it
        logger.debug(f"Loading feature: {file_name} from a file...")
        feature = storing_mechanism.load_computed_feature_from_folder(file_name, builder.get_feature_type())

        if isinstance(feature, VectorSetMap):
            # decompress feature
            feature = feature.decompress()
        if isinstance(feature, GenericAgents):
            # decompress feature
            feature = feature.decompress()

        assert feature.is_valid, 'Invalid feature loaded from cache!'

    return (
        feature,
        CacheMetadataEntry(file_name=file_name) if (need_to_compute_feature and feature_stored_sucessfully) else None,
        data_across_builders,
    )


def check_map_consistency(feature1: VectorSetMap, feature2: VectorSetMap) -> Dict[str, Dict[str, List[bool]]]:
    """
    Check consistency of two features.
    :param feature1: original feature.
    :param feature2: feature after compressing and decompressing.
    :return: dict specifying whether each property of the two features are consistent.
    """
    consistency: Dict[str, Dict[str, List[bool]]] = {}

    consistency['coords'] = {}
    for feature_name in feature1.coords.keys():
        consistency['coords'][feature_name] = []
        for sample_idx in range(len(feature1.coords[feature_name])):
            consistency['coords'][feature_name].append(
                np.all(feature1.coords[feature_name][sample_idx] - feature2.coords[feature_name][sample_idx] < 1e-8)
            )

    consistency['traffic_light_data'] = {}
    for feature_name in feature1.traffic_light_data.keys():
        consistency['traffic_light_data'][feature_name] = []
        for sample_idx in range(len(feature1.traffic_light_data[feature_name])):
            consistency['traffic_light_data'][feature_name].append(
                np.all(feature1.traffic_light_data[feature_name][sample_idx] - feature2.traffic_light_data[feature_name][sample_idx] < 1e-8)
            )

    consistency['availabilities'] = {}
    for feature_name in feature1.availabilities.keys():
        consistency['availabilities'][feature_name] = []
        for sample_idx in range(len(feature1.availabilities[feature_name])):
            consistency['availabilities'][feature_name].append(
                np.all(feature1.availabilities[feature_name][sample_idx] - feature2.availabilities[feature_name][sample_idx] < 1e-8)
            )

    return consistency


def check_agents_consistency(feature1: GenericAgents, feature2: GenericAgents) -> Dict[str, Dict[str, List[bool]]]:
    """
    Check consistency of two features.
    :param feature1: original feature.
    :param feature2: feature after compressing and decompressing.
    :return: dict specifying whether each property of the two features are consistent.
    """
    consistency: Dict[str, Dict[int, Dict[int, List[bool]]]] = {}

    consistency['ego'] = {}
    for sample_idx in range(len(feature1.ego)):
        consistency['ego'][sample_idx] = {}
        for i_traj in range(len(feature1.ego[sample_idx])):
            consistency['ego'][sample_idx][i_traj] = []
            for i_timestep in range(len(feature1.ego[sample_idx][i_traj])):
                consistency['ego'][sample_idx][i_traj].append(
                    np.all(feature1.ego[sample_idx][i_traj][i_timestep] - feature2.ego[sample_idx][i_traj][i_timestep] < 1e-8)
                )
            consistency['ego'][sample_idx][i_traj] = all(consistency['ego'][sample_idx][i_traj])
        consistency['ego'][sample_idx] = all(list(consistency['ego'][sample_idx].values()))

    consistency['agents'] = {}
    for feature_name in feature1.agents.keys():
        if '_ids' not in feature_name:
            for sample_idx in range(len(feature1.agents[feature_name])):
                consistency['agents'][sample_idx] = {}
                for i_traj in range(len(feature1.agents[feature_name][sample_idx])):
                    consistency['agents'][sample_idx][i_traj] = []
                    for i_timestep in range(len(feature1.agents[feature_name][sample_idx][i_traj])):
                        consistency['agents'][sample_idx][i_traj].append(
                            np.all(feature1.agents[feature_name][sample_idx][i_traj][i_timestep] - feature2.agents[feature_name][sample_idx][i_traj][i_timestep] < 1e-8)
                        )
                    consistency['agents'][sample_idx][i_traj] = all(consistency['agents'][sample_idx][i_traj])
                consistency['agents'][sample_idx] = all(list(consistency['agents'][sample_idx].values()))

    return consistency