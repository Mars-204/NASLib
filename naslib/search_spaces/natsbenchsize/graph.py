import numpy as np
import random
import itertools
import torch
import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
)

from naslib.utils.utils import get_project_root

from naslib.search_spaces.nasbench201.primitives import ResNetBasicblock


OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric


class NATSBenchSizeSearchSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.channel_candidates = [8*i for i in range(1, 9)]
        self.channels = [8, 8, 8, 8, 8]

        self.space_name = "natsbenchsizesearchspace"
        # Graph not implemented

        self.num_classes = self.NUM_CLASSES if hasattr(self, "NUM_CLASSES") else 10
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = "nasbench201"
        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing

        total_num_nodes = 20
        self.add_nodes_from(range(1, total_num_nodes + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

        channels = [16, 32, 64]

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set("op", ops.Stem(channels[0]))

        # stage 1
        for i in range(2, 7):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_1"))

        # stage 2
        self.edges[7, 8].set(
            "op", ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2)
        )
        for i in range(8, 13):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_2"))

        # stage 3
        self.edges[13, 14].set(
            "op", ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2)
        )
        for i in range(14, 19):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_3"))

        # post-processing
        self.edges[19, 20].set(
            "op",
            ops.Sequential(
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1], self.num_classes),
            ),
        )

        # set the ops at the cells (channel dependent)
        for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
            self.update_edges(
                update_func=lambda edge: _set_cell_ops(edge, C=c),
                scope=scope,
                private_edge_data=True,
            )

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
        hp=90,
        is_random=False
    ):
        """
        Query results from natsbench

        Args:
            metric      : Metric to query for
            dataset     : Dataset to query for
            epoch       : If specified, returns the metric of the arch at that epoch of training
            full_lc     : If true, returns the curve of the given metric from the first to the last epoch
            dataset_api : API to use for querying metrics
            hp          : Number of epochs the model was trained for. Value is in {1, 12, 90}
            is_random   : When True, the performance of a random architecture will be returned
                          When False, the performanceo of all trials will be averaged.
        """
        assert isinstance(metric, Metric)
        assert dataset in [
            "cifar10",
            "cifar100",
            "ImageNet16-120",
        ], "Unknown dataset: {}".format(dataset)
        assert epoch >= -1 and epoch < hp
        assert hp in [1, 12, 90], "hp must be 1, 12 or 90"
        if dataset=='cifar10':
            assert metric not in [Metric.VAL_ACCURACY, Metric.VAL_LOSS, Metric.VAL_TIME],\
            "Validation metrics not available for CIFAR-10"

        metric_to_natsbench = {
            Metric.TRAIN_ACCURACY: "train-accuracy",
            Metric.VAL_ACCURACY: "valid-accuracy",
            Metric.TEST_ACCURACY: "test-accuracy",
            Metric.TRAIN_LOSS: "train-loss",
            Metric.VAL_LOSS: "valid-loss",
            Metric.TEST_LOSS: "test-loss",
            Metric.TRAIN_TIME: "train-all-time",
            Metric.VAL_TIME: "valid-all-time",
            Metric.TEST_TIME: "test-all-time"
        }

        if metric not in metric_to_natsbench.keys():
            raise NotImplementedError(f"NATS-Bench does not support querying {metric}")
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query natsbench")

        arch_index = int(''.join([str(ch//8 - 1) for ch in self.channels]), 8)

        if epoch == -1:
            epoch = hp - 1
        hp = f"{hp:02d}"

        if full_lc:
            metrics = []

            for epoch in range(int(hp)):
                result = dataset_api.get_more_info(arch_index, dataset, iepoch=epoch, hp=hp, is_random=is_random)
                metrics.append(result[metric_to_natsbench[metric]])

            return metrics
        else:
            results = dataset_api.get_more_info(arch_index, dataset, iepoch=epoch, hp=hp, is_random=is_random)
            return results[metric_to_natsbench[metric]]

    def get_channels(self):
        return self.channels

    def set_channels(self, channels):
        self.channels = channels

    def get_hash(self):
        return tuple(self.get_channels())

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(self.channel_candidates, repeat=len(self.channels))

    def set_spec(self, channels, dataset_api=None):
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_channels(channels)

    def sample_random_architecture(self, dataset_api=None):
        """
        Randomly sample an architecture
        """
        channels = np.random.choice(self.channel_candidates, size=len(self.channels)).tolist()
        self.set_channels(channels)

    def mutate(self, parent, dataset_api=None):
        """
        Mutate one channel from the parent channels
        """

        base_channels = list(parent.get_channels().copy())
        mutate_index = np.random.randint(len(self.channels)) # Index to perform mutation at

        # Remove number of channels at that index in base_channels from the viable candidates
        candidates = self.channel_candidates.copy()
        candidates.remove(base_channels[mutate_index])

        base_channels[mutate_index] = np.random.choice(candidates)
        self.set_channels(base_channels)

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbours of the architecture
        """
        neighbours = []

        for idx in range(len(self.channels)):
            candidates = self.channel_candidates.copy()
            candidates.remove(self.channels[idx])

            for channels in candidates:
                neighbour_channels = list(self.channels).copy()
                neighbour_channels[idx] = channels
                neighbour = NATSBenchSizeSearchSpace()
                neighbour.set_channels(neighbour_channels)
                neighbour_model = torch.nn.Module()
                neighbour_model.arch = neighbour
                neighbours.append(neighbour_model)

        random.shuffle(neighbours)
        return neighbours

    def get_type(self):
        return "natsbenchsize"
    
    def _set_cell_ops(edge, C):
        edge.data.set(
        "op",
        [
            ops.Identity(),
            ops.Zero(stride=1),
            ops.ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
            ops.ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
            ops.AvgPool1x1(kernel_size=3, stride=1, affine=False),
        ],
    )

