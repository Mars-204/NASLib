"""
Test the evaluation accuracies from the checkpoints

"""
import codecs
from curses import flash

from naslib.search_spaces.core.graph import Graph
import time
import json
import logging
import os
import copy
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from fvcore.common.checkpoint import PeriodicCheckpointer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils
from naslib.utils.logging import log_every_n_seconds, log_first_n


logger = logging.getLogger(__name__)

import logging
from pyexpat import model
import sys
import naslib as nl


from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    OneShotNASOptimizer,
    RandomNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    DrNASOptimizer,
)

from naslib.search_spaces import NasBench201SearchSpace, DartsSearchSpace, NasBench101SearchSpace, NATSBenchSizeSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    "darts": DARTSOptimizer(config),
    "gdas": GDASOptimizer(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
    "re": RegularizedEvolution(config),
    "rs": RandomSearch(config),
    "ls": RandomSearch(config),
    "bananas": Bananas(config),
    "bp": BasePredictor(config),
    "drnas": DrNASOptimizer(config),
}

if config.dataset =='cifar100':
    num_classes=100
elif config.dataset=='ImageNet16-120':
    num_classes=120
else:
    num_classes=10
supported_search_space ={
    "nasbench201" : NasBench201SearchSpace(),#num_classes),
    "darts" : DartsSearchSpace(),#num_classes),
    "nasbench101" : NasBench101SearchSpace(),#num_classes)
    "natsbenchsize" : NATSBenchSizeSearchSpace(),
}

search_space = supported_search_space[config.search_space]
dataset_api = get_dataset_api(config.search_space, config.dataset)

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)
checkpoint = utils.get_last_checkpoint(config,search = False)
best_arch = optimizer.get_final_architecture()


# Initialization

eval_dataset = config.dataset
train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode = "val"
        )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# measure final test accuracy
top1 = utils.AverageMeter()
top5 = utils.AverageMeter()

best_arch.eval()

for i, data_test in enumerate(test_queue):
    input_test, target_test = data_test
    input_test = input_test.to(device)
    target_test = target_test.to(device, non_blocking=True)

    n = input_test.size(0)

    with torch.no_grad():
        logits = best_arch(input_test)

        prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    log_every_n_seconds(
        logging.INFO,
        "Inference batch {} of {}.".format(i, len(test_queue)),
        n=5,
    )

logger.info(
    "Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(
        top1.avg, top5.avg
    )
)

# Evaulating corruption accuracy

mean_CE = utils.test_corr(best_arch, eval_dataset, config)
logger.info(
"Corruption Evaluation finished. Mean Corruption Error: {:.9}".format(
    mean_CE)
)

# Querying results from benchmark

metric = Metric.TEST_ACCURACY
result = best_arch.query(
    metric=metric, dataset=config.dataset, dataset_api=dataset_api
)
logger.info("Queried results ({}): {}".format(metric, result))


best_arch.to(device)
