import numpy as np
import torch
import logging
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from naslib.search_spaces.core.primitives import AbstractPrimitive, MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DrNASOptimizer(DARTSOptimizer):
    """
    Implementation of DrNAS optimizer introduced in the paper
        DrNAS: Dirichlet Neural Architecture Search (ICLR2021)
    note: many functions are similar to the DARTS optimizer so it makes sense to inherit this class directly
    from DARTSOptimizer instead of MetaOptimizer
    """

    @staticmethod
    def sample_alphas(edge):
        beta = F.elu(edge.data.alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        edge.data.set("sampled_arch_weight", weights, shared=True)

    @staticmethod
    def remove_sampled_alphas(edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the DrNAS specific DrNASMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DrNASMixedOp(primitives))

    def __init__(
        self,
        config,
        op_optimizer=torch.optim.SGD,
        arch_optimizer=torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Initialize a new instance.

        Args:

        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)
        self.reg_type = "kl"
        self.reg_scale = 1e-3
        # self.reg_scale = config.reg_scale
        self.epochs = config.search.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmix = config.search.augmix   # JSD loss configured
        self.search_space = config.search_space
        self.jsd_factor = config.jsd_factor

    def new_epoch(self, epoch):
        super().new_epoch(epoch)

    def adapt_search_space(self, search_space, scope=None):
        """
        Same as in darts with a different mixop.
        If you want to checkpoint the dirichlet 'concentration' parameter (beta) add it to the buffer here.
        """
        super().adapt_search_space(search_space, scope)
        self.anchor = Dirichlet(
            torch.ones_like(
                torch.nn.utils.parameters_to_vector(self.architectural_weights)
            ).to(self.device)
        )

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val
        # sample weights (alphas) from the dirichlet distribution (parameterized by beta) and set to edges
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update architecture weights
        self.arch_optimizer.zero_grad()
        logits_val = self.graph(input_val)
        if self.augmix_search:
            logits_val, _, _ = torch.split(logits_val, len(logits_val) // 3)

        val_loss = self.loss(logits_val, target_val)

        if self.reg_type == "kl":
            val_loss += self._get_kl_reg()

        val_loss.backward()

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.architectural_weights.parameters(), self.grad_clip
            )
        self.arch_optimizer.step()

        # has to be done again, cause val_loss.backward() frees the gradient from sampled alphas
        # TODO: this is not how it is intended because the samples are now different. Another
        # option would be to set val_loss.backward(retain_graph=True) but that requires more memory.
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        if self.augmix_search :
            logits_train, augmix_loss = self.jsd_loss(logits_train)
            train_loss = self.loss(logits_train, target_train) + augmix_loss
        # elif self.augment and not self.augmix:
        #     logits_train, _, _ = torch.split(logits_train, len(logits_train) // 3)
        #     train_loss = self.loss(logits_train, target_train)
        else:
            train_loss = self.loss(logits_train, target_train)
        
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        # in order to properly unparse remove the alphas again
        self.graph.update_edges(
            update_func=self.remove_sampled_alphas,
            scope=self.scope,
            private_edge_data=False,
        )

        return logits_train, logits_val, train_loss, val_loss
    
    def jsd_loss(self, logits_train):
        logits_train, logits_aug1, logits_aug2 = torch.split(logits_train, len(logits_train) // 3)
        p_clean, p_aug1, p_aug2 = F.softmax(logits_train, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)

        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        augmix_loss = self.config.jsd_factor * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        return logits_train, augmix_loss

    def _get_kl_reg(self):
        cons = (
            F.elu(torch.nn.utils.parameters_to_vector(self.architectural_weights)) + 1
        )
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg

    def get_final_architecture(self):
        logger.info(
            "Arch weights before discretization: {}".format(
                [a for a in self.architectural_weights]
            )
        )
        # if self.search_space=='nasbench301':
        #     import copy
        #     graph = copy.copy(self.graph).unparse()
        # else:
        #     graph = self.graph.clone().unparse()
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                edge.data.set("op", primitives[np.argmax(alphas)])

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        graph = graph.to(self.device)
        return graph

    def get_checkpointables(self):
        """
        Return all objects that should be saved in a checkpoint during training.
        Will be called after `before_training` and must include key "model".
        Returns:
            (dict): with name as key and object as value. e.g. graph, arch weights, optimizers, ...
        """
        return {
            "graph": self.graph,
            "op_optimizer": self.op_optimizer,
            "op_optimizer_evaluate": self.op_optimizer_evaluate,
            "arch_optimizer": self.arch_optimizer,
            "arch_weights": self.architectural_weights,            
        }


class DrNASMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed ops

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def get_weights(self, edge_data):
        return edge_data.sampled_arch_weight

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        weighted_sum = sum(
            w * op(x, None)
            for w, op in zip(weights, self.primitives)
        )
        return weighted_sum
