################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

# Compute the average online accuracy as defined in the CLOC paper:
# Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data.
# https://arxiv.org/pdf/2108.09020.pdf

from avalanche_extend.evaluation.metrics import Accuracy

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
import torch
from typing import (
    List
)


class OnlineAccuracy(PluginMetric[float]):
    """
    This metric will return the average online accuracy on the next unseen training batch 
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self._accuracy_metric = Accuracy()

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._accuracy_metric.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._accuracy_metric.result()

    def before_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_test_task_id
        else:
            task_labels = task_labels[0]
        
        with torch.no_grad():
            strategy.test_model.eval()
            output = strategy.test_model(strategy.mb_test_x)
            self._accuracy_metric.update(output, strategy.mb_test_y, task_labels)
            strategy.test_model.train()


    def after_training_iteration(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)
        
        
    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self._accuracy_metric.result()
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Top1_Online_Acc_Iteration"

    
def online_accuracy_metric() -> List[PluginMetric]:
    """Helper method that returns an instance of OnlineAccuracy metric.
    """
    metrics = []
    metrics.append(OnlineAccuracy())
    return metrics


__all__ = [
    "OnlineAccuracy",
    "online_accuracy_metric"
]

