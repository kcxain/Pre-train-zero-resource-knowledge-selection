import time

from transformers import EvalPrediction
from transformers.trainer import Trainer


import collections
from typing import  Any, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset

import numpy as np
import torch

from torch import nn
from torch.utils.data.dataloader import DataLoader


from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    nested_concat,
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput, speed_metrics,
)
from torch.utils.data.sampler import  SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.utils import logging

logger = logging.get_logger(__name__)

class PtrTrainerPredictit(Trainer):
    def __init__(self,
        nagative_samples=9,
        eval_examples=None,
        *args,
        **kwargs,
    ):
        self.nagative_samples = nagative_samples
        self.eval_examples=eval_examples
        super().__init__(*args, **kwargs)

    # def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
    #     if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
    #             self.train_dataset, collections.abc.Sized
    #     ):
    #         return None
    #     else:
    #         return (
    #             SequentialSampler(self.train_dataset)
    #             if self.args.local_rank == -1
    #             else DistributedSampler(self.train_dataset)
    #         )

    def predict(
            self,
            test_dataset: Dataset,
            test_examples,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval"
    ) :

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,test_examples=test_examples
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))
        return output.metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            # loss = outputs["loss"]

        return (None, outputs["scores"], None)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        test_examples,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples*self.nagative_samples)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # logits = logits.squeeze(dim=0).squeeze(dim=1)
            # labels = labels.squeeze(dim=0).squeeze(dim=1)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None :
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids),test_examples)
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            test_examples=self.eval_examples,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        length = len(eval_dataset) if eval_dataset is not None else len(self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, length))
        return output.metrics

        # self.log(output.metrics)



