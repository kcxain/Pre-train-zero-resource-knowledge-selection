import copy

import numpy as np
import logging

logger = logging.getLogger(__name__)


def run_batch_train(args, model, batch, **kwargs):
    """ Run batch knowledge selection during training time """
    input_ids, attention_mask, token_type_ids = batch
    model_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    loss, logits = model_outputs[0], model_outputs[1]
    return loss, logits


def run_batch_eval(args, model, batch, **kwargs):
    """ Run batch knowledge selection during evaluation time """
    # return: loss, logits, labels
    input_ids, attention_mask, token_type_ids = batch

    topk_knowledge = args.topk_knowledge
    topk_response = args.topk_response

    model_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        is_predict=True,
        topk_knowledge=topk_knowledge,
        topk_response=topk_response
    )
    score, indics = model_outputs
    return np.array(score.tolist()), np.array(indics.tolist())
