import argparse
import logging
import os
import random
import json

from typing import Dict, Tuple
from argparse import Namespace

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from .modeling_bert import BertMultiSampelsWithInfoNceLoss
from .dataset import WowDataset

from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_train,
    run_batch_eval,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

os.environ['TRANSFORMERS_OFFLINE'] = '1'


def get_classes(args):
    """ Get classes for dataset, model, training func, and eval func """
    model = args.model_name_or_path
    return WowDataset, BertMultiSampelsWithInfoNceLoss, run_batch_train, run_batch_eval


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
          run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    """ Model training and evaluation """
    log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
    tb_writer = SummaryWriter(log_dir)
    args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    if 0 < args.warmup_steps < 1:
        args.warmup_steps = int(args.warmup_steps * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    set_seed(args)  # for reproducibility
    val_loss = float('inf')

    for _ in train_iterator:
        local_steps = 0  # update step
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        step = 0  # backward step
        total_log_loss = 0
        for _, batch in enumerate(epoch_iterator):
            model.train()
            loss, _ = run_batch_fn_train(args, model, batch, global_step=global_step)
            step += 1

            total_log_loss += loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)
                total_log_loss = 0

            if global_step != 0 and (global_step % 50 == 0):

                results = evaluate(args, eval_dataset, model, run_batch_fn_eval, desc=str(global_step))
                print(f'"steps:"{global_step}\n {results}')

                output_dir = os.path.join(args.output_dir, "{}-{}".format("checkpoint", global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )

                logger.info("Saving model checkpoint to %s", output_dir)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                    json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
                logger.info("Saving model checkpoint to %s", output_dir)

    tb_writer.flush()
    tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, run_batch_fn, desc="") -> Dict:
    """ Model evaluation for knowledge seeking turn detection and knowledge selection
        Report evaluation results if gold labels are available
    """
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle different number of candidates
    args.eval_batch_size = 1

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    model.eval()
    scores = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
        with torch.no_grad():
            score, indics = run_batch_fn(args, model, batch)
            scores.append(
                {
                    "score": score,
                    "indics": indics
                }
            )
    return get_eval_performance(args, scores)


def get_eval_performance(args, scores):
    """ Get evaluation performance when the gold labels are available """
    total = len(scores)
    r_cnt_1 = 0
    r_cnt_2 = 0
    r_cnt_3 = 0
    r_cnt_5 = 0

    k_cnt_1 = 0
    k_cnt_2 = 0
    k_cnt_3 = 0
    k_cnt_5 = 0

    rk_cnt_1 = 0
    rk_cnt_2 = 0
    rk_cnt_3 = 0
    rk_cnt_5 = 0
    for item in scores:
        indics = item["indics"]
        if len(indics) >= 1 and [0,0] in indics[:1]:
            rk_cnt_1 += 1
        if len(indics) >= 1 and 0 in indics[:1,0]:
            r_cnt_1 += 1
        if len(indics) >= 1 and 0 in indics[:1,1]:
            k_cnt_1 += 1

        if len(indics) >= 2 and [0,0] in indics[:2]:
            rk_cnt_2 += 1
        if len(indics) >= 2 and 0 in indics[:2,0]:
            r_cnt_2 += 1
        if len(indics) >= 2 and 0 in indics[:2,1]:
            k_cnt_2 += 1

        if len(indics) >= 3 and [0,0] in indics[:3]:
            rk_cnt_3 += 1
        if len(indics) >= 3 and 0 in indics[:3,0]:
            r_cnt_3 += 1
        if len(indics) >= 3 and 0 in indics[:3,1]:
            k_cnt_3 += 1

        if len(indics) >= 5 and [0,0] in indics[:5]:
            rk_cnt_5 += 1
        if len(indics) >= 5 and 0 in indics[:5,0]:
            r_cnt_5 += 1
        if len(indics) >= 5 and 0 in indics[:5,1]:
            k_cnt_5 += 1
            
    return {
        "knowledge": {
            "knowledge_accuracy": float(k_cnt_1 / total),
            "knowledge_recall_2": float(k_cnt_2 / total),
            "knowledge_recall_3": float(k_cnt_3 / total),
            "knowledge_recall_5": float(k_cnt_5 / total),
        },
        "response": {
            "response_accuracy": float(r_cnt_1 / total),
            "response_recall_2": float(r_cnt_2 / total),
            "response_recall_3": float(r_cnt_3 / total),
            "response_recall_5": float(r_cnt_5 / total),
        },
        "both": {
            "accuracy": float(rk_cnt_1 / total),
            "recall_2": float(rk_cnt_2 / total),
            "recall_3": float(rk_cnt_3 / total),
            "recall_5": float(rk_cnt_5 / total),
        }
    }


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--model_name_or_path", type=str, help="model_name_or_path", default='bert')
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--debug", type=int, default=0,
                        help="If set, will only use a small number (==debug) of data for training and test.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.eval_only = args.eval_only
    dataset_args.debug = args.debug

    # Setup CUDA & GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Set seed
    set_seed(args)

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args)

    if args.eval_only:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        # Evaluation
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset)
        result = evaluate(args, eval_dataset, model, run_batch_fn_eval, desc=args.eval_desc or "val")
        print(result)

    else:
        if args.checkpoint is not None:
            model = model_class.from_pretrained(args.checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
            model.to(args.device)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            tokenizer.model_max_length = min(1024, tokenizer.model_max_length)
            model = model_class.from_pretrained(args.model_name_or_path, config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)

        # load datasets and train the model
        train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type="val")  # main difference is during evaluation, val need to go through all snippets
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train,
                                     run_batch_fn_eval)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
