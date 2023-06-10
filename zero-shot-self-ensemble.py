import json
import random
from datetime import datetime
from time import sleep
import logging
import argparse
from tqdm import tqdm
import csv
import os

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils.dataset import *
from utils.template import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument(
        "--llm_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_train_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_logits.shape = [1, 50257]
    gen_logits = logits[:, -1, :]

    return gen_logits


def parse_response(gen_logits, tokenizer, id2verb):
    gen_prob = torch.softmax(gen_logits, dim=-1)
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
        prob_per_cls.append(gen_prob[:, label_verb_token_id])
    pred = torch.argmax(torch.cat(prob_per_cls, dim=0)).tolist()
    return pred

def parse_response_with_probs(gen_logits, tokenizer, id2verb):
    gen_prob = torch.softmax(gen_logits, dim=-1)
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
        prob_per_cls.append(gen_prob[:, label_verb_token_id])
    pred = torch.argmax(torch.cat(prob_per_cls, dim=0)).tolist()
    return pred, torch.cat(prob_per_cls, dim=0)


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
    model.to(device)
    model.eval()

    if 'gpt2' in args.llm_dir:
        max_context_len = 1024
    else:
        max_context_len = 2048

    # prepare dataset
    if args.dataset == 'sst2':
        AutoDataset = SST2Dataset
    elif args.dataset == 'subj':
        AutoDataset = SUBJDataset
    elif args.dataset == 'agnews':
        AutoDataset = AGNEWSDataset
    elif args.dataset == 'cb':
        AutoDataset = CBDataset
    elif args.dataset == 'cr':
        AutoDataset = CRDataset
    elif args.dataset == 'dbpedia':
        AutoDataset = DBPEDIADataset
    elif args.dataset == 'mpqa':
        AutoDataset = MPQADataset
    elif args.dataset == 'mr':
        AutoDataset = MRDataset
    elif args.dataset == 'rte':
        AutoDataset = RTEDataset
    elif args.dataset == 'trec':
        AutoDataset = TRECDataset

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(dataset_dir, mode='train')
    dev_data = AutoDataset(dataset_dir, mode='dev')

    # inference
    train_data.subsamplebyshot(1, args.seed)
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")

    dev_labels = []

    dev_pred_ensemble_before_softmax = []
    dev_pred_ensemble_after_softmax = []
    dev_pred2 = []

    label2id = dev_data.label2id
    id2verb = train_data.id2verb

    for ins in tqdm(dev_data.data, total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        print(f"groundtruth: {label2id[ins['label']]}")

        # Zero-shot(false one-shot)
        dummy_train_data = AutoDataset(dataset_dir, mode='train')
        dummy_ins = ins.copy()

        dev_pred_from_single_demon = []
        for label in label2id:
            dummy_ins['label'] = label
            dummy_train_data.data = [dummy_ins]
            prompt_prefix = make_prompt(dummy_train_data, args.dataset, mode='train')

            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
            pred, probs = parse_response_with_probs(gen_logits, tokenizer, id2verb)
            dev_pred_from_single_demon.append(probs)
        pred_matrix = torch.stack(dev_pred_from_single_demon)
        probs_ensemble_before_softmax = pred_matrix.sum(dim=0)
        print(pred_matrix)
        print(probs_ensemble_before_softmax)
        exit()
        probs_ensemble_after_softmax = torch.softmax(pred_matrix, dim=1).sum(dim=0)

        dev_pred_ensemble_before_softmax.append((torch.argmax(probs_ensemble_before_softmax).tolist(), probs_ensemble_before_softmax))
        dev_pred_ensemble_after_softmax.append((torch.argmax(probs_ensemble_after_softmax).tolist(), probs_ensemble_after_softmax))

        print(prompt)
        print(f"Zero--shot(ensemble_before_softmax): pred: {dev_pred_ensemble_before_softmax[-1][0]}; probs/verb: {dev_pred_ensemble_before_softmax[-1][1]}")
        print(f"Zero--shot(ensemble_after_softmax): pred: {dev_pred_ensemble_after_softmax[-1][0]}; probs/verb: {dev_pred_ensemble_after_softmax[-1][1]}")


    # for ins in tqdm(dev_data.data, total=dev_data.__len__()):
    #     dev_labels.append(label2id[ins['label']])
    #     prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
    #     gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
    #     dev_pred.append(parse_response(gen_logits, tokenizer, id2verb))
    def acc(dev_pred):
        dev_correct = [1 if dev_labels[i] == dev_pred[i][0] else 0 for i in range(len(dev_labels))]
        acc = sum(dev_correct) / len(dev_labels)
        soft_acc = sum([dev_pred[i][1] for i in range(len(dev_labels))])
        return acc, soft_acc

    logger.info(f"Zero-shot(ensemble_before_softmax) Acc: {acc(dev_pred_ensemble_before_softmax)}")
    logger.info(f"Zero--shot(ensemble_after_softmax) Acc: {acc(dev_pred_ensemble_after_softmax)}")


    # TODO: soft acc

    # # logging
    # save_results_file = os.path.join(args.output_dir, 'results_icl.csv')
    # csv_exists = os.path.isfile(save_results_file)
    # with open(save_results_file, 'a+', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     if not csv_exists:
    #         csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'seed', 'acc'])
    #     csvwriter.writerow([args.dataset,
    #                         args.llm_dir,
    #                         args.n_train_shot,
    #                         args.seed,
    #                         acc])

if __name__ == "__main__":
    main()

# nohup bash run_zero-shot-self-ensemble.sh 2>&1 >log/zero-shot-self-ensemble.log &