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

def list_move_left(A, a):
    for i in range(a):
        A.insert(len(A), A[0])
        A.pop(0)
    return A

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
    # train_data.subsamplebyshot(1, args.seed)
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")

    dev_labels = []

    dev_pred = []


    label2id = dev_data.label2id
    id2verb = train_data.id2verb

    save_data = {'ground-truth': [], 'zero-shot-p_pos': [], 'zero-shot-p_neg': [], '1st-self-one-shot-p_pos': [], '1st-self-one-shot-p_neg': [], '2nd-self-one-shot-p_pos': [], '2nd-self-one-shot-p_neg': []}

    for ins in tqdm(train_data.data[:len(dev_data)], total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        print(f"groundtruth: {label2id[ins['label']]}")

        # Zero-shot(false one-shot)
        dummy_train_data = AutoDataset(dataset_dir, mode='train')
        dummy_ins = ins.copy()

        dummy_train_data.data = []
        for label in label2id:
            dummy_ins['label'] = label
            dummy_train_data.data.append(dummy_ins.copy())

        prompt_prefix = make_prompt(dummy_train_data, args.dataset, mode='train')
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        print(prompt)
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        pred, probs = parse_response_with_probs(gen_logits, tokenizer, id2verb)
        dev_pred.append([(pred, probs)])
        print((pred, probs))

        for i in range(len(label2id) - 1):
            dummy_train_data.data = list_move_left(dummy_train_data.data, 1)
            prompt_prefix = make_prompt(dummy_train_data, args.dataset, mode='train')
            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            print(prompt)
            gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
            pred, probs = parse_response_with_probs(gen_logits, tokenizer, id2verb)
            dev_pred[-1].append((pred, probs))
            print((pred, probs))
        
        # Zero--shot
        prompt =  make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        pred, probs = parse_response_with_probs(gen_logits, tokenizer, id2verb)
        dev_pred[-1].append((pred, probs))
        print('Zero--shot', (pred, probs))


    def acc(dev_pred):
        dev_correct = [1 if dev_labels[i] == dev_pred[i][0] else 0 for i in range(len(dev_labels))]
        acc = sum(dev_correct) / len(dev_labels)
        return acc
    for i in range(len(label2id)):
        logger.info(f"Zero-shot({i}-th one shot) Acc: {acc([item[i] for item in dev_pred])}")


    # TODO: soft acc

    # save
    save_data['ground-truth'] = dev_labels
    save_data['1st-self-one-shot-p_neg'] = [item[0][1][0].item() for item in dev_pred]
    save_data['1st-self-one-shot-p_pos'] = [item[0][1][1].item() for item in dev_pred]
    save_data['2nd-self-one-shot-p_neg'] = [item[1][1][0].item() for item in dev_pred]
    save_data['2nd-self-one-shot-p_pos'] = [item[1][1][1].item() for item in dev_pred]
    save_data['zero-shot-p_neg'] = [item[2][1][0].item() for item in dev_pred]
    save_data['zero-shot-p_pos'] = [item[2][1][1].item() for item in dev_pred]

    import pandas as pd
    df = pd.DataFrame(save_data)
    save_results_file = os.path.join(args.output_dir, 'results_icl_train256.csv')
    df.to_csv(save_results_file, index=False)

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

# nohup bash zero-shot-self-one-shot.sh 2>&1 >log/zero-shot-self-one-shot.log &