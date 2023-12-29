import os
import sys 
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 

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
from utils.trainset import TrainsetStore
from utils.anchor import AnchorStore

import numpy as np

import time
import math


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Prompting.")
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
        default=None,
    )
    parser.add_argument(
        "--n_demo_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_anchor_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fuzzy_label",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--weight_knn",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--temperture",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature_choose",
        type=str,
        choices=[' only_label', 'fuzzy_label', 'all'],
        default=' only_label',
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
        output = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],output_hidden_states =True,
                               return_dict=True)
        hs = output.hidden_states[-1].detach().cpu()
        
        logits = output.logits.detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_hs.shape = [1, 1600]
    gen_hs = hs[0, -1, :]
    gen_logits = logits[0, -1, :]

    return gen_hs, gen_logits


def get_fuzzy_label_ids(model, tokenizer, id2verb, k=10):
    from transformers import GPT2PreTrainedModel
    embeddings = model.transformer.wte.weight.detach() if isinstance(model, GPT2PreTrainedModel) \
                    else model.model.embed_tokens.weight.detach()

    def l2norm(X, dim=-1, eps=1e-8):
        """L2-normalize columns of X"""
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X
    
    embeddings = l2norm(embeddings)
    queryT = torch.transpose(embeddings, 0, 1)
    similarity = torch.mm(embeddings, queryT)
    # print(similarity.shape)

    vocab = [v for v,i in tokenizer.get_vocab().items()]
    
    label_verb_token_id_candidate_list = []
    for label_verb in id2verb:
        label_verb_token_id_candidate_list.append(list(tokenizer.encode(' ' + label_verb)))
    min_len = min([len(_e) for _e in label_verb_token_id_candidate_list])
    for i in range(min_len):
        label_verb_token_id_list = [_e[i] for _e in label_verb_token_id_candidate_list]
        if len(set(label_verb_token_id_list)) < len(label_verb_token_id_list):
            continue
    assert len(set(label_verb_token_id_list)) == len(label_verb_token_id_list), "no valid label_verb_token_id_list"
    
    fuzzy_label_ids = []
    for label_verb_token_id in label_verb_token_id_list:
        similar_value, similar_id = torch.topk(similarity[label_verb_token_id], k=k)
        # print(label_verb, vocab[label_verb_token_id], list(zip(similar_value.tolist(), [vocab[i] for i in similar_id.tolist()])))

        fuzzy_label_ids.append(similar_id.tolist())
    
    return fuzzy_label_ids

def main():
    args = parse_args()

    args.n_anchor_shot = args.n_train_shot - args.n_demo_shot
    if args.n_anchor_shot <= 0:
        raise Exception("Num. of demonstration must be set smaller than num. of training.")

    setup_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    if "30" in args.llm_dir :
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir, device_map='auto') # , device_map='auto'
    elif "70" in args.llm_dir:
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir, device_map='auto',load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
        model.to(device)
    model.eval()

    if 'gpt2' in args.llm_dir:
        max_context_len = 1024
    else:
        max_context_len = 2048
    
    if 'llama2' in args.llm_dir:
        max_context_len = 4096

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
    
    logger.info(f"===== {args.dataset} / {args.n_train_shot} / {args.seed} =====")

    datadir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(datadir, mode='train')
    dev_data = AutoDataset(datadir, mode='dev')

    anchor_data = AutoDataset(datadir, mode='train')

    # Stage1: Meta Test
    train_data.subsamplebyshot(args.n_demo_shot, args.seed)
    prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
    anchor_data.subsamplebyshot(args.n_anchor_shot, args.seed, exclude=train_data.data)
    label2id = dev_data.label2id
    id2verb = train_data.id2verb
    
    # Stage1: train set

    logger.info(f"===== build train set store of {train_data.__len__()} examples =====")
    trainset_feature = []
    trainset_label = []
    for ins in tqdm(anchor_data.data, total=anchor_data.__len__()):
        labels = label2id[ins['label']]
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_hs, gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        # trainset_store.enqueue(gen_hs, torch.tensor(labels))
        trainset_feature.append(np.array(gen_hs))
        trainset_label.append(labels)


    # Stage2: dev set
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    chosen_feature_indexes = get_fuzzy_label_ids(model, tokenizer, id2verb, k=args.fuzzy_label)
    dev_labels = []
    dev_preds = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_hs, gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)

        # knn probs
        knn_probs = [0] * len(label2id)
        for train_hs, _label in zip(trainset_feature, trainset_label):
            dis = np.linalg.norm(gen_hs - train_hs)
            knn_probs[_label] += math.exp(- dis /args.temperture)
        knn_probs = np.array(knn_probs)
        knn_probs = knn_probs / sum(knn_probs)

        # LM probs
        orginal_probs = torch.softmax(gen_logits, dim=-1)
        LM_probs = []
        for i in range(len(label2id)):
            LM_probs.append(orginal_probs[chosen_feature_indexes[i]].sum())
        LM_probs = np.array(LM_probs)
        LM_probs = LM_probs / sum(LM_probs)

        # final interpolate
        final_probs = (1-  args.weight_knn) * LM_probs + args.weight_knn * knn_probs
        dev_preds.append(np.argmax(final_probs))


    # Stage 3: record

    dev_correct = [1 if dev_labels[i] == dev_preds[i] else 0 for i in range(len(dev_labels))]
    acc = sum(dev_correct) / len(dev_labels)
    logger.info(f"Acc: {acc}")

    # logging
    save_results_file = os.path.join(args.output_dir, 'results_knnprompt.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'n_demo_shot', 'n_anchor_shot', 'seed', 'fuzzy_label', 'acc'])
        csvwriter.writerow([args.dataset,
                            args.llm_dir,
                            args.n_train_shot,
                            args.n_demo_shot,
                            args.n_anchor_shot,
                            args.seed,
                            args.fuzzy_label,
                            acc])

if __name__ == "__main__":
    main()
