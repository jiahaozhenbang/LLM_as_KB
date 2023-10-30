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
from transformers import GPT2Config

import sys
if '/home/sist/lijiahao/ICL/KNNPrompting-main' not in sys.path:
    sys.path.insert(0, '/home/sist/lijiahao/ICL/KNNPrompting-main')
# if '/home/ljh/ICL/SICL' not in sys.path:
#     sys.path.insert(0, '/home/ljh/ICL/SICL')
from utils.dataset import *
from utils.template import *
from utils.trainset import TrainsetStore
from utils.anchor import AnchorStore

import numpy as np
import pandas as pd
# 导入模型
from sklearn.tree import DecisionTreeClassifier
# 评价包
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "MLP",
    "Naive Bayes",
    "Logistic Regression"
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    LinearSVC(random_state=0, multi_class='crammer_singer'),
    DecisionTreeClassifier(random_state=0),
    MLPClassifier(random_state=0,max_iter=1000),
    GaussianNB(),
    LogisticRegression(random_state=0, multi_class='multinomial'),
]


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
        "--knn",
        type=int,
        default=None,
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
        default='all',
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
        hs = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],output_hidden_states =True,
                               return_dict=True).hidden_states[-1].detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_hs.shape = [1, 1600]
    gen_hs = hs[:, -1, :]

    return gen_hs

def llm_gen_logits(model, prompt, tokenizer, max_context_len):
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

def llm_gen_hsandlogits(model, prompt, tokenizer, max_context_len):
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
    gen_hs = hs[:, -1, :]
    gen_logits = logits[:, -1, :]

    return gen_hs, gen_logits

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
    trainset_store = TrainsetStore(K=anchor_data.__len__(),
                               dim= model_config.n_embd if isinstance(model_config, GPT2Config) else model_config.hidden_size,
                               n_class=len(label2id))
    for ins in tqdm(anchor_data.data, total=anchor_data.__len__()):
        labels = label2id[ins['label']]
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_hs, gen_logits = llm_gen_hsandlogits(model, prompt, tokenizer, max_context_len)
        trainset_store.enqueue(gen_hs, torch.tensor(labels))
    

    # Stage2: dev set
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    dev_labels = []
    dev_pred = []
    dev_probs = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_hs, gen_logits = llm_gen_hsandlogits(model, prompt, tokenizer, max_context_len)
        dev_probs.append(np.array(torch.softmax(gen_logits[0], dim=-1)))
        dev_pred.append(gen_hs)
    import pandas as pd
    df = pd.DataFrame(dev_probs)
    df['class'] = [id2verb[l] for l in dev_labels]
    print(df.head())

    df.to_csv(f'data_statistics/{args.dataset}_train_{args.n_train_shot}_demo_{args.n_demo_shot}_probs.csv')

    def get_probs_from_classifier(feature_ids, feature_name=args.feature_choose):
        chosen_train_x, chosen_train_y = trainset_store.get_data(feature_ids)
        test_x = [all_feature[0, feature_ids].cpu().numpy().tolist() for all_feature in dev_pred]
        test_y = dev_labels

        logger.info(f"===== classification by {feature_name} =====")
        for name, clf in zip(names, classifiers):
            candidate_methods_for_all_feature = [ "Logistic Regression"]
            if name not in candidate_methods_for_all_feature:
                continue
            logger.info(f"Classification by {name}")
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(chosen_train_x, chosen_train_y)
            probs_for_test = clf.predict_proba(test_x)
        return probs_for_test

    # chosen feature's index for classification
    chosen_feature_indexes = list(range(model_config.n_embd if isinstance(model_config, GPT2Config) else model_config.hidden_size))
    probs_from_clf = get_probs_from_classifier(chosen_feature_indexes)
    df = pd.DataFrame(probs_from_clf)
    df['class'] = [id2verb[l] for l in dev_labels]

    print(df.head())

    df.to_csv(f'data_statistics/{args.dataset}_train_{args.n_train_shot}_demo_{args.n_demo_shot}_probs_after_clf.csv')


if __name__ == "__main__":
    main()

"""
export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/


DATASET=sst2
N_TRAIN_SHOT=1
N_DEMO_SHOT=1
SEED=1

python3 data_statistics/get_hs.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} 

"""