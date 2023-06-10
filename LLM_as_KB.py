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

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Gaussian Process",
    "Decision Tree",
    "MLP",
    "AdaBoost",
    "Naive Bayes",
    "Logistic Regression"
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    LinearSVC(random_state=0, multi_class='crammer_singer'),
    GaussianProcessClassifier(1.0 * RBF(1.0), multi_class='one_vs_rest', random_state=0),
    DecisionTreeClassifier(random_state=0),
    MLPClassifier(random_state=0,),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), random_state=0,),
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
        "--knn",
        type=int,
        default=None,
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

def main():
    args = parse_args()

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

    datadir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(datadir, mode='train')
    dev_data = AutoDataset(datadir, mode='dev')

    # Stage1: get train set 
    train_data.subsamplebyshot(args.n_train_shot, args.seed)
    label2id = dev_data.label2id
    id2verb = train_data.id2verb

    chosen_feature_indexes = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
        chosen_feature_indexes.append(label_verb_token_id)
    
    logger.info(f"===== build train set store of {train_data.__len__()} examples =====")
    trainset_store = TrainsetStore(K=train_data.__len__(),
                               dim=model_config.vocab_size,
                               n_class=len(label2id))
    for ins in tqdm(train_data.data, total=train_data.__len__()):
        labels = label2id[ins['label']]
        prompt =  make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        trainset_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))

    chosen_train_x, chosen_train_y = trainset_store.get_data(chosen_feature_indexes)

    # Stage2: dev set
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    dev_labels = []
    dev_pred = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        prompt = make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        dev_pred.append(torch.softmax(gen_logits, dim=-1)[0, chosen_feature_indexes].cpu().numpy().tolist())
    
    test_x = dev_pred
    test_y = dev_labels


    # Stage 3: dev set classification
    for name, clf in zip(names, classifiers):
        logger.info(f"Classification by {name}")
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(chosen_train_x, chosen_train_y)
        score = clf.score(test_x, test_y)
        logger.info(f"accuracy_score: {score}")

        # logging
        save_results_file = os.path.join(args.output_dir, 'results_KB_only_label_feature_{}.csv'.format(args.dataset))
        csv_exists = os.path.isfile(save_results_file)
        with open(save_results_file, 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not csv_exists:
                csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'seed', 'acc', 'method'])
            csvwriter.writerow([args.dataset,
                                args.llm_dir,
                                args.n_train_shot,
                                args.seed,
                                score,
                                name])
    



if __name__ == "__main__":
    main()
