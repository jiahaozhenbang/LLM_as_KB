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
        default=' only_label',
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args



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

    start_time =  time.time()
    # Stage1: Meta Test
    train_data.subsamplebyshot(args.n_demo_shot, args.seed)
    prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
    anchor_data.subsamplebyshot(args.n_anchor_shot, args.seed, exclude=train_data.data)
    label2id = dev_data.label2id
    id2verb = train_data.id2verb
    
    # Stage1: train set

    logger.info(f"===== build train set store of {train_data.__len__()} examples =====")

    from openai import OpenAI
    client = OpenAI(api_key="")
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    # df.to_csv('output/embedded_1k_reviews.csv', index=False)
    labels = []
    prompts = []
    for ins in tqdm(anchor_data.data, total=anchor_data.__len__()):
        label = label2id[ins['label']]
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        labels.append(label)
        prompts.append(prompt)
    
    import pandas as pd
    df = pd.DataFrame({'labels': labels, 'text':prompts})
    cash_path = f"data/{args.dataset}_embeddings_{args.n_train_shot}shot_seed_{args.seed}.csv"
    if not os.path.exists(cash_path):
        df["embedding"] = df.text.apply(lambda x: get_embedding(x))
        df.to_csv(cash_path)
    else:
        df = pd.read_csv(cash_path)
        df['embedding'] = df.embedding.apply(eval).apply(np.array)
    print(df.head())

    # Stage2: dev set
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    dev_labels = []
    dev_prompts = []
    for ins in tqdm(dev_data.data, total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        dev_prompts.append(prompt)

    dev_df = pd.DataFrame({'dev_labels': dev_labels, 'text':dev_prompts})
    dev_cash_path = f"data/{args.dataset}_dev_embeddings_{args.n_train_shot}shot_seed_{args.seed}.csv"
    if not os.path.exists(dev_cash_path):
        dev_df["embedding"] = dev_df.text.apply(lambda x: get_embedding(x))
        dev_df.to_csv(dev_cash_path)
    else:
        dev_df = pd.read_csv(dev_cash_path)
        dev_df['embedding'] = dev_df.embedding.apply(eval).apply(np.array)
    print(dev_df.head())

    # Stage 3: dev set classification
    
    def classification_by_feature(feature_name=args.feature_choose):
        chosen_train_x, chosen_train_y = [e.tolist() if not isinstance(e, list) else e for e in df['embedding']], df['labels']
        test_x = [e.tolist() if not isinstance(e, list) else e for e in dev_df['embedding']]
        test_y = dev_df['dev_labels']

        logger.info(f"===== classification by {feature_name} =====")
        for name, clf in zip(names, classifiers):
            candidate_methods_for_all_feature = ["Nearest Neighbors", "MLP", "Logistic Regression"]
            if feature_name == 'all' and  name not in candidate_methods_for_all_feature:
                continue
            logger.info(f"Classification by {name}")
            start_time =  time.time()
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(chosen_train_x, chosen_train_y)
            score = clf.score(test_x, test_y)
            end_time =  time.time()
            running_time = end_time - start_time
            logger.info(f"accuracy_score: {score}; running_time: {running_time}s")

            # logging
            save_results_file = os.path.join(args.output_dir, 'fads-icl.csv')
            csv_exists = os.path.isfile(save_results_file)
            with open(save_results_file, 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if not csv_exists:
                    csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'n_demo_shot', 'method', 'seed', 'acc', 'running_time'])
                csvwriter.writerow([args.dataset,
                                    args.llm_dir,
                                    args.n_train_shot,
                                    args.n_demo_shot,
                                    name,
                                    args.seed,
                                    score,
                                    running_time])
    
    classification_by_feature()

    # kNN prompting
    import torch.nn.functional as F
    anchor_store = AnchorStore(K=anchor_data.__len__(),
                               dim=len(df['embedding'][0]),
                               knn=3,
                               n_class=len(label2id))
    for i, ins in tqdm(enumerate(anchor_data.data), total=anchor_data.__len__()):
        labels = label2id[ins['label']]
        anchor_store.enqueue(torch.softmax(torch.tensor(df['embedding'][i]),dim=0).unsqueeze(0), torch.tensor(labels))

    # Stage2: Formal Test
    dev_labels = []
    dev_pred = []
    for i, ins in tqdm(enumerate(dev_data.data), total=dev_data.__len__()):
        dev_labels.append(label2id[ins['label']])
        dev_pred.extend(anchor_store.knn_infer(torch.softmax(torch.tensor(dev_df['embedding'][i]), dim=0).unsqueeze(0)))

    dev_correct = [1 if dev_labels[i] == dev_pred[i] else 0 for i in range(len(dev_labels))]
    acc = sum(dev_correct) / len(dev_labels)
    logger.info(f"kNN-prompting Acc: {acc}")

    save_results_file = os.path.join(args.output_dir, 'knn-prompting.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'n_demo_shot', 'seed', 'acc'])
        csvwriter.writerow([args.dataset,
                            args.llm_dir,
                            args.n_train_shot,
                            args.n_demo_shot,
                            args.seed,
                            acc])
    



if __name__ == "__main__":
    main()
