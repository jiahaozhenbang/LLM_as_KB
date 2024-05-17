from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, AutoPeftModelForCausalLM
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import GPT2Config
import numpy as np
import random
import sys 
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 

import argparse
from tqdm import tqdm
import csv

from utils.dataset import *
from utils.template import *
from utils.trainset import TrainsetStore
from utils.anchor import AnchorStore
import logging
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Lora")
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
        "--output_dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args
def main():
    
    args = parse_args()
    setup_seed(args.seed)
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    peft_config = LoraConfig(task_type="CAUSAL_LM" ,fan_in_fan_out=True, \
                             inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05) #layers_to_transform=35,

    model = model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # for name,param in model.named_parameters():
    #     print(name, param.numel(), param.requires_grad)

    if 'gpt2' in args.llm_dir:
        max_context_len = 1024
    else:
        max_context_len = 4096 if 'llama2' in args.llm_dir else 2048

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


    train_data.subsamplebyshot(args.n_train_shot, args.seed)
    # label2id = dev_data.label2id
    # id2verb = train_data.id2verb
    label2verb = train_data.label2verb

    add_eos_token=False

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_context_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_context_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, train_on_inputs=False):
        full_prompt = make_full_prompt(data_point, label2verb, args.dataset, mode="train")
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = make_prompt(data_point, args.dataset, mode='inference')
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            # tokenized_full_prompt["labels"]=  [-100] * user_prompt_len \
            #     + [tokenized_full_prompt["labels"][user_prompt_len]]\
            #         + [-100] * (len(tokenized_full_prompt["input_ids"]) - user_prompt_len - 1)
        return tokenized_full_prompt



    _data = Dataset.from_list(train_data.data)
    train_data = _data.shuffle().map(generate_and_tokenize_prompt).remove_columns(train_data.data[0].keys())
    _data = Dataset.from_list(dev_data.data)
    dev_data = _data.map(generate_and_tokenize_prompt).remove_columns(dev_data.data[0].keys())


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=4,
        eval_steps=4,
        logging_steps=4,
        metric_for_best_model="ACC",
        load_best_model_at_end=True,
        save_total_limit=1,
        lr_scheduler_type="constant",
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.Tensor(predictions)
        labels = torch.LongTensor(labels)
        # 创建布尔张量，表示每个元素是否不等于 -100
        bool_matrix = (labels != -100)

        # 找到每一行中第一个非 -100 元素的索引
        indices = torch.argmax(bool_matrix.int(), dim=1)
        label_ids = labels[torch.arange(labels.size(0)), indices]

        # print(label_ids)

        all_candidate_labels = torch.unique(label_ids)

        target_prediction = predictions[torch.arange(predictions.size(0)), indices, :]
        predict_result = torch.all(target_prediction[:, all_candidate_labels] - target_prediction[torch.arange(target_prediction.size(0)), label_ids].unsqueeze(-1) <= 0, dim=1)
        ACC =(torch.sum(predict_result) / predict_result.size(0)).item()
        # print(f"ACC: {ACC}")

        return {"ACC": round(ACC, 4) }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # print(trainer.evaluate())
        
    trainer.train()
    score = trainer.evaluate()['eval_ACC']

    # logging
    save_results_file = os.path.join(args.output_dir, 'results.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'seed', 'acc'])
        csvwriter.writerow([args.dataset,
                            args.llm_dir,
                            args.n_train_shot,
                            args.seed,
                            score,])


    # model.save_pretrained(args.output_dir)




    


if __name__ == "__main__":
    main()
