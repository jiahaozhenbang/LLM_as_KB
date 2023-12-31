import os
import json

class BASEDataset:
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__()
        if mode == 'dev':
            mode = 'dev_subsample'
        data_file = os.path.join(data_dir, mode + '.jsonl')
        self.data = []
        with open(data_file, 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                instance = json.loads(line.strip())
                self.data.append(instance)
        # customize your own label map in inheritance
        self.id2label = {0: 'negative', 1: 'positive'}
        self.label2id = {'negative': 0, 'positive': 1}

def make_prompt(dataset, dataset_name, mode, indices=None):
    if dataset_name == 'sst2':
        template_func = template_sst2
    elif dataset_name == 'subj':
        template_func = template_subj
    elif dataset_name == 'agnews':
        template_func = template_agnews
    elif dataset_name == 'cb':
        template_func = template_cb
    elif dataset_name == 'cr':
        template_func = template_cr
    elif dataset_name == 'dbpedia':
        template_func = template_dbpedia
    elif dataset_name == 'mpqa':
        template_func = template_mpqa
    elif dataset_name == 'mr':
        template_func = template_mr
    elif dataset_name == 'rte':
        template_func = template_rte
    elif dataset_name == 'sst5':
        template_func = template_sst5
    elif dataset_name == 'trec':
        template_func = template_trec
    if mode == 'inference':
        return template_func(dataset, None, mode)
    prompt = ''
    if mode == 'compose':  # inputs are different, list of examples instead of dataset class
        for ins in dataset.index(indices):
            prompt += template_func(ins, dataset.label2verb[ins['label']], 'train')
            prompt += '\n'
        return prompt
    for ins in dataset.data:
        prompt += template_func(ins, dataset.label2verb[ins['label']], mode)
        prompt += '\n'
    return prompt


def template_sst2(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    else:
        return f"Review: {ins['sentence']}\nSentiment:"


def template_subj(ins, label, mode):
    if mode == 'train':
        return f"Input: {ins['sentence']}\nType: {label}\n"
    else:
        return f"Input: {ins['sentence']}\nType:"


def template_agnews(ins, label, mode):
    if mode == 'train':
        return f"input: {ins['sentence']}\ntype: {label}\n"
    else:
        return f"input: {ins['sentence']}\ntype:"


def template_cb(ins, label, mode):
    if mode == 'train':
        return f"premise: {ins['premise']}\nhypothesis: {ins['hypothesis']}\nprediction: {label}\n"
    else:
        return f"premise: {ins['premise']}\nhypothesis: {ins['hypothesis']}\nprediction:"


def template_cr(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    else:
        return f"Review: {ins['sentence']}\nSentiment:"


def template_dbpedia(ins, label, mode):
    if mode == 'train':
        return f"input: {ins['sentence']}\ntype: {label}\n"
    else:
        return f"input: {ins['sentence']}\ntype:"


def template_mpqa(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    else:
        return f"Review: {ins['sentence']}\nSentiment:"


def template_mr(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    else:
        return f"Review: {ins['sentence']}\nSentiment:"


def template_rte(ins, label, mode):
    if mode == 'train':
        return f"premise: {ins['sentence_1']}\nhypothesis: {ins['sentence_2']}\nprediction: {label}\n"
    else:
        return f"premise: {ins['sentence_1']}\nhypothesis: {ins['sentence_2']}\nprediction:"


def template_sst5(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    else:
        return f"Review: {ins['sentence']}\nSentiment:"


def template_trec(ins, label, mode):
    if mode == 'train':
        return f"Question: {ins['sentence']}\nType: {label}\n"
    else:
        return f"Question: {ins['sentence']}\nType:"


def sent_sim_template(ins, dataset_name):
    # ['sst2', 'subj', 'mpqa', 'agnews', 'cb', 'cr', 'dbpedia', 'mr', 'rte', 'trec']
    if dataset_name in ['sst2', 'subj', 'mpqa', 'agnews', 'cr', 'dbpedia', 'mr', 'trec']:
        return ins['sentence']
    elif dataset_name in ['cb']:
        return f"premise: {ins['premise']}. hypothesis: {ins['hypothesis']}"
    elif dataset_name in ['rte']:
        return f"premise: {ins['sentence_1']}. hypothesis: {ins['sentence_2']}"

for name in  "sst2 subj mpqa agnews cb cr dbpedia mr rte trec".split():
    if os.path.exists(os.path.join(os.path.join("./", "data", name), "train" + '.jsonl')):
        dataset = BASEDataset(os.path.join("./", "data", name), "train")
    else:
        dataset = BASEDataset(os.path.join("./", "data", name), "train_subset")


    sents_len = [len(sent_sim_template(data, name)) for data in dataset.data]

    avg_len = sum(sents_len) / len(sents_len)

    print(name, avg_len)
