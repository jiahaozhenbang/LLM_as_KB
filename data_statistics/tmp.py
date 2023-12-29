import pandas as pd 
import numpy as np

MODEL='llama2-7B'
DATASETS = ['sst2', 'subj', 'mpqa', 'agnews', 'cb', 'cr', 'dbpedia', 'mr', 'rte', 'trec']
train_shot = 128
method_set = ['Logistic Regression'] #'MLP', 
# KB results 
result = f'./output/hs_as_feature_with_demo/{MODEL}/results_hs_as_feature.csv'
result_dfs = pd.read_csv(result)


avg_dicts = []
std_dicts = []
avg_running_time = []
for dataset in DATASETS:
    df = result_dfs[result_dfs['dataset']==dataset]
    # method_set = df['method'].unique().tolist()
    n_train_shot_set = df['n_train_shot'].unique().tolist()
    seed_set = df['seed'].unique().tolist()

    assert n_train_shot_set == [train_shot]

    avg_dict = {}
    std_dict = {}
    running_time_dict = {}
    for method in method_set:
        avg_dict[method] = []
        std_dict[method] = []
        running_time_dict[method] = []
        for n_train_shot in n_train_shot_set:
            choosed_acc = df[(df['method'] == method) & (df['n_train_shot'] == n_train_shot)]['acc']
            avg_dict[method].append(choosed_acc.mean())
            std_dict[method].append(choosed_acc.std())
            choosed_running_time = df[(df['method'] == method) & (df['n_train_shot'] == n_train_shot)]['running_time']
            running_time_dict[method].append(choosed_running_time.mean())
    avg_dicts.append(avg_dict)
    std_dicts.append(std_dict)
    print("S-ICL", dataset)
    print(avg_dict)
    print(std_dict)
    avg_running_time.append(running_time_dict)
for i, shot in enumerate([train_shot]):
    print(shot)
    _avg_list = [_avg_dict[method_set[0]][i] for _avg_dict in avg_dicts]
    _std_list = [_std_dict[method_set[0]][i] for _std_dict in std_dicts]
    _ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(_avg_list, _std_list)]
    output_str = "&".join(_ele_list)
    print(output_str)
    print(np.mean(_avg_list), np.mean(_std_list))


# icl results
icl_results = [f'./output/icl/{MODEL}/results_icl_{dataset}.csv' for dataset in DATASETS]
icl_dfs = [pd.read_csv(icl_result) for icl_result in icl_results]

icl_avgs = []
icl_stds = []
for index, icl_df in enumerate(icl_dfs):
    icl_n_train_shot_set = icl_df['n_train_shot'].unique().tolist()
    print(icl_n_train_shot_set)
    assert len(icl_n_train_shot_set) == 1

    icl_avg = []
    icl_std = []
    for n_train_shot in icl_n_train_shot_set:
        choosed_acc = icl_df[icl_df['n_train_shot'] == n_train_shot]['acc']
        icl_avg.append(choosed_acc.mean())
        icl_std.append(choosed_acc.std())
        # print(choosed_acc)
    
    icl_avgs.append(icl_avg[0])
    icl_stds.append(icl_std[0])


_ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(icl_avgs, icl_stds)]
output_str = "&".join(_ele_list)
print(output_str)
print(np.mean(icl_avgs), np.mean(icl_stds))

# KNN results
knn_results = f'./output/knn_prompting/{MODEL}/results_knnprompting.csv'
knn_dfs = pd.read_csv(knn_results)

knn_avgs = []
knn_stds = []
for index, dataset in enumerate(DATASETS):
    knn_df = knn_dfs[knn_dfs['dataset'] == dataset]
    assert len(knn_df['n_train_shot'].unique().tolist()) == 1

    knn_avg = []
    knn_std = []
    for n_train_shot in [train_shot]:
        choosed_acc = knn_df[knn_df['n_train_shot'] == n_train_shot]['acc']
        knn_avg.append(choosed_acc.mean())
        knn_std.append(choosed_acc.std())

    knn_avgs.append(knn_avg[0])
    knn_stds.append(knn_std[0])

print("knn-prompting")
_ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(knn_avgs, knn_stds)]
output_str = "&".join(_ele_list)
print(output_str)
print(np.mean(knn_avgs), np.mean(knn_stds))




# knn-prompt results
prompt_results = f'./output/knn-prompt/{MODEL}/results_knnprompt.csv'
prompt_dfs = pd.read_csv(prompt_results)

prompt_avgs = []
prompt_stds = []
for index, dataset in enumerate(DATASETS):
    prompt_df = prompt_dfs[prompt_dfs['dataset'] == dataset]
    assert len(prompt_df['n_train_shot'].unique().tolist()) == 1

    prompt_avg = []
    prompt_std = []
    for n_train_shot in [train_shot]:
        choosed_acc = prompt_df[prompt_df['n_train_shot'] == n_train_shot]['acc']
        prompt_avg.append(choosed_acc.mean())
        prompt_std.append(choosed_acc.std())

    prompt_avgs.append(prompt_avg[0])
    prompt_stds.append(prompt_std[0])

print("knn-prompt")
_ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(prompt_avgs, prompt_stds)]
output_str = "&".join(_ele_list)
print(output_str)
print(np.mean(prompt_avgs), np.mean(prompt_stds))