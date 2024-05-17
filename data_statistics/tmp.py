import pandas as pd 
import numpy as np

# MODEL='gpt2-large'
# DATASETS = ['sst2', 'subj', 'mpqa', 'agnews', 'cb', 'cr', 'dbpedia', 'mr', 'rte', 'trec']
# MAX_ICL_SHOTS = [16, 8, 32, 2, 2, 8, 1, 8, 4, 8]

# method_set = ['Logistic Regression'] #'MLP', 
# # KB results 
# result = f'./output/hs_as_feature_with_demo/{MODEL}/results_hs_as_feature.csv'
# result_dfs = pd.read_csv(result)

# for train_shot in [32,64]:
#     avg_dicts = []
#     std_dicts = []
#     avg_running_time = []
#     for dataset in DATASETS:
#         df = result_dfs[result_dfs['dataset']==dataset]
#         # method_set = df['method'].unique().tolist()
#         n_train_shot_set = df['n_train_shot'].unique().tolist()
#         seed_set = df['seed'].unique().tolist()

#         avg_dict = {}
#         std_dict = {}
#         running_time_dict = {}
#         for method in method_set:
#             avg_dict[method] = []
#             std_dict[method] = []
#             running_time_dict[method] = []
#             for n_train_shot in n_train_shot_set:
#                 if n_train_shot == train_shot:
#                     choosed_acc = df[(df['method'] == method) & (df['n_train_shot'] == n_train_shot)]['acc']
#                     avg_dict[method].append(choosed_acc.mean())
#                     std_dict[method].append(choosed_acc.std())
#                     choosed_running_time = df[(df['method'] == method) & (df['n_train_shot'] == n_train_shot)]['running_time']
#                     running_time_dict[method].append(choosed_running_time.mean())
#         avg_dicts.append(avg_dict)
#         std_dicts.append(std_dict)
#         print("S-ICL", dataset)
#         print(avg_dict)
#         print(std_dict)
#         avg_running_time.append(running_time_dict)
#     for i, shot in enumerate([train_shot]):
#         print(shot)
#         _avg_list = [_avg_dict[method_set[0]][i] for _avg_dict in avg_dicts]
#         _std_list = [_std_dict[method_set[0]][i] for _std_dict in std_dicts]
#         _ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(_avg_list, _std_list)]
#         output_str = "&".join(_ele_list)
#         print(output_str)
#         print(np.mean(_avg_list), np.mean(_std_list))


MODEL='gpt2-large'
DATASETS = ['sst2', 'subj', 'mpqa', 'agnews', 'cb', 'cr', 'dbpedia', 'mr', 'rte', 'trec']

# KB results 
prompt_results = f'./output/lora_constant_lr/{MODEL}/results.csv' # f'./output/lora/{MODEL}/results.csv'
prompt_dfs = pd.read_csv(prompt_results)

for train_shot in [32,64,128]:
    prompt_avgs = []
    prompt_stds = []
    for index, dataset in enumerate(DATASETS):
        prompt_df = prompt_dfs[prompt_dfs['dataset'] == dataset]
        assert len(prompt_df['n_train_shot'].unique().tolist()) == 3

        prompt_avg = []
        prompt_std = []
        for n_train_shot in [train_shot]:
            choosed_acc = prompt_df[prompt_df['n_train_shot'] == n_train_shot]['acc']
            prompt_avg.append(choosed_acc.mean())
            prompt_std.append(choosed_acc.std())

        prompt_avgs.append(prompt_avg[0])
        prompt_stds.append(prompt_std[0])

    print("lora", train_shot)
    _ele_list = [str(round(_avg * 100, 1)) + r"$_{\pm\textrm" + str(round(_std * 100, 1)) + r"}$" for _avg, _std in zip(prompt_avgs, prompt_stds)]
    output_str = "&".join(_ele_list)
    print(output_str)
    print(np.mean(prompt_avgs), np.mean(prompt_stds))
