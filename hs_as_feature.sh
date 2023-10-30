export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/

# Set max demonstration shot w.r.t. context length
if [[ "${LLM}" == "gpt2-xl" ]] || [[ "{$LLM}" == "gpt2-large" ]]; then
# max context length = 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr trec) # maxshot = 8
array4=(rte) # maxshot = 4
array5=(agnews cb) # maxshot = 2
array6=(dbpedia) # maxshot = 1
else
# max context length = 2048
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)
fi

# DATASET=sst2
# N_TRAIN_SHOT=4
# SEED=1

# python3 hidden_states_as_feature.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --seed ${SEED} \
#         --output_dir ./output/hs_as_feature/${LLM} \
#         --feature_choose all

for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do
# DATASET=dbpedia

if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=32
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=16
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=8
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=4
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=2
else
N_DEMO_SHOT=1
fi

# N_TRAIN_SHOT=32
# SEED=1
for N_TRAIN_SHOT in 4 8 16 32 64 128 256 512 1024; do
    for SEED in 1 2 3 4 5; do
    python3 hidden_states_as_feature.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --seed ${SEED} \
        --output_dir ./output/hs_as_feature/${LLM} \
        --feature_choose all
    done
done

done
# nohup bash hs_as_feature.sh 2>&1 >output/hs_as_feature/hs_as_feature.log &