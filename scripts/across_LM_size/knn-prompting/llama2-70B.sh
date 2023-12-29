export CUDA_VISIBLE_DEVICES=0

LLM=llama2-70B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/

# Set maxshot w.r.t. context length

# max context length = 4096
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)


for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec ; do

if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
NSHOT=64
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
NSHOT=32
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
NSHOT=16
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
NSHOT=8
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
NSHOT=4
else
NSHOT=1
fi


KNN=3

for N_TRAIN_SHOT in 128; do #8 32 128
for SEED in 1 2 3 4 5; do
N_DEMO_SHOT=1
python3 knn_prompting.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${N_TRAIN_SHOT} \
    --n_demo_shot ${N_DEMO_SHOT} \
    --seed ${SEED} \
    --output_dir ./output/knn_prompting/${LLM} \
    --knn ${KNN}

done
done

done

# nohup bash scripts/across_LM_size/knn-prompting/llama2-70B.sh 2>&1 >output/knn_prompting/llama2-70B.log &