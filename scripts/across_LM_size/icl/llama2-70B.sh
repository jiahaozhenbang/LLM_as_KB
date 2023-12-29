export CUDA_VISIBLE_DEVICES=0

LLM=llama2-70B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
BATCHSIZE=1

# Set maxshot w.r.t. context length

# max context length = 4096
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)


# for DATASET in   sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do # sst2 mpqa

# if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
# NSHOT=64
# elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
# NSHOT=32
# elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
# NSHOT=16
# elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
# NSHOT=8
# elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
# NSHOT=4
# else
# NSHOT=1
# fi


# echo ${DATASET}

# for SEED in 1 2 3 4 5; do

# python3 icl.py \
#     --llm_dir ${LLM_DIR} \
#     --dataset ${DATASET} \
#     --data_dir ${DATA_DIR} \
#     --n_train_shot ${NSHOT} \
#     --seed ${SEED} \
#     --output_dir ./output/icl/${LLM}

# done

# done

for DATASET in  rte; do # sst2 mpqa

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


echo ${DATASET}

for SEED in  4 ; do

python3 icl.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${NSHOT} \
    --seed ${SEED} \
    --output_dir ./output/icl/${LLM}

done

done

for DATASET in   trec; do # sst2 mpqa

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


echo ${DATASET}

for SEED in  2 3 4 5; do

python3 icl.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${NSHOT} \
    --seed ${SEED} \
    --output_dir ./output/icl/${LLM}

done

done
# nohup bash scripts/across_LM_size/icl/llama2-70B.sh 2>&1 >output/icl/llama2-70B.log &