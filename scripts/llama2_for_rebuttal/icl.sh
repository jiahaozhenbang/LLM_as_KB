export CUDA_VISIBLE_DEVICES=2

LLM=llama2-7B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
BATCHSIZE=1



for DATASET in  trec; do # sst2 mpqa


echo ${DATASET}
for N_TRAIN_SHOT in 8 32; do
for SEED in 1 2 3 4 5; do

python3 icl.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${N_TRAIN_SHOT} \
    --seed ${SEED} \
    --output_dir ./output/icl/${LLM}

done
done
done
# nohup bash scripts/across_LM_size/icl/llama-7B.sh 2>&1 >output/icl/llama-7B.log &