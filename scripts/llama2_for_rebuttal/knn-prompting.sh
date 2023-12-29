export CUDA_VISIBLE_DEVICES=2

LLM=llama2-7B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/

for DATASET in trec ; do

KNN=3

for N_TRAIN_SHOT in 512; do #8 32 128
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

# nohup bash scripts/across_LM_size/knn-prompting/llama-7B.sh 2>&1 >output/knn_prompting/llama-7B.log &