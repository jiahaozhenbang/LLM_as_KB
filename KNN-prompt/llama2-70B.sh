export CUDA_VISIBLE_DEVICES=0

LLM=llama2-70B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/

# Set max demonstration shot w.r.t. context length


for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do
# DATASET=sst2


for N_TRAIN_SHOT in 128; do
for SEED in 1 2 3 4 5; do
N_DEMO_SHOT=1
python3 KNN-prompt/main.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ./output/knn-prompt/${LLM} 

done
done

done
