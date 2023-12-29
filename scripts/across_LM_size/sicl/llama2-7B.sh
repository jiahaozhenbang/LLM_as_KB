export CUDA_VISIBLE_DEVICES=1

LLM=llama2-7B
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/


for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec ; do


N_DEMO_SHOT=1
for N_TRAIN_SHOT in 128; do #8 32 128
    for SEED in 1 2 3 4 5; do
    python3 hs_as_feature_with_demo.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ./output/hs_as_feature_with_demo/${LLM} \
        --feature_choose all
    done
done

done
