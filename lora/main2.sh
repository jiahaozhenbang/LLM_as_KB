export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-large
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
# for DATASET in sst2; do #subj

# for N_TRAIN_SHOT in 32; do
# for SEED in 1; do
# python3 lora/main.py \
#     --llm_dir ${LLM_DIR} \
#     --dataset ${DATASET} \
#     --data_dir ${DATA_DIR} \
#     --n_train_shot ${N_TRAIN_SHOT} \
#     --seed ${SEED} \
#     --output_dir ./output/lora/${LLM} 
# done
# done

# done

for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do #subj

for N_TRAIN_SHOT in 32 64 128; do
for SEED in 1 2 3 4 5; do
python3 lora/main2.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${N_TRAIN_SHOT} \
    --seed ${SEED} \
    --output_dir ./output/lora_constant_lr/${LLM} 
done
done

done
# nohup bash lora/main.sh 2>&1 >lora.log &