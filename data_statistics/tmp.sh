export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/


# DATASET=dbpedia
# N_TRAIN_SHOT=2
# N_DEMO_SHOT=1
# SEED=1

# python3 data_statistics/get_hs.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} 
    

# N_DEMO_SHOT=0

# python3 data_statistics/get_hs.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} 

# for DATASET in sst2 dbpedia;
# do
# N_TRAIN_SHOT=2
# N_DEMO_SHOT=1
# SEED=1

# python3 data_statistics/surface_form_competetion.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} 
    

# N_DEMO_SHOT=0

# python3 data_statistics/surface_form_competetion.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} 
# done

DATASET=dbpedia
N_TRAIN_SHOT=128
N_DEMO_SHOT=1
SEED=1

python3 data_statistics/get_prob_for_adaptation.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} 
    
