export CUDA_VISIBLE_DEVICES=4

LLM=gpt2-large
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
# N_DEMO_SHOT=1
# SEED=1

# python3 hs_as_feature_with_demo.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} \
#         --output_dir ./output/hs_as_feature_with_demo/${LLM} \
#         --feature_choose all

for DATASET in mpqa; do
# DATASET=dbpedia


N_DEMO_SHOT=1
for N_TRAIN_SHOT in 8 32 128; do
    for SEED in  1 2 3 4 5; do
    python3 get_embedding_from_openai.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ./output/openai_rebuttal \
        --feature_choose all
    done
done


done
