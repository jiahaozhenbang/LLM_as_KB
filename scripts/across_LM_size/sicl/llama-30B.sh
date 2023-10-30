export CUDA_VISIBLE_DEVICES=0,1

LLM=llama-30B
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


for DATASET in sst2 subj mpqa agnews cb cr dbpedia  mr rte trec; do #sst2 subj mpqa agnews cb cr dbpedia 


N_DEMO_SHOT=1
for N_TRAIN_SHOT in 128; do
    python3 saicl-multi-gpu.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --output_dir ./output/hs_as_feature_with_demo/${LLM} \
        --feature_choose all
done


done
# nohup bash scripts/across_LM_size/sicl/llama-30B.sh 2>&1 >>scripts/across_LM_size/sicl/llama-30B.log &