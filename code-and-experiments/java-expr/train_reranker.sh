wandb disabled

export CUDA_VISIBLE_DEVICES=$3

top_k=$1
if [ -z "$top_k" ]
then
    top_k=10
fi

DATA_DIR=data/$2/

python train_reranker.py \
    --base_model Salesforce/codet5-base \
    --data_dir $DATA_DIR \
    --output_model_path save/reranker_retrieved_$2_${top_k} \
    --save_total_limit 5 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 8 \
    --epochs 2 \
    --save_every_n_steps 100 \
    --learning_rate 3e-4 \
    --max_length 200 \
    --top_k $top_k \

file_name=$(basename $0)

wandb enabled