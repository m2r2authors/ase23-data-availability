export CUDA_VISIBLE_DEVICES=$1

DATA_DIR=java-expr/data/$2

# MODEL_NAME=google/flan-t5-base
MODEL_NAME=Salesforce/codet5-base
TRAIN_FILE=$3
OUTPUT_DIR=$4
RUN_NAME=$5
EVAL_STEPS=50
NUM_TRAIN_EPOCHS=5

cd ../

python retriever/simcse/run_train.py \
    --num_layers 12 \
    --model_name_or_path ${MODEL_NAME} \
    --sim_func cls_distance.cosine \
    --temp 0.05  \
    --train_file ${DATA_DIR}${TRAIN_FILE} \
    --eval_file None \
    --output_dir $OUTPUT_DIR \
    --eval_src_file ${DATA_DIR}valid_nl.txt \
    --eval_tgt_file ${DATA_DIR}man.tok.txt \
    --eval_root_folder ${DATA_DIR} \
    --eval_oracle_file cmd_valid.oracle_man.full.json \
    --run_name ${RUN_NAME} \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 512 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model recall@10 \
    --load_best_model_at_end \
    --eval_steps $EVAL_STEPS \
    --save_steps $EVAL_STEPS \
    --save_total_limit 5 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --eval_form retrieval
    # "$@"