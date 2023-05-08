# first run inference
export CUDA_VISIBLE_DEVICES=$1

cd ../

MODEL_NAME=java-expr/save/$2

DATA_DIR=java-expr/data/$3

python retriever/simcse/run_uni_inference.py \
  --model_name $MODEL_NAME \
  --source_file ${DATA_DIR}train_nl.txt \
  --target_file ${DATA_DIR}man.tok.txt \
  --source_embed_save_file ${DATA_DIR}.tmp/src_embedding \
  --target_embed_save_file ${DATA_DIR}.tmp/tgt_embedding \
  --oracle_eval_file ${DATA_DIR}cmd_train.oracle_man.full.json \
  --sim_func cls_distance.cosine \
  --num_layers 12 \
  --save_file ${DATA_DIR}train_retrieval_results.json \
  --normalize_embed

# then run preprocess