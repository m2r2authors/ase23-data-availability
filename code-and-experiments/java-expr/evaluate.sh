export CUDA_VISIBLE_DEVICES=1

cd ..

MODEL_NAME=java-expr/save/$1

DATA_DIR=java-expr/data/
TEST_DIR=$2/

# python retriever/simcse/run_uni_inference.py \
python retriever/simcse/run_uni_inference.py \
  --model_name $MODEL_NAME \
  --source_file ${DATA_DIR}${TEST_DIR}test_nl.txt \
  --target_file ${DATA_DIR}${TEST_DIR}man.tok.txt \
  --source_embed_save_file ${DATA_DIR}${TEST_DIR}.tmp/src_embedding \
  --target_embed_save_file ${DATA_DIR}${TEST_DIR}.tmp/tgt_embedding \
  --oracle_eval_file ${DATA_DIR}${TEST_DIR}cmd_test.oracle_man.full.json \
  --sim_func cls_distance.cosine \
  --num_layers 12 \
  --save_file ${DATA_DIR}${TEST_DIR}retrieval_results.json \
  --normalize_embed