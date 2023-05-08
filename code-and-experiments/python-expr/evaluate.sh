export CUDA_VISIBLE_DEVICES=0

cd ../
stage=$1

# MODEL_NAME=Salesforce/codet5-base
MODEL_NAME=python-expr/save/$stage/checkpoint-best

DATA_DIR=python-expr/data/
TEST_DIR=python_test_related/

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