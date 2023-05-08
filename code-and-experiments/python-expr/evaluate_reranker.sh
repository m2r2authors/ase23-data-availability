top_k=$1
DATA_DIR=data/python_test_related/

model_name_or_path=save/reranker_retrieved_python_test_related_20/checkpoint-1200

python evaluate_reranker.py  --top_k ${top_k}\
    --model_name_or_path ${model_name_or_path} \
    --retrieval_result_file ${DATA_DIR}retrieval_results.json \
    --query_text_file ${DATA_DIR}test_nl.txt \
    --corpus_text_file ${DATA_DIR}man.tok.txt \
    --output_file ${DATA_DIR}reranked_results_${top_k}.json \