import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('..')
from reranking.rerank.base import Query, Text
from reranking.rerank.transformer import T5Reranker
from transformers import (
    AutoTokenizer
)
# import jsonlines
# os.environ['CUDA_VISIBLE_DEVICES']="1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='unicamp-dl/mt5-base-multi-msmarco',
                        type=str, required=False,
                        help="Reranker model.")
    parser.add_argument("--retrieval_result_file", default=None,
                        help="JSON file called `retrieval_results.json`")
    parser.add_argument("--query_text_file", default='nl.txt')
    parser.add_argument("--corpus_text_file", default='man.txt',
                        help="Real corpus to encode in retrieval")
    parser.add_argument("--output_file", default='reranked_results.json',
                        help="output file name for final evaluation")
    parser.add_argument("--top_k", type=int, default=10, help='Using Top K')

    args = parser.parse_args()
    args.query_id_file = args.query_text_file.replace(".txt", ".id")
    args.corpus_id_file = args.corpus_text_file.replace(".txt", ".id")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5Reranker(
        pretrained_model_name_or_path=args.model_name_or_path,
        token_false='Ġfalse', token_true='Ġtrue')

    with open(args.retrieval_result_file, 'r') as f:
        retrieval_results = json.load(f)

    query_map = {}
    with open(args.query_id_file, 'r') as f1, \
            open(args.query_text_file, 'r') as f2:
        for qid, qtext in zip(f1.readlines(), f2.readlines()):
            query_map[qid.strip()] = qtext.strip()

    corpus_id, corpus_text = [], []
    with open(args.corpus_id_file, 'r') as f1, \
            open(args.corpus_text_file, 'r') as f2:
        for cid, ctext in zip(f1.readlines(), f2.readlines()):
            #             corpus_map[cid.strip()] = ctext.strip()
            corpus_id.append(cid.strip())
            corpus_text.append(ctext.strip())

    reranked_results = {}
    print(f'Top_k: {args.top_k}')
    for key, value in tqdm(retrieval_results.items()):
        question_id = key
        result = {'retrieved': [], 'score': [], 'retrieved_index': []}
        query = Query(query_map[question_id])
        texts = [Text(corpus_text[rid], {
                      'cid': corpus_id[rid], 'rid': rid}, 0) for rid in value['retrieved_index'][:args.top_k]]
        reranked = model.rerank(query, texts)
        for rank, document in enumerate(reranked):
            #             print(rank, document.text, document.metadata['cid'], document.score)
            result['retrieved'].append(document.metadata['cid'])
            result['score'].append(document.score)
            result['retrieved_index'].append(document.metadata['rid'])
        reranked_results[question_id] = result
#         break

    with open(args.output_file, 'w') as f:
        json.dump(reranked_results, f)


if __name__ == '__main__':
    main()
