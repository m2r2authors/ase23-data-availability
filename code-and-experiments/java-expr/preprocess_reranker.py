import argparse
import json
import os
import pandas as pd
import random


def dump_reranker_train(args, top_k=50, eval_prop=0.1, mode='retrieved-only'):
    "Consider balancing problem"

    with open(os.path.join(args.data_dir, 'cmd_train.oracle_man.full.json'), 'r') as f:
        oracle_man_dict = json.load(f)
        gold = [item['oracle_man'] for item in oracle_man_dict]
        queries = [item['nl'] for item in oracle_man_dict]

    with open(os.path.join(args.data_dir, 'train_retrieval_results.json'), 'r') as f:
        results = json.load(f)
        for k, v in results.items():
            print(v.keys())
            break
        pred = [results[x['question_id']]['retrieved'][:top_k]
                for x in oracle_man_dict]
        pred_indexes = [results[x['question_id']]['retrieved_index'][:top_k]
                for x in oracle_man_dict]
        print(len(pred))
    with open(os.path.join(args.data_dir, 'man.tok.txt'), 'r') as f:
        corpus = [line.strip() for line in f.readlines()]

    reranker_train = []
    reranker_eval = []

    if mode == 'retrieved-only-api':  # first step, mimic `eval_retrieval_from_file`
        for query, oracle_man, pred_man in zip(queries, gold, pred):
            if random.random() < eval_prop:
                for x in pred_man:
                    if x in oracle_man:
                        reranker_eval.append((query, x, 'true'))
                    else:
                        reranker_eval.append((query, x, 'false'))
            else:
                for x in pred_man:
                    if x in oracle_man:
                        reranker_train.append((query, x, 'true'))
                    else:
                        reranker_train.append((query, x, 'false'))
        print(reranker_train[0])
        print('total:', len(reranker_train), ', positive:', sum([x[2] == 'true' for x in reranker_train]),
              ', negative:', sum([x[2] == 'false' for x in reranker_train]))
    
    elif mode == 'retrieved-only-apinl':
        for query, oracle_man, pred_man, pred_idx in zip(queries, gold, pred, pred_indexes):
            if random.random() < eval_prop:
                for x, i in zip(pred_man, pred_idx):
                    if query in corpus[i]:
                        continue
                    if x in oracle_man:
                        reranker_eval.append((query, corpus[i], 'true'))
                    else:
                        reranker_eval.append((query, corpus[i], 'false'))
            else:
                for x, i in zip(pred_man, pred_idx):
                    if query in corpus[i]:
                        continue
                    if x in oracle_man:
                        reranker_train.append((query, corpus[i], 'true'))
                    else:
                        reranker_train.append((query, corpus[i], 'false'))
        print(reranker_train[0])
        print('total:', len(reranker_train), ', positive:', sum([x[2] == 'true' for x in reranker_train]),
              ', negative:', sum([x[2] == 'false' for x in reranker_train]))
    elif mode == 'balanced-apinl':  # TODO:
        for query, oracle_man, pred_man, pred_idx in zip(queries, gold, pred, pred_indexes):
            if random.random() < eval_prop:
                for x, i in zip(pred_man, pred_idx):
                    if query in corpus[i]:
                        continue
                    if x in oracle_man:
                        reranker_eval.append((query, corpus[i], 'true'))
                    else:
                        reranker_eval.append((query, corpus[i], 'false'))
            else:
                for x, i in zip(pred_man, pred_idx):
                    if query in corpus[i]:
                        continue
                    if x in oracle_man:
                        reranker_train.append((query, corpus[i], 'true'))
                    else:
                        reranker_train.append((query, corpus[i], 'false'))
        print(reranker_train[0])
        print('total:', len(reranker_train), ', positive:', sum([x[2] == 'true' for x in reranker_train]),
              ', negative:', sum([x[2] == 'false' for x in reranker_train]))
    elif mode == 'all-prov':   # second step
        pass
    else:
        raise NotImplementedError

    with open(os.path.join(args.data_dir, f'reranker_train_top_{top_k}.json'), 'w') as f:
        json.dump(reranker_train, f, indent=2)
    with open(os.path.join(args.data_dir, f'reranker_eval_top_{top_k}.json'), 'w') as f:
        json.dump(reranker_eval, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

#     dump_reranker_train(top_k=args.top_k, mode='retrieved-only-api')
    dump_reranker_train(args, top_k=args.top_k, mode='retrieved-only-apinl')
