import argparse
import csv
import json
import os
import sys
import time


def pretty_result(eval_result):
    try:
        import prettytable
    except:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "prettytable"])
    finally:
        import prettytable

    recall = eval_result[0]['recall']
    precision = eval_result[0]['precision']
    MRR = eval_result[1]['MRR']
    MAP = eval_result[1]['MAP']

    recpre = prettytable.PrettyTable()
    recpre.field_names = ['K', 'Recall@K', 'Precision@K']

    for k in recall.keys():
        recpre.add_row([k, recall[k], precision[k]])

    recpre.align['K'] = 'r'
    recpre.align['Recall@K'] = 'l'
    recpre.align['Precision@K'] = 'l'

    mmrmap = prettytable.PrettyTable()
    mmrmap.field_names = ['Metric', 'Value']
    mmrmap.add_row(['MRR', MRR])
    mmrmap.add_row(['MAP', MAP])
    mmrmap.align['Value'] = 'l'

    print(recpre)
    print(mmrmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument("--data_dir", default='.', type=str)
    parser.add_argument("--oracle_file", default='cmd_test.oracle_man.full.json')
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, f'reranked_results_{args.top_k}.json'), 'r') as f:
        retrieval_results = json.load(f)

    dedup_results = {}
    for key, value in retrieval_results.items():
        dedup_retrieved = []
        dedup_scores = []

        for i, r in enumerate(value['retrieved']):
            if r not in dedup_retrieved:
                dedup_retrieved.append(r)
                dedup_scores.append(value['score'][i])

        while len(dedup_retrieved) < args.top_k:
            dedup_retrieved.append('NONE')
            dedup_scores.append(0.0)

        dedup_results[key] = {
            'retrieved': dedup_retrieved, 'score': dedup_scores}

    with open(os.path.join(args.data_dir, f'dedup_reranked_results_{args.top_k}.json'), 'w') as f:
        json.dump(dedup_results, f, indent=2)

    sys.path.append('../evaluator/')
    from eval_ar import eval_retrieval_from_file

    print(f'\n{"="*10} Reranked results {"="*10}\n')
    reranked_result = eval_retrieval_from_file(
        os.path.join(args.data_dir, args.oracle_file),
        os.path.join(args.data_dir, f'reranked_results_{args.top_k}.json'))

    pretty_result(reranked_result)

    print(f'\n{"="*10} Dedup reranked results {"="*10}\n')
    dedup_reranked_result = eval_retrieval_from_file(
        os.path.join(args.data_dir, args.oracle_file),
        os.path.join(args.data_dir, f'dedup_reranked_results_{args.top_k}.json'))

    pretty_result(dedup_reranked_result)

    # write to csv
    # include: time, top_k, rerank mrr, rerank map, recall@k, precision@k
    with open(f'reranked_result_score.csv', 'a') as f:
        writer = csv.writer(f)
        time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # If the file is empty, write the header
        if f.tell() == 0:
            writer.writerow(['time', 'mode', 'top_k', 'rerank_mrr',
                            'rerank_map', 'recall@k', 'precision@k'])
        writer.writerow(
            [
                time,
                'rerank',
                args.top_k,
                reranked_result[1]['MRR'],
                reranked_result[1]['MAP'],
                reranked_result[0]['recall'],
                reranked_result[0]['precision']
            ]
        )
        writer.writerow(
            [
                time,
                'dedup_rerank',
                args.top_k,
                dedup_reranked_result[1]['MRR'],
                dedup_reranked_result[1]['MAP'],
                dedup_reranked_result[0]['recall'],
                dedup_reranked_result[0]['precision']
            ]
        )
