import json
import copy
from collections import OrderedDict

import numpy as np
import argparse

TOP_K = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]


def align_src_pred(src_file, pred_file):
    with open(src_file, "r") as fsrc, open(pred_file, "r") as fpred:
        src = json.load(fsrc)
        pred = json.load(fpred)['results']
        # assert len(src) == len(pred), (len(src), len(pred))

    # re-order src
    src_nl = [x['nl'] for x in src]
    _src = []
    _pred = []
    for p in pred:
        if p['nl'] in src_nl:
            _src.append(src[src_nl.index(p['nl'])])
            _pred.append(p)

    src = _src
    pred = _pred

    for s, p in zip(src, pred):
        assert s['nl'] == p['nl'], (s['nl'], p['nl'])

    print(f"unique nl: {len(set(src_nl))}")
    print(f"number of samples (src/pred): {len(src)}/{len(pred)}")
    print("pass nl matching check")

    return src, pred


def calc_metrics(src_file, pred_file):
    src, pred = align_src_pred(src_file, pred_file)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        oracle_man = get_oracle(s, cmd_name)
        pred_man = p['pred']
        _src.append(oracle_man)
        _pred.append(pred_man)
    calc_recall(_src, _pred)

    # description only
    _src = []
    for s in src:
        _src.append(s['matching_info']['|main|'])
    calc_recall(_src, _pred)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        _src.append(cmd_name)
        _pred.append(pred_man)
    calc_hit(_src, _pred)
    # calc_mean_rank(src, pred)


def calc_mean_rank(src, pred):
    rank = []
    for s, p in zip(src, pred):
        cur_rank = []
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        oracle_man = get_oracle(s, cmd_name)
        for o in oracle_man:
            if o in pred_man:
                cur_rank.append(oracle_man.index(o))
            else:
                cur_rank.append(101)
        if cur_rank:
            rank.append(np.mean(cur_rank))

    print(np.mean(rank))


def calc_hit(src, pred, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    hit_n = {x: 0 for x in top_k}
    assert len(src) == len(pred), (len(src), len(pred))

    for s, p in zip(src, pred):
        cmd_name = s
        pred_man = p

        for tk in hit_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = any([cmd_name in x for x in cur_result_vids])
            hit_n[tk] += cur_hit

    hit_n = {k: v / len(pred) for k, v in hit_n.items()}
    for k in sorted(hit_n.keys()):
        print(f"{hit_n[k] :.3f}", end="\t")
    print()
    return hit_n


def get_oracle(item, cmd_name):
    # oracle = [f"{cmd_name}_{x}" for x in itertools.chain(*item['matching_info'].values())]
    oracle = [f"{cmd_name}_{x}" for x in item['oracle_man']]
    return oracle


def calc_recall(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = sum([x in cur_result_vids for x in oracle_man])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / \
                (len(oracle_man)) if len(oracle_man) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(
                f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}


def reciprocal_rank(src, pred):
    for i, p in enumerate(pred):
        if p in src:
            return 1 / (i + 1)
    return 0.0


def average_precision(src, pred):
    num_hits = 0.0
    sum_precisions = 0.0
    for i, p in enumerate(pred):
        if p in src:
            num_hits += 1.0
            sum_precisions += num_hits / (i + 1.0)
    return sum_precisions / min(len(src), len(pred))


def calc_r(src, pred):
    mean_reciprocal_rank = 0.0
    mean_average_precision = 0.0

    for s, p in zip(src, pred):
        mean_reciprocal_rank += reciprocal_rank(s, p)
        mean_average_precision += average_precision(s, p)

    mean_reciprocal_rank = mean_reciprocal_rank / len(pred)
    mean_average_precision = mean_average_precision / len(pred)

    return {'MRR': mean_reciprocal_rank, 'MAP': mean_average_precision}


def eval_retrieval_from_file(data_file, retrieval_file,
                             oracle_entry='oracle_man', retrieval_entry='retrieved', top_k=None):

    assert 'oracle_man.full' in data_file or 'conala' not in data_file, (
        data_file)
    # for conala
    with open(data_file, "r") as f:
        d = json.load(f)
    gold = [item[oracle_entry] for item in d]

    with open(retrieval_file, "r") as f:
        r_d = json.load(f)
    pred = [r_d[x['question_id']][retrieval_entry] for x in d]
    metrics = calc_recall(gold, pred, top_k=top_k)
    try:
        m_r = calc_r(gold, pred)
    except Exception as e:
        m_r = e
    return metrics, m_r
#     return metrics


def eval_post_retrieval_from_file(data_file, retrieval_file, corpus_file, answer_file, top_k=20, ALL_K=[1,3,5,10], dedup_answer=False):
    """
    1. use `run_uni_inferece.py` to get the index of retrieved corpus
    2. answer_file is the man.tok.answers
    """
    with open(data_file, "r") as f:
        o_d = json.load(f)
    with open(retrieval_file, "r") as f:
        r_d = json.load(f)
    pred_index = [r_d[x['question_id']]['retrieved_index'] for x in o_d]
    with open(corpus_file, "r") as f:
        corpus = [line.strip() for line in f.readlines()]
    with open(answer_file, "r") as f:
        answers = [json.loads(line.strip()) for line in f.readlines()]
    
    results = {
        "recall": {k: 0.0 for k in ALL_K},
        "precision": {k: 0.0 for k in ALL_K},
        "MRR": 0.0,
        "MAP": 0.0
    }
    
    for gold_map, pred_idx_list in zip(o_d, pred_index):
        gold_answers = gold_map['oracle_man']
        query = gold_map['nl']
        
        # dedup (following CLEAR's setting)
        # - deduplicate same title (which means there can be titles with the same semantics
        retrieved_titles = []
        corresponding_answers = []
        for idx in pred_idx_list:
            title = corpus[idx].split(': ')[1]  # assume {API}: {TITLE}
            if title in retrieved_titles:
                continue
            if dedup_answer and answers[idx] in corresponding_answers:
                continue
            retrieved_titles.append(title)
            corresponding_answers.append(answers[idx])
         
        # fill
        while len(retrieved_titles) < top_k:
            retrieved_titles.append("None")
            corresponding_answers.append([])
        
        precision_hits = [0 for _ in range(top_k)]
        recall_hits = [0 for _ in range(top_k)]
        counter = -1
        tmp_mrr = 0.0
        tmp_map = 0.0
        tmp_hits = 0
        tmp_answers = copy.deepcopy(gold_answers)
        # compute metrics
        for i, pred_title, pred_answers in zip(range(top_k), retrieved_titles, corresponding_answers):
            for a in pred_answers:
                if a in gold_answers:
#                     print('precision hit')
                    if not query == pred_title.strip().replace("?",""):
                        precision_hits[i] = 1
                if a in tmp_answers:
#                     print('recall hit')
                    if not query == pred_title.strip().replace("?",""):
                        precision_hits[i] = 1
                        if counter == -1:
                            counter = i + 1
                        tmp_hits += 1
                        recall_hits[i] = 1
                        tmp_map += tmp_hits / (i + 1)
                        tmp_answers.remove(a)
        tmp_map /= len(gold_answers)
        if counter != -1:
            tmp_mrr = 1 / counter
#         print(precision_hits)
#         print(recall_hits)
        
        for k in ALL_K:
            results['precision'][k] += sum(precision_hits[:k])/k
            results['recall'][k] += sum(recall_hits[:k])/len(gold_answers)
        
        results['MRR'] += tmp_mrr
        results['MAP'] += tmp_map
    
    for k in ALL_K:
        results['precision'][k] /= len(pred_index)
        results['recall'][k] /= len(pred_index)
    results['MRR'] /= len(pred_index)
    results['MAP'] /= len(pred_index)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--retrieval_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--answer_file", type=str)
    args = parser.parse_args()
    
    # dedup APIs
    with open(args.retrieval_file, 'r') as f:
        retrieval_results = json.load(f)
    dedup_results = {}
    for key, value in retrieval_results.items():
        dedup_retrieved = []
        dedup_scores = []

        for i, r in enumerate(value['retrieved']):
            if r not in dedup_retrieved:
                dedup_retrieved.append(r)
                dedup_scores.append(value['score'][i])

        while len(dedup_retrieved) < 200:
            dedup_retrieved.append('NONE')
            dedup_scores.append(0.0)

        dedup_results[key] = {
            'retrieved': dedup_retrieved, 'score': dedup_scores}
    with open(args.retrieval_file.replace('.json', '_deduped.json'), 'w') as f:
        json.dump(dedup_results, f)
    
    
    results = eval_retrieval_from_file(
        args.data_file,
        args.retrieval_file.replace('.json', '_deduped.json'),
    )

    print(args.retrieval_file.replace('.json', '_deduped.json'))

    print(results)
    results = eval_post_retrieval_from_file(
        args.data_file,
        args.retrieval_file,
        args.corpus_file,
        args.answer_file,
        dedup_answer=False
    )
    print(results)
    results = eval_post_retrieval_from_file(
        args.data_file,
        args.retrieval_file,
        args.corpus_file,
        args.answer_file,
        dedup_answer=True,
    )
    print(results)
