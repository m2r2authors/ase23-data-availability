# java only right now
# direct (nl, api) pairs and (no, nl+api) pair
# man.id will have duplicates


import argparse
import csv
import json
import os
import pandas as pd
import random
from tqdm import tqdm

import sys
sys.path.append('../')
sys.path.append('../retriever')
sys.path.append('../retriever/simcse')

from collections import defaultdict

from retriever.simcse.model import RetrievalModel
from retriever.simcse.arguments import ModelArguments
from retriever.simcse.data_utils import OurDataCollatorWithPadding, tok_sentences

import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
)


def dump_contrastive_train(args, mode='sup'):
    """
    text1, text2 file
    oracle_man file, for reranking

    (api, api)
    (nl, nl)
    (nl1, nl2+api)  exponential problem (but api is not so important, make sure nl match semantically)
    (nl, api)
    """
#     with open(os.path.join(args.output_dir, 'man.tok.id'), 'r') as f:
    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.id'), 'r') as f:
#     with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.txt'), 'r') as f:
        doc_set = set([line.strip() for line in f.readlines()])
    df = pd.read_csv(args.train_file)
    train_dataset = []
    for idx in df.index:
        query = df['title'][idx]
        answer_list = df['answer'][idx].strip('][').split(',')
        for answer in answer_list:
            answer = answer.strip(" \'")
            train_dataset.append(
                json.dumps(
                    {
                        'text1': query,
                        'text2': answer
                    }
                )
            )
            train_dataset.append(
                json.dumps(
                    {
                        'text1': query,
                        'text2': query
                    }
                )
            )
    for doc in doc_set:
        train_dataset.append(
            json.dumps(
                {
                    'text1': doc,
                    'text2': doc
                }
            )
        )
    
    with open(os.path.join(args.output_dir, args.test_dir, f'train_{mode}.json'), 'w') as f:
        for jstr in train_dataset:
            f.write(jstr + '\n')


def dump_contrastive_train_title_sup(args, batch_size=512, seed=42, n_batches=50, strategy=1, drop_rate=0.0, n_samples=100):
    """
    Sampling training dataset used for sequential dataset
    
    Strategy1 - simplest:
    -  With replacement, then the groups with single sample will be used.
    
    Strategy2 - complex:
    - Without replacement, then only groups with more than one samples can be used.
    - 
    """
    
    df = pd.read_csv(args.train_file)
    train_dataset = []
    gps = df.groupby("answer")
    
    stats = {'single': 0, 'multi': 0}
    for k, v in gps:
        if v.shape[0] == 1:
            stats['single'] += 1
        else:
            stats['multi'] += 1
    print(stats)
    
    
    if strategy == 1:
        for _ in range(n_batches):
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            total_pairs = pd.concat([title1, title2], axis=1).sample(n=batch_size)
            for title1, title2 in zip(total_pairs['title1'], total_pairs['title2']):
                train_dataset.append(
                    json.dumps(
                        {
                            'text1': title1,
                            'text2': title2
                        }
                    )
                )
    elif strategy == 2:
        gp_sizes = gps.transform('size')
        m_gps = df[gp_sizes > 1].groupby('answer')
        for _ in range(n_batches):
            total_pairs = m_gps.sample(n=2, replace=False, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            total_pairs = pd.concat([title1, title2], axis=1).sample(n=batch_size)
            for title1, title2 in zip(total_pairs['title1'], total_pairs['title2']):
                    train_dataset.append(
                        json.dumps(
                            {
                                'text1': title1,
                                'text2': title2
                            }
                        )
                    )
    elif strategy == 3:
        for _ in range(n_batches):
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            answers = total_pairs[total_pairs.index%2==0]['answer'].reset_index(drop=True)
            total_pairs = pd.concat([title1, title2, answers], axis=1).sample(n=batch_size)
            for title1, title2, answer_list in zip(total_pairs['title1'], total_pairs['title2'], total_pairs['answer']):
                answer_list = [a.strip(" \'") for a in answer_list.strip('][').split(',')]
                a = random.choice(answer_list)
                train_dataset.append(
                    json.dumps(
                        {
                            'text1': title1,
                            'text2': f"{a.replace('.', ' ')}: {title2}"
                        }
                    )
                )
    elif strategy == 4:
        for _ in range(n_batches):
            unique_answers_in_batch = set()
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            answers = total_pairs[total_pairs.index%2==0]['answer'].reset_index(drop=True)
            total_pairs = pd.concat([title1, title2, answers], axis=1).sample(n=batch_size)
            for title1, title2, answer_list in zip(total_pairs['title1'], total_pairs['title2'], total_pairs['answer']):
                answer_list = [a.strip(" \'") for a in answer_list.strip('][').split(',')]
                answer_set = set(answer_list)
                diff_set = answer_set.difference(unique_answers_in_batch)
                if diff_set:
                    a = random.choice(list(diff_set))
                    unique_answers_in_batch.add(a)
                else:
                    a = random.choice(answer_list)
                train_dataset.append(
                    json.dumps(
                        {
                            'text1': title1,
                            'text2': f"{a.replace('.', ' ')}: {title2}"
                        }
                    )
                )
    elif strategy == 5:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = AutoConfig.from_pretrained(args.pretrained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        model_args = ModelArguments()
        model_args.sim_func = 'cls_distance.cosine'
        model_args.temp = 0.05
        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
        model = RetrievalModel(config=config,
                       model_type=args.pretrained_model_path,
                       num_layers=12,
                       all_layers=False,
                       idf=idf_dict,
                       rescale_with_baseline=False,
                       baseline_path=None,
                       tokenizer=tokenizer,
                       training_args = None,
                       model_args=model_args).to(device)
        collator = OurDataCollatorWithPadding(tokenizer.pad_token_id, idf_dict)
        mini_batches = []
        for _ in range(n_samples):
            current_batch = {
                'data': [],
                'score': 100.0
            }
            unique_answers_in_batch = set()
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            answers = total_pairs[total_pairs.index%2==0]['answer'].reset_index(drop=True)
            total_pairs = pd.concat([title1, title2, answers], axis=1).sample(n=batch_size)
            for title1, title2, answer_list in zip(total_pairs['title1'], total_pairs['title2'], total_pairs['answer']):
                answer_list = [a.strip(" \'") for a in answer_list.strip('][').split(',')]
                answer_set = set(answer_list)
                diff_set = answer_set.difference(unique_answers_in_batch)
                if diff_set:
                    a = random.choice(list(diff_set))
                    unique_answers_in_batch.add(a)
                else:
                    a = random.choice(answer_list)
                current_batch['data'].append(
                    {
                        'text1': title1,
                        'text2': f"{a.replace('.', ' ')}: {title2}"
                    }
                )
            # compute score
            features = tok_sentences(tokenizer, 
                             [d['text1'] for d in current_batch['data']] + [d['text2'] for d in current_batch['data']],
                             has_hard_neg=False,
                             total=batch_size,
                             max_length=32)
            batched_features = [{} for _ in range(batch_size)]
            for key in features.keys():
                for i in range(batch_size):
                    batched_features[i][key] = features[key][i]
            batched_features = collator(batched_features)
            batched_features = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batched_features.items()}
            with torch.no_grad():
                loss = model.forward(**batched_features)[0]
                print(loss)
                current_batch['score'] = loss
            mini_batches.append(current_batch)
            # sort
        mini_batches.sort(key=lambda x: x['score'], reverse=True)
        for batch in mini_batches[:n_batches]:
            for js in batch['data']:
                train_dataset.append(json.dumps(js))
    elif strategy == 6:
        for _ in range(n_batches):
            unique_answers_in_batch = set()
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            answers = total_pairs[total_pairs.index%2==0]['answer'].reset_index(drop=True)
            total_pairs = pd.concat([title1, title2, answers], axis=1).sample(n=batch_size)
            for title1, title2, answer_list in zip(total_pairs['title1'], total_pairs['title2'], total_pairs['answer']):
                answer_list = [a.strip(" \'") for a in answer_list.strip('][').split(',')]
                answer_set = set(answer_list)
                diff_set = answer_set.difference(unique_answers_in_batch)
                if diff_set:
                    a = random.choice(list(diff_set))
                    unique_answers_in_batch.add(a)
                else:
                    a = random.choice(answer_list)
                train_dataset.append(
                    json.dumps(
                        {
                            'text1': f"{a.replace('.', ' ')}: {title2}",
                            'text2': title1
                        }
                    )
                )
    elif strategy == 7:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = AutoConfig.from_pretrained(args.pretrained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        model_args = ModelArguments()
        model_args.sim_func = 'cls_distance.cosine'
        model_args.temp = 0.05
        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
        model = RetrievalModel(config=config,
                       model_type=args.pretrained_model_path,
                       num_layers=12,
                       all_layers=False,
                       idf=idf_dict,
                       rescale_with_baseline=False,
                       baseline_path=None,
                       tokenizer=tokenizer,
                       training_args = None,
                       model_args=model_args).to(device)
        collator = OurDataCollatorWithPadding(tokenizer.pad_token_id, idf_dict)
        mini_batches = []
        for _ in range(n_samples):
            current_batch = {
                'data': [],
                'score': 100.0
            }
            unique_answers_in_batch = set()
            total_pairs = gps.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
            title1 = total_pairs[total_pairs.index%2==0]['title'].rename('title1').reset_index(drop=True)
            title2 = total_pairs[total_pairs.index%2==1]['title'].rename('title2').reset_index(drop=True)
            answers = total_pairs[total_pairs.index%2==0]['answer'].reset_index(drop=True)
            total_pairs = pd.concat([title1, title2, answers], axis=1).sample(n=batch_size)
            for title1, title2, answer_list in zip(total_pairs['title1'], total_pairs['title2'], total_pairs['answer']):
                answer_list = [a.strip(" \'") for a in answer_list.strip('][').split(',')]
                answer_set = set(answer_list)
                diff_set = answer_set.difference(unique_answers_in_batch)
                if diff_set:
                    a = random.choice(list(diff_set))
                    unique_answers_in_batch.add(a)
                else:
                    a = random.choice(answer_list)
                current_batch['data'].append(
                    {
                        'text1': f"{a.replace('.', ' ')}: {title2}",
                        'text2': title1
                    }
                )
            # drop some colliding batches
            if len(unique_answers_in_batch) < 490:
                continue
            # compute score
            features = tok_sentences(tokenizer, 
                             [d['text1'] for d in current_batch['data']] + [d['text2'] for d in current_batch['data']],
                             has_hard_neg=False,
                             total=batch_size,
                             max_length=32)
            batched_features = [{} for _ in range(batch_size)]
            for key in features.keys():
                for i in range(batch_size):
                    batched_features[i][key] = features[key][i]
            batched_features = collator(batched_features)
            batched_features = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batched_features.items()}
            with torch.no_grad():
                loss = model.forward(**batched_features)[0]
                print(loss, len(unique_answers_in_batch))
                current_batch['score'] = loss
            mini_batches.append(current_batch)
            # sort
        mini_batches.sort(key=lambda x: x['score'], reverse=True)
        for batch in mini_batches[:n_batches]:
            for js in batch['data']:
                train_dataset.append(json.dumps(js))
        print('total batches:', len(train_dataset) / batch_size)
    elif strategy == 8:
        total = 0
        max_prod = 0
        prod_list = []
        # threshold = 20000
        # threshold = 5000
        threshold = 1000
        for k, v in tqdm(gps):  # groups: 78678
            if v.shape[0] <= 1:  # pair doesn't exist
                continue
            add_to_train_dataset = []
            product = v.shape[0] * (v.shape[0]-1)
            for query2, answer_list in zip(v['title'], v['answer']):
                answer_list = answer_list.strip('][').split(',')
                for answer in answer_list:
                    answer = answer.strip(" \'")
                    for query1 in v['title']:
                        if query1 == query2:
                            continue
                        add_to_train_dataset.append(
                            json.dumps(
                                {
                                    'text1': query1,
                                    'text2': f"{answer.replace('.', ' ')}: {query2}"
                                }
                            )
                        )            # product = v.shape[0] * len(query_answer_pairs)
            if product > max_prod:
                max_prod = product
            total += product

            prod_list.append(product)

            if len(add_to_train_dataset) > threshold:   # FIXME: thresholding might not reasonable
                add_to_train_dataset = random.choices(add_to_train_dataset, k=threshold)
                train_dataset += add_to_train_dataset
    elif strategy == 9:
        # map answer to answer list
        # group by API in API set
        # 
        pass
    else:
        raise NotImplementedError("Mini-batch sampling strategy not implemeneted!")
    
    with open(os.path.join(args.output_dir, args.test_dir, f'train_uni_sup_strategy_{strategy}.json'), 'w') as f:
        for jstr in train_dataset:
            f.write(jstr + '\n')
        


def dump_test(args, csv_file, split, strip_str=']['):
    "mimic `cmd_dev.oracle_man.full.json`"

    df = pd.read_csv(csv_file)
    dev_oracle = []
    nl_list = []    # id directly from the list id
    nl_id_list = []

    for idx in df.index:
        # print(df['title'][idx], df['answer'][idx])
        query = df['title'][idx]
        answer_list = df['answer'][idx].strip(strip_str).split(',')
        answer_list = [answer.strip(" \'") for answer in answer_list]
        dev_oracle.append(
            {
                'nl': query,
                'question_id': str(idx),
                'oracle_man': answer_list
            }
        )
        nl_list.append(query)
        nl_id_list.append(idx)
    
    with open(os.path.join(args.output_dir, args.test_dir, f'{split}_nl.txt'), 'w') as f:
        for nl in nl_list:
            f.write(nl + '\n')
    with open(os.path.join(args.output_dir, args.test_dir, f'{split}_nl.id'), 'w') as f:
        for nl_id in nl_id_list:
            f.write(str(nl_id) + '\n')
    with open(os.path.join(args.output_dir, args.test_dir, f'cmd_{split}.oracle_man.full.json'), 'w') as f:
        json.dump(dev_oracle, f, indent=2)


def dump_corpus_pool(args):

    df = pd.read_csv(args.train_file)

    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.id'), 'w') as f:
        for query, answer_list in zip(df['title'], df['answer']):
            answer_list = answer_list.strip('][').split(',')
            for answer in answer_list:
                answer = answer.strip(" \'")
                f.write(answer + '\n')
    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.txt'), 'w') as f:
        for query, answer_list in zip(df['title'], df['answer']):
            answer_list = answer_list.strip('][').split(',')
            for answer in answer_list:
                answer = answer.strip(" \'")
                f.write(f"{answer.replace('.', ' ')}: {query}" + '\n')
#                 f.write(f"{query}: {answer.replace('.', ' ')}" + '\n')
    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.answers'), 'w') as f:
        for query, answer_list in zip(df['title'], df['answer']):
            answer_list = answer_list.strip('][').split(',')
            answer_list = [answer.strip(" \'") for answer in answer_list]
            for _ in answer_list:
                f.write(json.dumps(answer_list) + '\n')


def clean_train_test_split(args, valid_size=500):
    df_original_train = pd.read_csv(args.original_train_file)
    df_test = pd.read_csv(args.test_file)
    df_train = df_original_train[~(df_original_train["title"].isin(df_test["title"].to_list()))]
    df_valid = df_train.sample(valid_size)
    df_train = df_train.drop(df_valid.index)
    df_train.to_csv(args.train_file)
    df_valid.to_csv(args.valid_file)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_train_file", default='data/raw/BIKER_train.QApair.csv', type=str)
    parser.add_argument("--train_file", default='data/raw/Biker_train_clean_biker_test.csv', type=str)
    parser.add_argument("--valid_file", default='data/raw/Biker_valid_clean_biker_test.csv', type=str)
    parser.add_argument("--test_file", default='data/raw/Biker_test_filtered.csv', type=str)
    parser.add_argument("--test_dir", default='biker_test_related', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pretrained_model_path", default=None, type=str)
    parser.add_argument("--output_dir", default='data/', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.test_dir)):
        os.mkdir(os.path.join(args.output_dir, args.test_dir))
#     random.seed(42)

    if not args.pretrained_model_path:
        random.seed(42)
        clean_train_test_split(args)
        dump_corpus_pool(args)
        dump_contrastive_train(args, mode='sup_unsup')
        dump_test(args, args.train_file, 'train')
        dump_test(args, args.valid_file, 'valid')
        dump_test(args, args.test_file, 'test')
    else:
#         dump_contrastive_train_title_sup(args, strategy=1, n_batches=50)
#         dump_contrastive_train_title_sup(args, strategy=2)
#         dump_contrastive_train_title_sup(args, strategy=3, n_batches=50)
#         dump_contrastive_train_title_sup(args, strategy=4, n_batches=100)
        dump_contrastive_train_title_sup(args, strategy=5, n_batches=200, n_samples=300)
#         dump_contrastive_train_title_sup(args, strategy=6, n_batches=100)
#         dump_contrastive_train_title_sup(args, strategy=7, n_batches=100, n_samples=200)
#         dump_contrastive_train_title_sup(args, strategy=8, n_batches=100, n_samples=100)


if __name__ == '__main__':
    main()