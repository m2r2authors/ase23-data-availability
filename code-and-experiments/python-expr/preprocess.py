import argparse
import csv
import json
import os
import pandas as pd
import random
from tqdm import tqdm


def dump_contrastive_train(args, mode):
    with open(os.path.join(args.output_dir, 'man.tok.id'), 'r') as f:
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

            
def dump_contrastive_train_title_sup(args, batch_size=512, seed=42, n_batches=50, strategy=1, drop_rate=0.0):
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
    
    
    # FIXME: need acceleration, current implementation is too slow
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
    elif strategy == 3:  # still need API. positive pair means: (1) same api; (2) different / or maybe same title of same group
        pass
    elif strategy == 3:  # lib-aware
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
    """
    can be problematic if training set doesn't cover test set

    with only train the corpus can be 500k
    1000+ conv2d
    """
    df = pd.read_csv(args.train_file)

    # # tmp doc
    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.id'), 'w') as f:
        # for doc in doc_set:
        #     f.write(doc + '\n')
        for query, answer_list in zip(df['title'], df['answer']):
            answer_list = answer_list.strip('][').split(',')
            for answer in answer_list:
                answer = answer.strip(" \'")
                f.write(answer + '\n')
    with open(os.path.join(args.output_dir, args.test_dir, 'man.tok.txt'), 'w') as f:
        # for doc in doc_set:
        #     f.write(doc + '\n')
        for query, answer_list in zip(df['title'], df['answer']):
            answer_list = answer_list.strip('][').split(',')
            for answer in answer_list:
                answer = answer.strip(" \'")
                f.write(f"{answer.replace('.', ' ')}: {query}" + '\n')
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
    parser.add_argument("--original_train_file", default='data/raw/python_ml_total_unique_API_train.csv', type=str)
    parser.add_argument("--train_file", default='data/raw/python_train_clean_biker_test.csv', type=str)
    parser.add_argument("--valid_file", default='data/raw/python_valid_clean_biker_test.csv', type=str)
    parser.add_argument("--test_file", default='data/raw/python_ml_test_set_manual_filter_if_how_to_labeled.csv', type=str)
    parser.add_argument("--test_dir", default='python_test_related', type=str)
    parser.add_argument("--output_dir", default='data/', type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.test_dir)):
        os.mkdir(os.path.join(args.output_dir, args.test_dir))
    random.seed(args.seed)
#     clean_train_test_split(args)
    dump_corpus_pool(args)
#     dump_contrastive_train(args, mode='sup_unsup')
#     dump_contrastive_train_title_sup(args, strategy=1, n_batches=200)
#     dump_contrastive_train_title_sup(args, strategy=2)
#     dump_test(args, args.train_file, 'train')
#     dump_test(args, args.valid_file, 'valid')
#     dump_test(args, args.test_file, 'test', '\}\{')


if __name__ == '__main__':
    main()