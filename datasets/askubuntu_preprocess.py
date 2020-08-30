"""
given askubuntu links, find the distribution of clusters formed of connected subgraphs

needs networkx installed (conda install networkx)

conda install -y networkx beautifulsoup4 lxml
python -c 'import nltk; nltk.download("stopwords")'

"""
import random
import argparse
import csv
from collections import defaultdict
import time
from os.path import expanduser as expand
from xml.dom import minidom

import networkx as nx
import torch
import numpy as np

from preprocessing.askubuntu import Preprocessor


class NullPreprocessor(object):
    def __call__(self, text):
        text = text.replace('\n', ' ').replace('\r', ' ')
        return True, text


def index_posts(args):
    """
    got through Posts.xml, and find for each post id, the id of the answer. store both in a csv file

    example row:
        <row Id="5" PostTypeId="1" CreationDate="2010-07-28T19:23:40.273" Score="22" ViewCount="581"
        Body="&lt;p&gt;What are some alternatives to upgrading without using the standard upgrade system?
        Suppose for example that I wanted to upgrade an Ubuntu installation on a machine with a poor Internet connection. 
        What would my options be? Could I just use a standard Ubuntu disk to upgrade this machine? If I already have a standard Ubuntu
         disk and want to use that, could I do a clean install without wiping data?&lt;/p&gt;&#xA;"
         OwnerUserId="10" LastEditorUserId="10581" LastEditDate="2014-02-18T13:34:25.793" LastActivityDate="2014-02-18T13:34:25.793"
         Title="What are some alternatives to upgrading without using the standard upgrade system?"
         Tags="&lt;upgrade&gt;&lt;live-cd&gt;&lt;system-installation&gt;"
         AnswerCount="2" CommentCount="1" FavoriteCount="1" />
    """
    in_f = open(expand(args.in_posts), 'r')
    out_f = open(expand(args.out_posts_index), 'w')
    dict_writer = csv.DictWriter(out_f, fieldnames=['question_id', 'answer_id'])
    dict_writer.writeheader()
    last_print = time.time()
    for n, row in enumerate(in_f):
        row = row.strip()
        if not row.startswith('<row'):
            continue
        dom = minidom.parseString(row)
        node = dom.firstChild
        att = node.attributes
        if 'AcceptedAnswerId' in att:
            id = att['Id'].value
            accepted_answer_id = att['AcceptedAnswerId'].value
            dict_writer.writerow({'question_id': id, 'answer_id': accepted_answer_id})
        if args.in_max_posts is not None and n >= args.in_max_posts:
            print('reached max rows => breaking')
            break
        if time.time() - last_print >= 3.0:
            print(n)
            last_print = time.time()


def get_clusters(in_dupes_file, in_max_dupes, max_clusters):
    """
    we're reading in the list of pairs of dupes. These are a in format of two post ids per line, like:
        615465 8653
        833376 377050
        30585 120621
        178532 152184
        69455 68850

    When we read these in, we have no idea whether these posts have answers etc. We just read in all
    the pairs of post ids. We are then going to add these post ids to a graph (each post id forms a node),
    and the pairs of post ids become connectsion in the graph. We then form all connected components.

    We sort the connected components by size (reverse order), and take the top num_test_clusters components
    We just ignore the other components
    """
    f_in = open(expand(in_dupes_file), 'r')
    csv_reader = csv.reader(f_in, delimiter=' ')
    G = nx.Graph()
    for n, row in enumerate(csv_reader):
        left = int(row[0])
        right = int(row[1])
        G.add_edge(left, right)
        if in_max_dupes is not None and n >= in_max_dupes:
            print('reached max tao rows => break')
            break
    print('num nodes', len(G))
    print('num clusters', len(list(nx.connected_components(G))))
    count_by_size = defaultdict(int)
    clusters_by_size = defaultdict(list)
    for i, cluster in enumerate(nx.connected_components(G)):
        size = len(cluster)
        count_by_size[size] += 1
        clusters_by_size[size].append(cluster)
    print('count by size:')
    clusters = []
    top_clusters = []
    for size, count in sorted(count_by_size.items(), reverse=True):
        for cluster in clusters_by_size[size]:
            clusters.append(cluster)
            if len(top_clusters) < max_clusters:
                top_clusters.append(cluster)

    print('len(clusters)', len(clusters))
    top_cluster_post_ids = [id for cluster in top_clusters for id in cluster]

    post2cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for post in cluster:
            post2cluster[post] = cluster_id

    return top_clusters, top_cluster_post_ids, post2cluster


def read_posts_index(in_posts_index):
    with open(expand(args.in_posts_index), 'r') as f:
        dict_reader = csv.DictReader(f)
        index_rows = list(dict_reader)
    index_rows = [{'question_id': int(row['question_id']), 'answer_id': int(row['answer_id'])} for row in index_rows]
    post2answer = {row['question_id']: row['answer_id'] for row in index_rows}
    answer2post = {row['answer_id']: row['question_id'] for row in index_rows}
    print('loaded index')
    return post2answer, answer2post


def load_posts(answer2post, post2answer, in_posts):
    # load in all the posts from Posts.xml
    post_by_id = defaultdict(dict)
    posts_f = open(expand(in_posts), 'r')
    last_print = time.time()
    for n, row in enumerate(posts_f):
        row = row.strip()
        if not row.startswith('<row'):
            continue
        row_id = row.partition(' Id="')[2].partition('"')[0]
        if row_id == '':
            print(row)
            assert row_id != ''
        row_id = int(row_id)
        if row_id in post2answer:
            """
            example row:
            <row Id="5" PostTypeId="1" CreationDate="2010-07-28T19:23:40.273" Score="22" ViewCount="581"
            Body="&lt;p&gt;What are some alternatives to upgrading without using the standard upgrade system?
            Suppose for example that I wanted to upgrade an Ubuntu installation on a machine with a poor Internet connection. 
            What would my options be? Could I just use a standard Ubuntu disk to upgrade this machine? If I already have a standard Ubuntu
             disk and want to use that, could I do a clean install without wiping data?&lt;/p&gt;&#xA;"
             OwnerUserId="10" LastEditorUserId="10581" LastEditDate="2014-02-18T13:34:25.793" LastActivityDate="2014-02-18T13:34:25.793"
             Title="What are some alternatives to upgrading without using the standard upgrade system?"
             Tags="&lt;upgrade&gt;&lt;live-cd&gt;&lt;system-installation&gt;"
             AnswerCount="2" CommentCount="1" FavoriteCount="1" />
            """
            dom = minidom.parseString(row)
            node = dom.firstChild
            att = node.attributes
            assert att['PostTypeId'].value == '1'
            post_by_id[row_id]['question_title'] = att['Title'].value
            post_by_id[row_id]['question_body'] = att['Body'].value
        elif row_id in answer2post:
            dom = minidom.parseString(row)
            node = dom.firstChild
            att = node.attributes
            assert att['PostTypeId'].value == '2'
            post_id = answer2post[row_id]
            post_by_id[post_id]['answer_body'] = att['Body'].value
        if time.time() - last_print >= 3.0:
            print(len(post_by_id))
            last_print = time.time()
        if args.in_max_posts is not None and n > args.in_max_posts:
            print('reached in_max_posts => terminating')
            break
    print('loaded info from Posts.xml')
    return post_by_id


def create_labeled(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_f = open(expand(args.out_labeled), 'w')  # open now to check we can

    clusters, post_ids, post2cluster = get_clusters(
        in_dupes_file=args.in_dupes, in_max_dupes=args.in_max_dupes, max_clusters=args.max_clusters)
    print('cluster sizes:', [len(cluster) for cluster in clusters])
    print('len(clusters)', len(clusters))
    print('len(post_ids) from dupes graph', len(post_ids))

    print('removing post ids which dont have answers...')
    post2answer, answer2post = read_posts_index(in_posts_index=args.in_posts_index)
    post_ids = [id for id in post_ids if id in post2answer]
    print('len(post_ids) after removing no answer', len(post_ids))
    new_clusters = []
    for cluster in clusters:
        cluster = [id for id in cluster if id in post2answer]
        new_clusters.append(cluster)
    clusters = new_clusters
    print('len clusters after removing no answer', [len(cluster) for cluster in clusters])

    post_ids_set = set(post_ids)
    print('len(post_ids_set)', len(post_ids_set))
    answer_ids = [post2answer[id] for id in post_ids]
    print('len(answer_ids)', len(answer_ids))

    preprocessor = Preprocessor(max_len=args.max_len) if not args.no_preprocess else NullPreprocessor()
    post_by_id = load_posts(answer2post=answer2post, post2answer=post2answer, in_posts=args.in_posts)

    count_by_state = defaultdict(int)
    n = 0
    dict_writer = csv.DictWriter(out_f, fieldnames=[
        'id', 'cluster_id', 'question_title', 'question_body', 'answer_body'])
    dict_writer.writeheader()
    for post_id in post_ids:
        if post_id not in post_by_id:
            count_by_state['not in post_by_id'] += 1
            continue
        post = post_by_id[post_id]
        if 'answer_body' not in post or 'question_body' not in post:
            count_by_state['no body, or no answer'] += 1
            continue
        count_by_state['ok'] += 1
        cluster_id = post2cluster[post_id]
        row = {
            'id': post_id,
            'cluster_id': cluster_id,
            'question_title': preprocessor(post['question_title'])[1],
            'question_body': preprocessor(post['question_body'])[1],
            'answer_body': preprocessor(post['answer_body'])[1]
        }
        dict_writer.writerow(row)
        n += 1
    print(count_by_state)
    print('rows written', n)


def create_unlabeled(args):
    """
    this is going to do:
    - take a question (specific type in Posts.xml)
    - match it with the accepted answer (using the index)
    - write these out
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_f = open(expand(args.out_unlabeled), 'w')  # open now, to check we can...

    post2answer, answer2post = read_posts_index(in_posts_index=args.in_posts_index)
    print('loaded index')
    print('posts in index', len(post2answer))

    preprocessor = Preprocessor(max_len=args.max_len) if not args.no_preprocess else NullPreprocessor()
    post_by_id = load_posts(post2answer=post2answer, answer2post=answer2post, in_posts=args.in_posts)
    print('loaded all posts, len(post_by_id)', len(post_by_id))

    count_by_state = defaultdict(int)
    n = 0
    dict_writer = csv.DictWriter(out_f, fieldnames=['id', 'question_title', 'question_body', 'answer_body'])
    dict_writer.writeheader()
    last_print = time.time()
    for post_id, info in post_by_id.items():
        if 'answer_body' not in info or 'question_body' not in info:
            count_by_state['no body, or no answer'] += 1
            continue
        count_by_state['ok'] += 1
        dict_writer.writerow({
            'id': post_id,
            'question_title': preprocessor(info['question_title'])[1],
            'question_body': preprocessor(info['question_body'])[1],
            'answer_body': preprocessor(info['answer_body'])[1]
        })
        if time.time() - last_print >= 10:
            print('written', n)
            last_print = time.time()
        n += 1

    print(count_by_state)


if __name__ == '__main__':
    root_parser = argparse.ArgumentParser()
    parsers = root_parser.add_subparsers()

    parser = parsers.add_parser('index-posts')
    parser.add_argument('--in-posts-dir', type=str, default='~/data/askubuntu.com')
    parser.add_argument('--in-posts', type=str, default='{in_posts_dir}/Posts.xml')
    parser.add_argument('--out-posts-index', type=str, default='{in_posts_dir}/Posts_index.csv')
    parser.add_argument('--in-max-posts', type=int, help='for dev/debugging mostly')
    parser.set_defaults(func=index_posts)

    parser = parsers.add_parser('create-labeled')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--in-dupes-dir', type=str, default='~/data/askubuntu')
    parser.add_argument('--in-dupes', type=str, default='{in_dupes_dir}/full.pos.txt')
    parser.add_argument('--max-clusters', type=int, default=20)
    parser.add_argument('--in-posts-dir', type=str, default='~/data/askubuntu.com')
    parser.add_argument('--in-posts', type=str, default='{in_posts_dir}/Posts.xml')
    parser.add_argument('--in-posts-index', type=str, default='{in_posts_dir}/Posts_index.csv')
    parser.add_argument('--in-max-dupes', type=int, help='for dev/debugging mostly')
    parser.add_argument('--in-max-posts', type=int, help='for dev/debugging mostly')
    parser.add_argument('--out-labeled', type=str, required=True)
    parser.add_argument('--no-preprocess', action='store_true')
    parser.add_argument('--max-len', type=int, default=500, help='set to 0 to disable')
    parser.set_defaults(func=create_labeled)

    parser = parsers.add_parser('create-unlabeled')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--in-posts-dir', type=str, default='~/data/askubuntu.com')
    parser.add_argument('--in-dupes-dir', type=str, default='~/data/askubuntu')
    parser.add_argument('--in-posts', type=str, default='{in_posts_dir}/Posts.xml')
    parser.add_argument('--in-posts-index', type=str, default='{in_posts_dir}/Posts_index.csv')
    parser.add_argument('--in-max-posts', type=int, help='for dev/debugging mostly')
    parser.add_argument('--out-unlabeled', type=str, required=True)
    parser.add_argument('--max-len', type=int, default=500, help='set to 0 to disable')
    parser.add_argument('--no-preprocess', action='store_true')
    parser.set_defaults(func=create_unlabeled)

    args = root_parser.parse_args()
    for k in [
            'in_dupes', 'out_labeled', 'in_posts', 'out_posts_index', 'in_posts_index',
            'out_unlabeled']:
        if k in args.__dict__:
            args.__dict__[k] = args.__dict__[k].format(**args.__dict__)

    func = args.func
    del args.__dict__['func']
    func(args)
