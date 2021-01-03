"""
preprocessing twitter dataset
"""
import time
import os
from os import path
from os.path import join, expanduser as expand
import json
import re
import csv
from collections import defaultdict
import argparse

import nltk

from preprocessing import twitter_airlines


all_companies = 'AppleSupport,Amazon,Uber_Support,SpotifyCares,Delta,Tesco,AmericanAir,TMobileHelp,comcastcares,British_Airways,SouthwestAir,VirginTrains,Ask_Spectrum,XboxSupport,sprintcare,hulu_support,AskPlayStation,ChipotleTweets,UPSHelp,AskTarget'.split(',')

"""
example rows:

tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,sprintcare,False,Tue Oct 31 22:10:47 +0000 2017,@115712 I understand. I would like to assist you. We would need to get you into a private secured link to further assist.,2,3
2,115712,True,Tue Oct 31 22:11:45 +0000 2017,@sprintcare and how do you propose we do that,,1
"""

class NullPreprocessor(object):
    def __call__(self, text, info_dict=None):
        is_valid = True
        if 'DM' in text:
            is_valid = False
        if 'https://t.co' in text:
            is_valid = False

        text = text.replace('\n', ' ').replace('\r', ' ')

        target_company = info_dict['target_company']
        cust_twitter_id = info_dict['cust_twitter_id']
        text = text.replace('@' + target_company, ' __company__ ').replace('@' + cust_twitter_id, ' __cust__ ')

        return is_valid, text

def run_for_company(in_csv_file, in_max_rows, examples_writer, target_company, no_preprocess):
    f_in = open(expand(in_csv_file), 'r')
    node_by_id = {}
    start_node_ids = []
    dict_reader = csv.DictReader(f_in)
    next_by_prev = {}
    for row in dict_reader:
        id = row['tweet_id']
        prev = row['in_response_to_tweet_id']
        next_by_prev[prev] = id
        if prev == '' and ('@' + target_company) in row['text']:
            start_node_ids.append(id)
        node_by_id[id] = row
        if in_max_rows is not None and len(node_by_id) >= in_max_rows:
            print('reached max rows', in_max_rows, '=> breaking')
            break
    print('len(node_by_id)', len(node_by_id))
    print('len(start_node_ids)', len(start_node_ids))
    count_by_status = defaultdict(int)
    count_by_count = defaultdict(int)

    preprocessor = twitter_airlines.Preprocessor() if not no_preprocess else NullPreprocessor()

    for i, start_node_id in enumerate(start_node_ids):
        conversation_texts = []
        first_utterance = None
        is_valid = True
        node = node_by_id[start_node_id]
        cust_twitter_id = node['author_id']
        while True:
            text = node['text'].replace('\n', ' ')
            if node['inbound'] == 'True':
                start_tok = '<cust__'
                end_tok = '__cust>'
            else:
                start_tok = '<rep__'
                end_tok = '__rep>'

            _valid, text = preprocessor(text, info_dict={
                'target_company': target_company,
                'cust_twitter_id': cust_twitter_id
            })
            if not _valid:
                is_valid = False
            text = start_tok + ' ' + text + ' ' + end_tok
            if first_utterance is None:
                first_utterance = text
            else:
                conversation_texts.append(text)
            response_id = next_by_prev.get(node['tweet_id'], None)
            if response_id is None:
                count_by_count[len(conversation_texts) + 1] += 1
                if is_valid:
                    examples_writer.writerow({
                        'first_tweet_id': start_node_id,
                        'first_utterance': first_utterance,
                        'context': ' '.join(conversation_texts)
                    })
                    count_by_status['accept'] += 1
                else:
                    count_by_status['not_valid'] += 1
                break
            if response_id not in node_by_id:
                count_by_status['response id not found'] += 1
                print('warning: response_id', response_id, 'not found => skipping conversation')
                break
            node = node_by_id[response_id]
    print(count_by_status)

def run_aggregated(in_csv_file, in_max_rows, out_examples, target_companies, no_preprocess):
    with open(expand(out_examples), 'w') as f_examples:
        examples_writer = csv.DictWriter(f_examples, fieldnames=['first_tweet_id', 'first_utterance', 'context'])
        examples_writer.writeheader()
        for company in target_companies:
            print(company)
            run_for_company(in_csv_file, in_max_rows, examples_writer, company, no_preprocess)

def run_by_company(in_csv_file, in_max_rows, out_examples_templ, target_companies, no_preprocess):
    for company in target_companies:
        print(company)
        out_examples = out_examples_templ.format(company=company)
        with open(expand(out_examples), 'w') as f_examples:
            examples_writer = csv.DictWriter(f_examples, fieldnames=['first_tweet_id', 'first_utterance', 'context'])
            examples_writer.writeheader()
            run_for_company(in_csv_file, in_max_rows, examples_writer, company, no_preprocess)

if __name__ == '__main__':
    root_parser = argparse.ArgumentParser()
    parsers = root_parser.add_subparsers()

    parser = parsers.add_parser('run-aggregated')
    parser.add_argument('--in-csv-file', type=str, default='~/data/twittersupport/twcs.csv')
    parser.add_argument('--in-max-rows', type=int, help='for dev/debugging mostly')
    parser.add_argument('--out-examples', type=str, required=True)
    parser.add_argument('--target-companies', type=str, default='Delta,AmericanAir,British_Airways,SouthwestAir')
    parser.add_argument('--no-preprocess', action='store_true')
    parser.set_defaults(func=run_aggregated)

    parser = parsers.add_parser('run-by-company')
    parser.add_argument('--in-csv-file', type=str, default='~/data/twittersupport/twcs.csv')
    parser.add_argument('--in-max-rows', type=int, help='for dev/debugging mostly')
    parser.add_argument('--out-examples-templ', type=str, default='~/data/twittersupport/{company}.csv')
    parser.add_argument('--target-companies', type=str, default=','.join(all_companies))
    parser.add_argument('--no-preprocess', action='store_true')
    parser.set_defaults(func=run_by_company)

    args = root_parser.parse_args()

    if args.func == run_aggregated:
        args.out_examples = args.out_examples.format(**args.__dict__)
        args.target_companies = args.target_companies.split(',')
    elif args.func == run_by_company:
        args.target_companies = args.target_companies.split(',')

    func = args.func
    del args.__dict__['func']
    func(**args.__dict__)
