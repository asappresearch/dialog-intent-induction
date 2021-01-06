"""
takes contents of airlines_raw.csv, and of data/airlines_mergedb.csv, and creates
airlines_500_merged.csv, using the raw tweets from airlines_raw.csv, and the tags from airlines_500_mergedb.csv
"""
import argparse, os, time, csv, json
from os import path
from os.path import join, expanduser as expand

def run(args):
    with open(expand(args.airlines_raw_file), 'r') as f:
        dict_reader = csv.DictReader(f)
        raw_rows = list(dict_reader)
    raw_row_by_tweet_id = {int(row['first_tweet_id']): row for i, row in enumerate(raw_rows)}

    f_in = open(expand(args.airlines_merged_file), 'r')
    f_out = open(expand(args.out_file), 'w')
    dict_reader = csv.DictReader(f_in)
    dict_writer = csv.DictWriter(f_out, fieldnames=['first_tweet_id', 'tag', 'first_utterance', 'context'])
    dict_writer.writeheader()
    for i, old_merged_row in enumerate(dict_reader):
        if old_merged_row['first_tweet_id'] == '':
            continue
        tweet_id = int(old_merged_row['first_tweet_id'])
        raw_row = raw_row_by_tweet_id[tweet_id]
        raw_row['tag'] = old_merged_row['tag']
        dict_writer.writerow(raw_row)
    # so, we accidentally had a blank line at the end of the non-raw dataset. add that in here...
    dict_writer.writerow({'first_tweet_id': '', 'tag': 'UNK', 'first_utterance': '', 'context': ''})
    f_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--airlines-raw-file', type=str, default='~/data/twittersupport/airlines_raw.csv')
    parser.add_argument('--airlines-merged-file', type=str, default='data/airlines_500_mergedb.csv')
    parser.add_argument('--out-file', type=str, default='~/data/twittersupport/airlines_raw_merged.csv')
    args = parser.parse_args()
    run(args)
