"""
given a labeled and unlabeled datafile, creates a single file, taht contains the data
from both. the labels for the unlabeled data file will be UNK
"""
import argparse
from collections import OrderedDict
import csv


def get_col_by_role(role_string):
    col_by_role = {}
    for s in role_string.split(','):
        split_s = s.split('=')
        col_by_role[split_s[0]] = split_s[1]
    print('col_by_role', col_by_role)
    return col_by_role


def run(labeled_files, unlabeled_files, out_file, unlabeled_columns, labeled_columns, no_add_cust_tokens, add_columns, column_order):
    add_columns = add_columns.split(',')
    out_f = open(out_file, 'w')
    dict_writer = csv.DictWriter(out_f, fieldnames=column_order.split(','))
    dict_writer.writeheader()

    labeled_col_by_role = get_col_by_role(labeled_columns)
    unlabeled_col_by_role = get_col_by_role(unlabeled_columns)

    labeled_view1 = labeled_col_by_role['view1']
    labeled_label = labeled_col_by_role['label']

    unlabeled_view1 = unlabeled_col_by_role['view1']
    unlabeled_view2 = unlabeled_col_by_role['view2']

    unlabeled_by_first_utterance = OrderedDict()
    for filename in unlabeled_files:
        print(filename)
        with open(filename, 'r') as in_f:
            dict_reader = csv.DictReader(in_f)
            for row in dict_reader:
                view1 = row[unlabeled_view1]
                if not no_add_cust_tokens and not view1.startswith('<cust__'):
                    view1 = '<cust__ ' + view1 + ' __cust>'
                out_row = {
                    'view1': view1,
                    'view2': row[unlabeled_view2],
                    'label': 'UNK'
                }
                for k in add_columns:
                    out_row[k] = row[k]
                unlabeled_by_first_utterance[view1] = out_row
    print('loaded unlabeled')

    for filename in labeled_files:
        print(filename)
        with open(filename, 'r') as in_f:
            dict_reader = csv.DictReader(in_f)
            for row in dict_reader:
                view1 = row[labeled_view1]
                if view1 not in unlabeled_by_first_utterance:
                    print('warning: not found in unlabelled', view1)
                    continue
                out_row = unlabeled_by_first_utterance[view1]
                out_row['label'] = row[labeled_label]
                for k in add_columns:
                    assert out_row[k] == row[k]

    for row in unlabeled_by_first_utterance.values():
        dict_writer.writerow(row)

    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled-files', type=str, nargs='+', required=True)
    parser.add_argument('--unlabeled-files', type=str, nargs='+', required=True)
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--unlabeled-columns', type=str, default='view1=first_utterance,view2=context')
    parser.add_argument('--labeled-columns', type=str, default='view1=text,label=tag')
    parser.add_argument('--no-add-cust-tokens', action='store_true')
    parser.add_argument('--add-columns', type=str, default='id,question_body')
    parser.add_argument('--column-order', type=str, default='id,label,view1,view2,question_body')
    args = parser.parse_args()
    run(**args.__dict__)
