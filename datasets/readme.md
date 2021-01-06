To generate the data from raw data:

For askubuntu:

```
export PYTHONPATH=.
mkdir ~/data/askubuntu_induction
python datasets/askubuntu_preprocess.py create-labeled --out-labeled ~/data/askubuntu_induction/preproc_labeled.csv
python datasets/askubuntu_preprocess.py create-unlabeled --out-unlabeled ~/data/askubuntu_induction/preproc_unlabeled.csv
python datasets/labeled_unlabeled_merger.py --labeled-files ~/data/askubuntu_induction/preproc_labeled.csv --unlabeled-files ~/data/askubuntu_induction/preproc_unlabeled.csv --out-file ~/data/askubuntu_induction/preproc_merged.csv --labeled-columns view1=question_title,label=cluster_id --unlabeled-columns view1=question_title,view2=answer_body --no-add-cust-tokens
```

For Twitter customer support:
```
(cd data; bunzip2 airlines_500onlyb.csv.bz2)
python datasets/twitter_dataset_preprocess.py --no-preprocess --out-examples ~/data/twittersupport/airlines_raw.csv
python datasets/twitter_airlines_raw_merger2.py
```
