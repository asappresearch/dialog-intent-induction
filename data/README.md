Data
----

Training vs test data
---------------------

We train on all data, without labels. We use the labels in order to evaluate the resulting clusters.

Twitter Airlines Customer Support
---------------------------------

The data is available in two version:

- raw: minimal redaction (company and customer twitter id), no preprocessing: [airlines_raw.csv.bz2](airlines_raw.csv.bz2)
- redacted, and preprocessed:  [airlines_processed.csv.bz2](airlines_processed.csv.bz2)

We sampled 500 examples, and annotated them. 8 examples were rejected because not English, leaving 492 labeled examples. The remaining examples were labeled `UNK`.

AskUbuntu
---------

- raw, no preprocessing: [askubuntu_raw.csv.bz2](askubuntu_raw.csv.bz2)
- preprocessed: [askubuntu_processed.csv.bz2](askubuntu_processed.csv.bz2)
