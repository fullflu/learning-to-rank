# Learning-to-rank repository

## Requirement
- Python (2.7.6)
- chainer  (1.16.0)
- six (1.10.0)
- numpy (1.11.1)
- pandas (0.17.1)

at least worked well on Ubuntu 14.04

## Listnet
This implementation is based on the paper: "Learning to Rank: From Pairwise Approach to Listwise Approach", Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

Although learning by Listwise cost based on Top-k probability has been proposed, for simplicity, we focus on Listwise cost based on Top-1 probability.

### Usage
- File preparation

Users need to prepare at least one csv file ("datasets_train.csv) in the following format:
```
In [1]: import pandas as pd
In [2]: datasets_train = pd.read_csv("datasets_train.csv")
In [3]: datasets_train
Out[3]: 
   session  score  feature_1
0        1      1   5.571190
1        2      9   8.736900
2        1      8   7.251731
3        1      3   8.368537
4        3      2  10.787266
5        3      7   6.008305
6        2      4  11.639217
7        2     11   8.271735
```

"session" column means session ID of the row (e.g., query ID , the date, or others).

"score" column implies the 'goodness' of the row (e.g., the relevance score of each document to the corresponding query, sales of each item , or others). Notice that scores must be non-negative to calculate normalized dcg.

"feature_1" column is a feature you made. Multiple features can also be used.

If you want to test and evaluate the performance of the learnt model, "datasets_test.csv" is necesarry. The format of this file is the same as "datasets_train.csv"

If you want to predict rankings of new datasets, whose true scores are unknown, "datasets_new.csv" is necessary. The format of this file is the same as "datasets_train.csv", except that "score" column is not included.

- Calculation after cloning
```
cd learning-to-rank/blackbox
python listnet_open.py -trf "datasets_train.csv" - tef "datasets_test.csv" -nf "datasets_new.csv"
```

You can add other argments if you like.


If you find bugs, I hope you could reports them.
