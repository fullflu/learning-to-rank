# Learning to Rank repository

A simple Python implementation of Learning to Rank.

## Requirements
- Python (2.7.6)
- chainer  (1.16.0)
- six (1.10.0)
- numpy (1.11.1)
- pandas (0.17.1)

at least worked well on Ubuntu 14.04

## Listnet
This implementation is based on the paper: "Learning to Rank: From Pairwise Approach to Listwise Approach", Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

Although learning by Listwise cost based on Top-k probability has been proposed, for simplicity, I focus on Listwise cost based on Top-1 probability here.

### Usage
- File preparation

Users need to prepare at least one csv file (e.g., "datasets.csv") in the following format:
```
In [1]: import pandas as pd
In [2]: datasets = pd.read_csv("datasets.csv")
In [3]: datasets
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
Other column names are not available for session.

"score" column implies the 'goodness' of the row (e.g., the relevance score of each document to the corresponding query, sales of each item , or others). Notice that scores must be non-negative to calculate normalized dcg.
Other column names are not available for score.

"feature_1" column is a feature you use. Multiple features can also be used.
Other column names are available for features.

You can execute training and validation using this datasets.
If you do not want to execute validation (i.e., want to use all data for training), you can use a "val_ratio" argument as follows: "-vr 0."

- File preparation (option)

If you want to test and evaluate the performance of the learnt model, "datasets_test.csv" is necessary. The format of this file is the same as "datasets.csv"

If you want to predict rankings of new datasets, whose true scores are unknown, "datasets_test_noscore.csv" is necessary. The format of this file is the same as "datasets.csv", except that "score" column is not included.

- Execution
```
git clone https://github.com/fullflu/learning-to-rank.git
cd learning-to-rank/blackbox
python listnet.py -trvf "datasets.csv" - tesf "datasets_test.csv" -tenf "datasets_test_noscore.csv" -vr 0.5
```

You can add other argments if you like.
 

If you find bugs, I hope you could reports them.
Other methods (both parametric and non-parametric) will be implemented.

Nov, 2016
