# Hypernym discovery

A hypernym discovery system which learns to predict is-a relationships between words using projection learning (see 
http://aclweb.org/anthology/S18-1116). 

The data of SemEval-2018 Task 9  is used for training and testing.


## Requirements

- Python 3 (tested using version 3.6.1)
- PyTorch (tested using version 0.3.0.post4)
- Pyhocon 
- Joblib
- Lots of disk space (downloading, unzipping, and preprocessing the corpus requires around 40 GB for sub-task 1A)
- Bash (to get the data and corpora, and to install word2vec)
- C compiler if you install word2vec


## Usage

Make directory where we can store a lot of data:

```bash
mkdir [dir-data]
```

Get training and evaluation data from the website of SemEval-2018 Task 9 (also copies scoring script in current directory):

```bash
./get_data.sh [dir-data]
```

Get corpus from the website of SemEval-2018 Task 9:

```bash
./get_corpus.sh [subtask dir-data]
```

Make preprocessed corpus and vocab:

```bash
python prep_corpus.py [subtask path-corpus dir-datasets path-output]
```

Install word2vec in current directory:

```bash
./install_word2vec.sh
```

Train word embeddings using word2vec:

```bash
word2vec/trunk/word2vec -train [path-corpus] -read-vocab [path-vocab] -output [path-output] -cbow 0 -negative 10 -size 200 -window 7 -sample 1e-5 -min-count 1 -iter 10 -threads 8 -binary 0 
```

Preprocess data and write in a pickle file:

```bash
python prep_data.py [subtask dir-datasets path-embeddings path-output]
```

Review hyperparameter settings in hparams.conf.

Train model on training and dev data in pickle file, write a model and a log file in `dir-model`:

```bash
python train.py [path-pickle path-hparams dir-model]
```

Load trained model, make predictions on test queries:

```bash
python predict.py [path-model path-pickle path-output]
```

Evaluate predictions on the test set using scoring script of SemEval-2018 Task 9:

```bash
python2.7 path/to/SemEval-Task9/task9-scorer.py path/to/SemEval-Task9/test/gold/<subtask>.<language>.test.gold.txt path/to/output/pred.txt
```


 
