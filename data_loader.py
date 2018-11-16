import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import glob
import time
import math
from sklearn.model_selection import train_test_split
import ast
import scipy
import re
import sentencepiece as spm
import json
import tqdm
import swifter
from multiprocessing import Pool, cpu_count, freeze_support
import gc
import msgpack


from utils import replace_tokens, norm_len, replace_tandems
try:
    from urllib.parse import unquote
except ImportError:
    from urlparse import unquote


def preprocessing_worker(x):
    x = replace_tokens(x, FLAGS.multy_replace)
    if FLAGS.multy_replace:
        if len(x.split()) > FLAGS.max_seq_len:
            x = b" ".join(x.split()[:FLAGS.max_seq_len])
        x = replace_tandems(x)
    if FLAGS.tqdm_imap:
        gc.collect()
    return x


def main(argv=None):
    if not os.path.exists(FLAGS.output_dataset):
        os.mkdir(FLAGS.output_dataset)

    print("per class limiter - ", FLAGS.per_class_limiter)

    info_df = pd.read_csv(FLAGS.dataset_info_file).set_index('id')
    with open(FLAGS.dataset_file, 'rb') as f:
        vectors = msgpack.unpackb(f.read())
    vectors_df = pd.DataFrame({'id': [v[0] for v in vectors], 'line': [v[1] for v in vectors]}).set_index('id')

    df = pd.concat([info_df, vectors_df], axis=1)
    df = df.drop_duplicates(subset=['line'])
    df = df.dropna()
    df['label'] = df.injection.apply(lambda x: 0 if not x else 1)
    print("df:\n", df.head())

    start_time = time.time()
    print("preprocessing data: ")
    with Pool(FLAGS.num_parallel_worker) as pool:
        if FLAGS.tqdm_imap:
            seq_in_data = tqdm.tqdm(pool.imap(preprocessing_worker, df["line"].values, FLAGS.chunksize),
                                    total=len(df["line"].values))
        else:
            seq_in_data = pool.map(
                preprocessing_worker,
                df["line"].values,
                FLAGS.chunksize)
            df["line"] = seq_in_data

    print("end preprocessing: ", time.time()-start_time)
    df = df[df['line'] != b""]
    df = df[df['line'] != b" "]
    df = df[df['line'] != b"\n"]

    if not FLAGS.test:
        with open("all_corpus.txt", "w") as f:
            for line in df.line.values:
                f.write(line.decode()+"\n")

        spm.SentencePieceTrainer.Train('--input=./all_corpus.txt --model_prefix='+FLAGS.output_dataset +
                                       '/main_vocab --vocab_size='+str(FLAGS.vocab_size)+' --model_type=unigram  --hard_vocab_limit=false')

        sp = spm.SentencePieceProcessor()
        sp.Load(FLAGS.output_dataset+"/main_vocab.model")
    else:
        sp = spm.SentencePieceProcessor()
        sp.Load(FLAGS.spm_model)

    print(df.head())
    print("Encoding as Ids..")
    df["line"] = df["line"].apply(sp.EncodeAsIds)
    print("Calculating len of seqs..")
    df["seq_in_len"] = df.line.apply(len).apply(
        lambda l: min(l, FLAGS.max_seq_len))
    print("Norming seqs..")
    df["line"] = df["line"].apply(
        lambda s: norm_len(s, FLAGS.max_seq_len))

    if not FLAGS.test:
        postfix = "train"
    else:
        postfix = "test"
    np.savez(
        os.path.join(FLAGS.output_dataset, postfix),
        seq_in=np.array(df.line.values.tolist()),
        label=np.array(df.label.tolist()),
        seq_in_len=np.array(df.seq_in_len.values.tolist())
    )

    print("\nBalance: 0:1 - {}:{}".format(len(df[df.injection == 0].values),
                                          len(df[df.injection == 1].values)))
    print(np.array(df.line.values.tolist()).shape)

    stats = {"max_seq_len": FLAGS.max_seq_len,
             "multy_replace": FLAGS.multy_replace}
    with open(os.path.join(FLAGS.output_dataset, "stats.json"), "w") as f:
        json.dump(stats, f)


if __name__ == '__main__':
    freeze_support()

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_boolean(
        'test', False, 'processing test dataset?')
    tf.app.flags.DEFINE_string(
        'output_dataset', './dataset', 'Path for storing dataset')
    tf.app.flags.DEFINE_string(
        'dataset_info_file', './raw_dataset/train.csv', 'Path to .csv file with dataset')
    tf.app.flags.DEFINE_string(
        'dataset_file', './raw_dataset/dataset.msgpack', 'Path to .msgpack file with vectors')
    tf.app.flags.DEFINE_integer(
        'max_seq_len', 800, 'Len of sentences')
    tf.app.flags.DEFINE_integer(
        'vocab_size', 4000, '(optional) Size of vocabulary for spm. Required if test := False')
    tf.app.flags.DEFINE_string(
        'spm_model', './dataset/main_vocab.model', "(optional) path to spm model")
    tf.app.flags.DEFINE_boolean(
        'multy_replace', True, 'If True - replacing consecutive tokens from one group to only one token')
    tf.app.flags.DEFINE_integer(
        'per_class_limiter', 290000, 'Limiter for num samples per class from dataset without nn errors. Set 0 for disable')
    tf.app.flags.DEFINE_integer(
        'num_parallel_worker', cpu_count() - 2, "Num parallel worker; Default cpu_count - 2")
    tf.app.flags.DEFINE_integer(
        'chunksize', 500, "chunksize for Pool")
    tf.app.flags.DEFINE_boolean(
        'tqdm_imap', False, "Progress bar for preprocessing. Warning: It's beta function, can lead to Memory Error")
    tf.app.run()
