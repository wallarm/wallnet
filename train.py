import tensorflow as tf
import os
import numpy as np
import sentencepiece as spm
import math
import time
import json
from shutil import copy2
from utils import calculate_model_hash


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', './run', 'Path for storing model')
tf.app.flags.DEFINE_string('dataset', './dataset', 'Path to dataset')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Size of min batch for training')
tf.app.flags.DEFINE_integer('max_epoches', 300, 'Number of epoches for training')
tf.app.flags.DEFINE_integer('rnn_num_hiden', 256, 'Num hiden neurons in each RNN layer')
tf.app.flags.DEFINE_integer('rnn_num_layers', 2, 'Num RNN layers')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of embedding space')
tf.app.flags.DEFINE_integer('attention_hiden_layer_size', 128, 'Num neurons in hiden layer in attention')
# tf.app.flags.DEFINE_float('lambda_l2_regl', 0.0, "Value of lambda for Tikhonov's regularization")
tf.app.flags.DEFINE_integer('suffle_buffer_size', 10000, "Size of buffer for shuffeling data by TF Dataset API")
tf.app.flags.DEFINE_integer('max_to_keep', 3, "max checkpoint for storing")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, "dropout keep probability on traning")
tf.app.flags.DEFINE_integer('checkpoint_every_n_steps', 0, "if 0, checkpoints only every epoch")

class Model():
    def __init__(self, vocab_size, max_seq_len,
                 embedding_size=50,
                 rnn_num_hiden=256,
                 rnn_num_layers=2,
                 attention_hiden_layer_size=128,
                 batch_size=32,
                 suffle_buffer_size=100000):
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0), [], name="dropout_keep_prob")
        self.learning_rate = tf.placeholder_with_default(tf.constant(1e-4), [], name="learning_rate")

        self.ph_seq_len = tf.placeholder(tf.int32, [None, ], name="seq_len")
        self.ph_seq_in = tf.placeholder(tf.int32, [None, max_seq_len], name="sequence")
        self.ph_labels = tf.placeholder(tf.float32, [None], name="labels")

        train_dataset = tf.data.Dataset.from_tensor_slices((self.ph_seq_len, self.ph_seq_in, self.ph_labels)).shuffle(
            buffer_size=suffle_buffer_size).batch(batch_size)

        self.ph_valid_batch_size = tf.placeholder_with_default(tf.constant(1, dtype=tf.int64), [],
                                                               name="valid_batch_size")
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.ph_seq_len, self.ph_seq_in, self.ph_labels)).batch(
            self.ph_valid_batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        next_elements = iterator.get_next()

        self.training_init_op = iterator.make_initializer(train_dataset, name="training_init_op")
        self.validation_init_op = iterator.make_initializer(valid_dataset, name="validation_init_op")

        seq_len, seq_in, labels = next_elements

        embeddings_var = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True)
        embedded_seq = tf.nn.embedding_lookup(embeddings_var, seq_in)

        rnn_fw_cells = [
            tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(rnn_num_hiden),
                input_keep_prob=self.keep_prob,
                output_keep_prob=self.keep_prob,
                state_keep_prob=self.keep_prob
            ) for _ in range(rnn_num_layers)]

        rnn_bw_cells = [
            tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(rnn_num_hiden),
                input_keep_prob=self.keep_prob,
                output_keep_prob=self.keep_prob,
                state_keep_prob=self.keep_prob
            ) for _ in range(rnn_num_layers)]

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            rnn_fw_cells,
            rnn_bw_cells,
            embedded_seq,
            sequence_length=seq_len,
            dtype=tf.float32)

        with tf.variable_scope("pooling_layer"):
            self.c1 = tf.layers.conv1d(outputs, rnn_num_hiden*rnn_num_layers*2, 2, padding='valid',
                                           kernel_initializer=tf.glorot_uniform_initializer())
            self.c2 = tf.layers.conv1d(self.c1, rnn_num_hiden*rnn_num_layers, 2, padding='valid',
                                           kernel_initializer=tf.glorot_uniform_initializer())
            self.c2 = tf.layers.dropout(self.c2, self.keep_prob)

            self.avg_pool = tf.contrib.keras.layers.GlobalAveragePooling1D()(self.c2)
            self.max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()(self.c2)

        with tf.variable_scope("attention"):
            hidden_layer = tf.layers.dense(outputs, attention_hiden_layer_size, activation=tf.nn.relu)
            logits = tf.layers.dense(hidden_layer, 1, activation=None)
            alphas = tf.nn.softmax(logits, axis=1)
            self.attention_c = tf.reduce_sum(outputs*alphas, 1)

        self.concat_c = tf.concat((self.avg_pool, self.max_pool, self.attention_c), axis=-1)

        W = tf.Variable(tf.random_normal([int(self.concat_c.get_shape()[1]), 1]))
        b = tf.Variable(tf.random_normal([1]))
        self.logits = tf.nn.xw_plus_b(self.concat_c, W, b)
        self.prediction = tf.nn.sigmoid(self.logits)
        self.prediction = tf.identity(self.prediction, "prediction")

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.stop_gradient(tf.reshape(labels, (-1, 1))),
            logits=self.logits)
        self.loss = tf.reduce_mean(self.loss)
        self.loss = tf.identity(self.loss, "loss")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            optimizer=tf.train.AdamOptimizer,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            name="train_step",
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm', 'global_gradient_norm'])
        self.train_step = tf.identity(self.train_step, "train_step")

        print("Main graph is builded")
        self._make_metrics_and_summaries(labels)
        print("Metrics and summaries is builded")

    def _make_metrics_and_summaries(self, labels):
        with tf.variable_scope("metrics_and_summaries"):
            _, self.accuracy = tf.metrics.accuracy(labels, predictions=tf.round(self.prediction), name="accuracy")
            _, self.roc_auc = tf.metrics.auc(labels, tf.round(self.prediction), name="ROC_AUC")
            _, self.roc_prc = tf.metrics.auc(labels, tf.round(self.prediction), name="ROC_PRC",
                                             curve='PR', summation_method='careful_interpolation')
            _, self.recall = tf.metrics.recall(labels, tf.round(self.prediction), name="recall")
            _, self.precision = tf.metrics.precision(labels, tf.round(self.prediction), name="precision")

            labels_acc_summary = tf.summary.scalar("metrics/labels_accuracy", self.accuracy)
            recall_summary = tf.summary.scalar("metrics/stream_recall", self.recall)
            precision_summary = tf.summary.scalar("metrics/stream_precission", self.precision)
            roc_auc_summary = tf.summary.scalar("metrics/stream_roc_auc", self.roc_auc)
            roc_auc_summary = tf.summary.scalar("metrics/stream_roc_prc", self.roc_prc)
            self.summaries = tf.summary.merge_all()


def main(argv=None):
    with np.load(os.path.join(FLAGS.dataset, "train.npz")) as data:
        ds_seq_in = data["seq_in"]
        ds_label = data["label"]
        ds_seq_in_len = data["seq_in_len"]

    with np.load(os.path.join(FLAGS.dataset, "test.npz")) as data:
        ds_test_seq_in = data["seq_in"]
        ds_test_label = data["label"]
        ds_test_seq_in_len = data["seq_in_len"]

    # Assume that each row of `ds_seq_in` corresponds to the same row as `ds_label` and 'ds_seq_in_len'.
    assert ds_seq_in.shape[0] == ds_label.shape[0] == ds_seq_in_len.shape[0]
    assert ds_test_seq_in.shape[0] == ds_test_label.shape[0] == ds_test_seq_in_len.shape[0]

    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)
    copy2(os.path.join(FLAGS.dataset, "main_vocab.model"), FLAGS.run_dir)
    copy2(os.path.join(FLAGS.dataset, "main_vocab.vocab"), FLAGS.run_dir)
    copy2(os.path.join(FLAGS.dataset, "stats.json"), FLAGS.run_dir)

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.run_dir + "/main_vocab.model")
    SEQ_IN_VOCABULARY_SIZE = sp.GetPieceSize()
    MAX_SEQ_LEN = ds_seq_in.shape[1]
    del sp

    model = Model(SEQ_IN_VOCABULARY_SIZE, MAX_SEQ_LEN,
                  embedding_size=FLAGS.embedding_size,
                  rnn_num_hiden=FLAGS.rnn_num_hiden,
                  rnn_num_layers=FLAGS.rnn_num_layers,
                  batch_size=FLAGS.batch_size,
                  suffle_buffer_size=FLAGS.suffle_buffer_size)
    print("Model is builded...")

    if not os.path.exists(os.path.join(FLAGS.run_dir, "checkpoints")):
        os.mkdir(os.path.join(FLAGS.run_dir, "checkpoints"))
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if not os.path.exists(os.path.join(FLAGS.run_dir, "logs")):
        os.mkdir(os.path.join(FLAGS.run_dir, "logs"))
    timestamp = str(math.trunc(time.time()))
    test_writer = tf.summary.FileWriter(os.path.join(FLAGS.run_dir, "logs", timestamp + "-validation"))
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.run_dir, "logs", timestamp + "-training"), sess.graph)

    def test_and_checkpoint():
            # TEST
            print("\nTest: \n")
            sess.run(model.validation_init_op, feed_dict={model.ph_seq_len: ds_test_seq_in_len,
                                                        model.ph_seq_in: ds_test_seq_in,
                                                        model.ph_labels: ds_test_label,
                                                        model.ph_valid_batch_size: FLAGS.batch_size})
            all_acc = []
            all_precision = []
            all_recall = []
            all_roc_auc = []
            while True:
                try:
                    loss, accarucy, recall, precision, roc_auc, roc_prc, summaries, step = sess.run([
                        model.loss,
                        model.accuracy,
                        model.recall,
                        model.precision,
                        model.roc_auc,
                        model.roc_prc,
                        model.summaries,
                        model.global_step
                    ])
                    test_writer.add_summary(summaries, step)
                    # print("Test step: Loss: {:1.3}; Accarucy: {:1.3}; Precision: {:1.3}; Recall: {:1.3}; ROC_AUC: {:1.3}; ROC_PRC: {:1.3}".format(
                        # loss, accarucy, precision, recall, roc_auc, roc_prc
                    # ))
                    all_acc.append(accarucy)
                    all_precision.append(precision)
                    all_recall.append(recall)
                    all_roc_auc.append(roc_auc)
                except tf.errors.OutOfRangeError:
                    break
            mean_acc = np.mean(all_acc)
            mean_precision = np.mean(all_precision)
            mean_recall = np.mean(all_recall)
            mean_roc_auc = np.mean(all_roc_auc)
            print("Mean accuracy: {}".format(mean_acc))
            print("Mean precision: {}".format(mean_precision))
            print("Mean recall: {}".format(mean_recall))
            print("Mean ROC AUC: {}".format(mean_roc_auc))

            timestamp = str(math.trunc(time.time()))
            saved_file = saver.save(sess, os.path.join(FLAGS.run_dir, "checkpoints", 'model_' + timestamp),
                                    global_step=step)
            print("Saved file: " + saved_file)

            model_hash = calculate_model_hash(tf.train.latest_checkpoint(os.path.join(FLAGS.run_dir, "checkpoints")))
            with open(os.path.join(FLAGS.run_dir, "stats.json"), "r") as f:
                stats = json.loads(f.read())
                stats.update({
                    "step": int(step),
                    "test_accuracy": float(mean_acc),
                    "test_precision": float(mean_precision),
                    "test_recall": float(mean_recall),
                    "model": model_hash,
                    "batch_size": FLAGS.batch_size,
                    "time_of_saving": time.time()})
            with open(os.path.join(FLAGS.run_dir, "stats.json"), "w") as f:
                f.write(json.dumps(stats))

    step = 0
    for eid in range(FLAGS.max_epoches):
        print("Epoch: ", eid)
        # TRAIN
        sess.run(model.training_init_op, feed_dict={model.ph_seq_len: ds_seq_in_len,
                                                    model.ph_seq_in: ds_seq_in,
                                                    model.ph_labels: ds_label})

        while True:
            try:
                if step % 100 == 0 or step < 10:
                    _, loss, accarucy, recall, precision, roc_auc, roc_prc, summaries, step = sess.run([
                        model.train_step,
                        model.loss,
                        model.accuracy,
                        model.recall,
                        model.precision,
                        model.roc_auc,
                        model.roc_prc,
                        model.summaries,
                        model.global_step
                    ])
                    print("Step: {:3}; Loss: {:1.3}; Accarucy: {:1.3}; Precision: {:1.3}; Recall: {:1.3}; ROC_AUC: {:1.3}; ROC_PRC: {:1.3}".format(
                        step, loss, accarucy, precision, recall, roc_auc, roc_prc))
                    summary_writer.add_summary(summaries, step)
                else:
                    _, step = sess.run([model.train_step, model.global_step], feed_dict={model.keep_prob: 0.5})
                
                if FLAGS.checkpoint_every_n_steps != 0 and step % FLAGS.checkpoint_every_n_steps == 0:
                    test_and_checkpoint()

            except tf.errors.OutOfRangeError:
                break
        
        test_and_checkpoint()

if __name__ == '__main__':
    tf.app.run()
