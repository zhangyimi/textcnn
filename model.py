#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
import tensorflow as tf
import time
import os
import numpy as np
class TextCNN():
    def __init__(self, embeddings,  update_embedding=True, n_epoch=60, batch_size=64, lr=0.001, num_filters=32, vocab_dim=100,
                 sen_max_len=150, num_class=2):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.embeddings = embeddings
        self.update_embedding = update_embedding
        self.lr = lr
        self.num_filters = num_filters
        self.vocab_dim = vocab_dim
        self.sen_max_len = sen_max_len
        self.num_class = num_class
        self.clip_grad = 5
        self.scores = 0
        self.timestamp = str(int(time.time()))
        # self.global_step = 0


    def bulid_graph(self):
        self.add_placeholders()
        self.get_dataset()
        self.lookup_layer_op()
        self.convAndMaxpool_op()
        self.dense_op()
        self.loss_op()
        self.optimizer_op()
        self.init_op()
        self.evaluate()

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        with tf.Session() as sess:
            summary_path = os.path.join('data/'+self.timestamp+'/logs/train')
            self.add_summary(sess, summary_path=summary_path)
            sess.run(self.init)
            sess.run(self.iterator.initializer, feed_dict={self.x: train[0], self.y: train[1]})
            for step in range(100000):
                try:

                    _, loss = sess.run([self.train_op, self.loss],feed_dict={self.dropout: 0.5} )  # train

                    if step % 50 == 0:
                        summary1 , loss, step_nums = sess.run([ self.merged,self.loss, self.global_step], feed_dict={self.dropout: 0.5})
                        # test feed进test的数据，不影响外面，循环继续依旧用的是train数据（亲测）。


                        summary2, accuracy, precision, recall, f1,logits = sess.run((self.merged, self.accuracy, self.precision, self.recall, self.f1,self.logits), feed_dict={self.bx: dev[0], self.by: dev[1],self.dropout:1.0})

                        print('step: %i,epoch: %d}' % (step_nums,step_nums//self.batch_size+1) , '|train loss:', loss, )
                        print( '|test accuracy:', accuracy ,'precision:', precision, 'recall:', recall, 'f1:', f1)
                        print('---------')

                        self.file_writer.add_summary(summary1, step_nums)
                        self.file_writer.add_summary(summary2, step_nums)
                        if f1 > self.scores:
                            self.scores = f1
                            dir_check = 'data/'+self.timestamp+'/checkpoints'

                            if not os.path.exists(dir_check):
                                os.mkdir(dir_check)
                            saver.save(sess, os.path.join(dir_check,'model'), global_step=step_nums)
                except tf.errors.OutOfRangeError:  # if training takes more than 3 epochs, training will be stopped
                    print('Finish the last epoch.')
                    break
    def test(self, dev,timestamp):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            dir_check = 'data/' + timestamp + '/checkpoints'
            ckpt_file = tf.train.latest_checkpoint(dir_check)
            print('restore from :',ckpt_file)
            saver.restore(sess, ckpt_file)
            accuracy, precision, recall, f1,logits = sess.run(( self.accuracy, self.precision, self.recall, self.f1,self.logits), feed_dict={self.bx: dev[0], self.by: dev[1],self.dropout:1.0})
            print( '|test accuracy:', accuracy ,'precision:', precision, 'recall:', recall, 'f1:', f1)




    def add_placeholders(self):
        self.x = tf.placeholder(tf.int32, [None, self.sen_max_len])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.global_step = tf.Variable(tf.constant(0))

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        dataset = dataset.shuffle(buffer_size=2000)  # choose data randomly from this buffer
        dataset = dataset.batch(self.batch_size)  # batch size you will use
        dataset = dataset.repeat(self.n_epoch)  # repeat for 3 epochs
        self.iterator = dataset.make_initializable_iterator()
        # 构建网络
        self.bx, self.by = self.iterator.get_next()

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding, name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.bx, name="word_embeddings")

        word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
        # 相当于增加chanel那一维。
        self.word_embeddings = tf.expand_dims(word_embeddings, -1)

    def convAndMaxpool_op(self):
        pooled_outputs = []
        # filter_sizes = [3, 4, 5]
        filter_sizes = [2, 3 ,4]
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.vocab_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.word_embeddings,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sen_max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                self.num_filters_total = self.num_filters * len(filter_sizes)
                
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
                self.h = tf.nn.dropout(h_pool_flat, self.dropout)

    def dense_op(self):
        W = tf.get_variable(
            "W",
            shape=[self.num_filters_total, self.num_class],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="b")

        self.logits = tf.nn.xw_plus_b(self.h, W, b, name="scores")

    def loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.by))

    def optimizer_op(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)


    def init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess, summary_path):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(summary_path, sess.graph)

    def evaluate(self):
        predictions = tf.argmax(self.logits, 1, name="predictions")
        actuals = tf.argmax(self.by, 1)
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)), tf.float32))
        tn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ), tf.float32))
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)), tf.float32))
        fn = tf.reduce_sum(tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)), tf.float32
        ))

        tpr = tp / (tp + fn)
        fpr = fp / (tp + fn)


        self.accuracy = (tp + tn) / (tp + fp + fn + tn)
        self.recall = tpr
        self.precision = tp / (tp + fp)
        self.f1 = (2 * (self.precision * self.recall)) / (self.precision + self.recall)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('recall', self.recall)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('f1_score', self.f1)
