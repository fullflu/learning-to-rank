#!/usr/bin/env python
# coding: utf-8

import numpy as np
import chainer
import time
from chainer import Variable, optimizers
import chainer.functions as F
import net
import pandas as pd
import six
import argparse

np.random.seed(71)

# calculate ndcg for k elements (y_true must be non-negative and include at least one non-zero element.)
def ndcg(y_true, y_score, k=40):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    nthres = min(y_true.shape[0],k)
    for i in range(nthres):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(nthres):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

# Listnet class
class ListNet(object):
    def __init__(self, n_hidden1 = 200, n_hidden2 = 78, batch_size = 28, max_iter = 1000, n_thres_cand = 40, val_ratio = 0.5, verbose = 10):
        super(ListNet, self).__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.val_ratio = val_ratio
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_thres_cand = n_thres_cand

    def prepare_data(self,filename=None,T=500, dim = 30, train = True, noscore = True):
        if train:
            if filename:
                data = pd.read_csv(filename)
                self.Y = []
                self.X = []
                for t in data.session.unique():
                    dt = data[data.session == t]
                    if dt.score.sum() == 0:
                        continue
                    xt = dt.loc[:,np.in1d(dt.columns,["session","score"],invert=True)]
                    self.X.append(xt.values.astype(np.float32))
                    self.Y.append(dt.score.values.astype(np.float32))
                self.T = len(self.X)
            else:
                # TODO: random sample generator
                print("Error: Sorry, random sample generator has not been implemented. Input filename is necessary!!")
        else:
            if filename:
                data = pd.read_csv(filename)
                if noscore:
                    self.test_X = []
                    for t in data.session.unique():
                        dt = data[data.session == t]
                        if dt.score.sum() == 0:
                            continue
                        xt = dt.loc[:,np.in1d(dt.columns,["session"],invert=True)]
                        self.test_X.append(xt.values.astype(np.float32))                
                else:
                    self.test_X = []
                    self.test_Y = []
                    for t in data.session.unique():
                        dt = data[data.session == t]
                        if dt.score.sum() == 0:
                            continue
                        xt = dt.loc[:,np.in1d(dt.columns,["session","score"],invert=True)]
                        self.test_X.append(xt.values.astype(np.float32))
                        self.test_Y.append(dt.score.values.astype(np.float32))
                    self.test_T = len(self.test_X)                

        perm_all = np.random.permutation(self.T)
        self.train_indices = perm_all[int(self.val_ratio * self.T):]
        self.val_indices = perm_all[:int(self.val_ratio * self.T)]
        self.dim = xt.shape[1]

    def get_loss(self, x_t, y_t):
        x_t = Variable(x_t)
        y_t = Variable(y_t)
        y_t = F.reshape(y_t,(1,y_t.shape[0]))
        # normalize output score to avoid divergence
        y_t = F.normalize(y_t)
        self.model.zerograds()
        pred = self.model(x_t)
        # ---- start loss calculation ----
        pred = F.reshape(pred,(pred.shape[1],pred.shape[0]))
        p_true = F.softmax(F.reshape(y_t,(y_t.shape[0],y_t.shape[1])))
        xm = F.max(pred,axis=1,keepdims = True)
        logsumexp = F.logsumexp(pred,axis=1)
        #xm_broadcast = F.broadcast_to(xm,(xm.shape[0],pred.shape[1]))
        #logsumexp = F.reshape(xm,(xm.shape[0],)) + F.log(F.sum(F.exp(pred-xm_broadcast),axis=1))
        logsumexp = F.broadcast_to(logsumexp,(xm.shape[0],pred.shape[1]))
        loss = -1 * F.sum( p_true * (pred - logsumexp) )
        trainres = ndcg(y_t.data,pred.data,self.n_thres_cand)#,nthres)
        if np.isnan(trainres):
            print y_t.data.max(),y_t.data.min()
        return loss,trainres

    def fit(self,train_val_filename):
        self.prepare_data(filename = train_val_filename)
        # model initialization
        self.model = net.MLPListNet(self.dim, self.n_hidden1, self.n_hidden2)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        # start training
        trainres = 0.0
        traincnt = 0
        for iter_ in range(self.max_iter):
            perm_tr = np.random.permutation(self.train_indices)        
            for batch_idx in six.moves.range(0,self.train_indices.shape[0],self.batch_size):
                loss = 0.
                for t in perm_tr[batch_idx:batch_idx + self.batch_size]:
                    traincnt += 1
                    sorted_idxes = np.argsort(self.Y[t])[::-1]
                    nthres = min(self.n_thres_cand, sorted_idxes.shape[0])
                    x_t = self.X[t][sorted_idxes[:nthres]]
                    y_t = self.Y[t][sorted_idxes[:nthres]]
                    loss_t, trainres_t = self.get_loss(x_t,y_t)
                    loss += loss_t
                    trainres += trainres_t
                loss.backward()
                self.optimizer.update()
            # start validation
            if self.verbose:
                if (iter_ + 1) % self.verbose == 0:
                    print("step:{},train_loss:{}".format(iter_,loss.data))
                    print("train_ndcg:{}".format(trainres/traincnt))
                    trainres = 0.0
                    traincnt = 0
                    if len(self.val_indices) != 0:
                        testres = self.validation()
                        print("valid_ndcg:{}".format(testres/len(self.val_indices)))

    def validation(self):
        testres = 0.0
        for j in self.val_indices:
            sorted_idxes = np.argsort(self.Y[j])[::-1]
            nthres = min(self.n_thres_cand, sorted_idxes.shape[0])
            x_j = Variable(self.X[j][sorted_idxes[:nthres]])
            y_j = Variable(self.Y[j][sorted_idxes[:nthres]])
            y_j = F.reshape(y_j,(1,y_j.shape[0]))
            # normalize output score to avoid divergence
            y_j = F.normalize(y_j)
            pred_j = self.predict(x_j)
            pred_j = F.reshape(pred_j,(pred_j.data.shape[0],))
            testres += ndcg(y_j.data,pred_j.data,self.n_thres_cand)
        return testres

    def predict(self,test_X):
        if test_X.ndim == 2:
            return self.model(test_X)
        else:
            pred = []
            for t,x_t in enumerate(test_X):
                pred_t = self.model(x_t)
                pred.append(pred_t)
            return pred

    def test(self,filename,noscore = True):
        testres = 0
        self.prepare_data(filename = filename, train = False, noscore = noscore)
        if noscore:
            pred = pd.DataFrame()
            for j in range(self.test_T):
                x_j = Variable(self.test_X[j])
                pred_j = self.predict(x_j)
                pred_j = F.reshape(pred_j,(pred_j.data.shape[0],))
                pred = pd.concat([pred,pd.DataFrame(np.sort(pred_j.data)[::-1]).T])
            pred.to_csv("new_results.csv",index=False)
            print("save new_results.csv !")

        else:
            for j in range(self.test_T):
                sorted_idxes = np.argsort(self.test_Y[j])[::-1]
                nthres = min(self.n_thres_cand, sorted_idxes.shape[0])
                x_j = Variable(self.test_X[j][sorted_idxes[:nthres]])
                y_j = Variable(self.test_Y[j][sorted_idxes[:nthres]])
                y_j = F.reshape(y_j,(1,y_j.shape[0]))
                # normalize output score to avoid divergence
                y_j = F.normalize(y_j)
                pred_j = self.predict(x_j)
                pred_j = F.reshape(pred_j,(pred_j.data.shape[0],))
                testres += ndcg(y_j.data,pred_j.data,self.n_thres_cand)
            print("test_ndcg:{}".format(testres / self.test_T))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This script is: lisenet_open.py')
    parser.add_argument('--train_val_filename', '-trvf', type = str,
                        help='train_val_filename')
    parser.add_argument('--test_score_filename', '-tesf', type = str,
                        help='test_score_filename')
    parser.add_argument('--test_noscore_filename', '-tenf', type = str,
                        help='test_noscore_filename')
    parser.add_argument('--verbose', '-ve', default = 10, type=int,
                        help='the number of verbose step')
    parser.add_argument('--max_iter', '-mi', default = 1000, type=int,
                        help='the number of max_iteration')
    parser.add_argument('--val_ratio', '-vr', default = .5, type=float,
                        help='the ratio of validation')
    parser.add_argument('--rank', '-r', default = 40, type=int,
                        help='the number of rank used for evaluation')

    args = parser.parse_args()
    train_val_filename = args.train_val_filename
    test_score_filename = args.test_score_filename
    test_noscore_filename = args.test_noscore_filename
    verbose = args.verbose
    max_iter = args.max_iter
    val_ratio = args.val_ratio
    rank = args.rank

    agent = ListNet(verbose = verbose, max_iter = max_iter, val_ratio = val_ratio, n_thres_cand = rank)
    agent.fit(train_val_filename = train_val_filename)
    if test_score_filename:
        agent.test(filename = test_score_filename, noscore = False)
    if test_noscore_filename:
        agent.test(filename = test_noscore_filename, noscore = True)






