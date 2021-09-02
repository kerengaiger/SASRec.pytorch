import sys
import copy
import pandas as pd
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.choice(list(user_train.keys()), 1)[0]
        while len(user_train[user]) <= 1: user = np.random.choice(list(user_train.keys()), 1)[0]

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, split_char):
    print(fname)
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    items_lst = []
    for line in f:
        u, i = line.rstrip().split(split_char)
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if i not in items_lst:
            items_lst.append(i)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, items_lst]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, cnfg):
    [train, valid, test, _, _, items_list] = copy.deepcopy(dataset)

    users_lst = list(train.keys())

    NDCG = 0.0
    HT = 0.0
    test_user = 0.0

    users = users_lst

    users_test = []
    items_test = []
    preds_test = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([cnfg['maxlen']], dtype=np.int32)
        idx = cnfg['maxlen'] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # rated = set(train[u])
        # rated.add(0)
        item_idx = test[u]
        item_idx = item_idx + list(set(items_list) - set(item_idx))

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()
        users_test.append(u)
        items_test.append(test[u][0])
        preds_test.append(rank)

        test_user += 1

        if rank < 20:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if test_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / test_user, HT / test_user, pd.DataFrame([users_test, items_test, preds_test]).T


# evaluate on val set
def evaluate_valid(model, dataset, cnfg):
    [train, valid, _, _, _, items_list] = copy.deepcopy(dataset)
    users_lst = list(train.keys())

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    users = users_lst
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([cnfg['maxlen']], dtype=np.int32)
        idx = cnfg['maxlen'] - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        item_idx = item_idx + list(np.random.choice(list(set(items_list) - rated), 100))

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 20:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user