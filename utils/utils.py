import numpy as np, pandas as pd, pickle
from sklearn.metrics import roc_auc_score

randomSeed = 88

def get_minibatches_idx(n, batch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def splitIndices(nTotal, ratio, shuffle=False):
    """split index by ratio"""
    assert type(ratio) in (tuple, list), "type of ratio must in (tuple, list)"

    lenAry = [int(nTotal * e) for e in ratio]
    offset = 0
    peice = []
    ary = np.arange(0, nTotal)
    if shuffle:
        ary = np.random.permutation(ary)
        
    for i, len_ in enumerate(lenAry):
        if i < len(lenAry) - 1:
            peice.append(ary[offset : offset + len_])
        # last round
        else:
            peice.append(ary[offset:])
        offset += len_
    return peice

def preview(fpath, chunksize=5, names=None):
    for chunk in pd.read_csv(fpath, chunksize=chunksize, names=names):
        return chunk

def loadPickle(fpath):
    with open(fpath, "rb") as r:
        return pickle.load(r)

def dumpPickle(fpath, obj):
    with open(fpath, "wb") as w:
        pickle.dump(obj, w)


from sklearn.base import BaseEstimator, TransformerMixin
class CounterEncoder(BaseEstimator, TransformerMixin):
    """依照出現頻率進行編碼, 頻率由高到低的index = 0, 1, 2, 3 ..., 以此類推"""
    def fit(self, y):
        counter = pd.Series(y).value_counts()
        self.enc = dict(zip([None] + counter.index.tolist(), range(len(counter) + 1)))
        self.invEnc = dict(zip(self.enc.values(), self.enc.keys()))
        self.classes_ = counter.index.values
        return self

    def transform(self, y):
        return pd.Series(y).map(self.enc).fillna(0).values

    def fit_transform(self, y, **fit_params):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return pd.Series(y).map(self.invEnc).values

class OrderedMapper(CounterEncoder):
    def fit(self, y):
        uniq = pd.Series(y).unique()
        self.enc = dict(zip(uniq, range(len(uniq))))
        self.invEnc = dict(zip(range(len(uniq)), uniq))
        self.classes_ = uniq
        return self


from collections import defaultdict, Counter, OrderedDict

def split_ratings(data, pos_thres=4, testRatio=0.3):
    """依照比例切割movielens train test資料"""
    tr, te = [], []
    for u, df in data.groupby("userId"):
        if len(df) < 5: continue

        pos, neg = df.query("rating >= {}".format(pos_thres)), df.query("rating < {}".format(pos_thres))
        pos_len = int(len(pos) * (1 - testRatio))
        tr_pos = pos[:pos_len]
        te_pos = pos[pos_len:]

        neg_len = int(len(neg) * (1 - testRatio))
        tr_neg = neg[:neg_len]
        te_neg = neg[neg_len:]

        tr.append(tr_pos.append(tr_neg))
        te.append(te_pos.append(te_neg))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

def doMovies(movies):
    """處理 movie: genres 轉換成數字"""
    movies = movies.reset_index(drop=True)
    movies.loc[movies.genres == "(no genres listed)", "genres"] = ""
    movies["genres"] = movies.genres.str.split("\|")
    genresMap = Counter()
    movies.genres.map(genresMap.update)
    om = OrderedMapper().fit([e[0] for e in genresMap.most_common()])
    movies["genres"] = movies.genres.map(lambda lst: [om.enc[e] for e in lst])
    return movies, om


def auc_mean(y, pred_mat):
    """mean auc score of each user"""
    tot_auc, cnt = 0, 0
    for i in range(len(y)):
        nnz = y[i].nonzero()[0]
        if len(nnz) <= 1: continue

        labels = y[i][nnz]
        labels = (labels >= 4).astype(int)
        pred = pred_mat[i][nnz]
        if (labels == 1).all() or (labels == 0).all(): continue

        # print(i, ":", labels, predProba[i][nnz])
        tot_auc += roc_auc_score(labels, pred)
        cnt += 1
    return tot_auc / cnt


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best
