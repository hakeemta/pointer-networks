import numpy as np

def order(X, by_idx=False):
    # Are these the same?!
    # Ip => Index I occupying pth position
    # NB: with padding, offset is 1
    y = X.argsort() + 1
    padding = np.zeros(shape=(y.shape[0], 1))
    y = np.concatenate((y,  padding), axis=1)

    if not by_idx:
        return y

    # Pi => Position P for ith index
    return y.argsort()


def gen_seq(**kwargs):
    X = np.random.randint(**kwargs)

    def _order(order_by_idx=False):
        y = order(X, order_by_idx)
        return X, y

    return _order


def gen_data(N, seq_len, order_by_idx=False, low=1, high=100):
    seq_gen = gen_seq(low=low, high=high, size=(N, seq_len))

    X, y = seq_gen(order_by_idx=False)
    return X, y


def gen_jagged_data(N, max_seq_len, order_by_idx=False, low=1, high=100):
    seq_lens = np.random.randint(1, max_seq_len+1, size=(N) )
    seq_lens_counts = np.unique(seq_lens, return_counts=True)

    all_X = []
    all_y = []
    for seq_len, n in zip(*seq_lens_counts):
        X, y = gen_data(n, seq_len, order_by_idx, low, high)
        all_X += X.tolist()
        all_y += y.tolist()

    return all_X, all_y

