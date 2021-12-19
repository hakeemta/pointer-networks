import numpy as np

def order(X, by_idx=False):
    # Are these the same?!
    # Ip => Index I occupying pth position
    y = X.argsort()

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


def gen_data(N, seq_len, order_by_idx=False, low=0, high=100):
    seq_gen = gen_seq(low=low, high=high, size=(N, seq_len))

    X, y = seq_gen(order_by_idx=False)
    return X, y

