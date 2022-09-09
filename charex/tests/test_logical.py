"""Test numpy logical operators"""

from charex.tests.definitions import LogicalOperators
from charex.tests.support import pack_arguments, StandardTest
import numpy as np


def main(method='test'):
    np.random.seed(1)
    ch = LogicalOperators()

    length = 10_000
    p = (1/length) * 10
    a = np.random.choice([0, 1], length * length, p=[1.-p, p])
    b = a[:2000].reshape(-1, 2)
    c = a[:2000].reshape(2, -1)
    d = a[:8000].reshape(8, -1)
    e = a[:8000].reshape(-1, 8)
    f = a.reshape(length, length)

    g = a[:length * (length // 2)].reshape(length // 2, length)
    g[:, 0] = g[0, :] = 0

    h = a[:length * (length // 2)].reshape(length, length // 2)
    h[:, 0] = h[0, :] = 0

    arrays = (
        b,
        b.astype('bool'),
        b.astype('int8'),
        b.astype('int16'),
        b.astype('int32'),
        b.astype('int64'),
        b.astype('float32'),
        b.astype('float64'),
        c, d, e, f, g, h
    )
    args = [(None, 0, 1)]
    arg_pack = pack_arguments(arrays, args)
    StandardTest(ch.numba_any, np.any, arg_pack).run(method)

    arrays = (
        a.astype('bool'),
        a.astype('int8'),
        a.astype('int16'),
        a.astype('int32'),
        a.astype('int64'),
        a.astype('float32'),
        a.astype('float64')
    )
    args = [(None,)]
    arg_pack = pack_arguments(arrays, args)
    StandardTest(ch.numba_any, np.any, arg_pack).run(method)


if __name__ == '__main__':
    main(method='graph')
