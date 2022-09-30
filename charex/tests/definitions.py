from numba import njit
import numpy as np


class ComparisonOperators:
    @staticmethod
    @njit(nogil=True, cache=True)
    def char_equal(x1, x2):
        return np.char.equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_not_equal(x1, x2):
        return np.char.not_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_greater(x1, x2):
        return np.char.greater(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_greater_equal(x1, x2):
        return np.char.greater_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_less(x1, x2):
        return np.char.less(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_less_equal(x1, x2):
        return np.char.less_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_compare_chararrays(a1, b1, cmp, rstrip):
        return np.char.compare_chararrays(a1, b1, cmp, rstrip)


class StringInformation:
    @staticmethod
    @njit(nogil=True, cache=True)
    def char_count(a, sub, start=0, end=None):
        return np.char.count(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def str_len(a):
        return np.char.str_len(a)
