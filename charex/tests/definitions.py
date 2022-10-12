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
    # Occurrence Methods
    @staticmethod
    @njit(nogil=True, cache=True)
    def char_count(a, sub, start=0, end=None):
        return np.char.count(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_endswith(a, suffix, start=0, end=None):
        return np.char.endswith(a, suffix, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_startswith(a, prefix, start=0, end=None):
        return np.char.startswith(a, prefix, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_find(a, sub, start=0, end=None):
        return np.char.find(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_index(a, sub, start=0, end=None):
        return np.char.index(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_rfind(a, sub, start=0, end=None):
        return np.char.rfind(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_rindex(a, sub, start=0, end=None):
        return np.char.rindex(a, sub, start, end)

    # Property Methods
    @staticmethod
    @njit(nogil=True, cache=True)
    def char_str_len(a):
        return np.char.str_len(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isalpha(a):
        return np.char.isalpha(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isalnum(a):
        return np.char.isalnum(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isdecimal(a):
        return np.char.isdecimal(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isdigit(a):
        return np.char.isdigit(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_islower(a):
        return np.char.islower(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isnumeric(a):
        return np.char.isnumeric(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isspace(a):
        return np.char.isspace(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_istitle(a):
        return np.char.istitle(a)

    @staticmethod
    @njit(nogil=True, cache=True)
    def char_isupper(a):
        return np.char.isupper(a)
