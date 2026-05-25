from numba import njit
import numpy as np


class ComparisonOperators:
    @staticmethod
    @njit(nogil=True, cache=False)
    def char_equal(x1, x2):
        return np.char.equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_not_equal(x1, x2):
        return np.char.not_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_greater(x1, x2):
        return np.char.greater(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_greater_equal(x1, x2):
        return np.char.greater_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_less(x1, x2):
        return np.char.less(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_less_equal(x1, x2):
        return np.char.less_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_compare_chararrays(a1, b1, cmp, rstrip):
        return np.char.compare_chararrays(a1, b1, cmp, rstrip)


class StringsComparisonOperators:
    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_equal(x1, x2):
        return np.strings.equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_not_equal(x1, x2):
        return np.strings.not_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_greater(x1, x2):
        return np.strings.greater(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_greater_equal(x1, x2):
        return np.strings.greater_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_less(x1, x2):
        return np.strings.less(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_less_equal(x1, x2):
        return np.strings.less_equal(x1, x2)

    @staticmethod
    @njit(nogil=True, cache=False)
    def numpy_equal(x1, x2):
        return np.equal(x1, x2)


class StringsInformation:
    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_count(a, sub, start=0, end=None):
        return np.strings.count(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_endswith(a, suffix, start=0, end=None):
        return np.strings.endswith(a, suffix, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_startswith(a, prefix, start=0, end=None):
        return np.strings.startswith(a, prefix, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_find(a, sub, start=0, end=None):
        return np.strings.find(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_index(a, sub, start=0, end=None):
        return np.strings.index(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_rfind(a, sub, start=0, end=None):
        return np.strings.rfind(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_rindex(a, sub, start=0, end=None):
        return np.strings.rindex(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_str_len(a):
        return np.strings.str_len(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isalpha(a):
        return np.strings.isalpha(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isalnum(a):
        return np.strings.isalnum(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isdecimal(a):
        return np.strings.isdecimal(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isdigit(a):
        return np.strings.isdigit(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_islower(a):
        return np.strings.islower(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isnumeric(a):
        return np.strings.isnumeric(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isspace(a):
        return np.strings.isspace(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_istitle(a):
        return np.strings.istitle(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def strings_isupper(a):
        return np.strings.isupper(a)


class StringInformation:
    # Occurrence Methods
    @staticmethod
    @njit(nogil=True, cache=False)
    def char_count(a, sub, start=0, end=None):
        return np.char.count(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_endswith(a, suffix, start=0, end=None):
        return np.char.endswith(a, suffix, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_startswith(a, prefix, start=0, end=None):
        return np.char.startswith(a, prefix, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_find(a, sub, start=0, end=None):
        return np.char.find(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_index(a, sub, start=0, end=None):
        return np.char.index(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_rfind(a, sub, start=0, end=None):
        return np.char.rfind(a, sub, start, end)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_rindex(a, sub, start=0, end=None):
        return np.char.rindex(a, sub, start, end)

    # Property Methods
    @staticmethod
    @njit(nogil=True, cache=False)
    def char_str_len(a):
        return np.char.str_len(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isalpha(a):
        return np.char.isalpha(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isalnum(a):
        return np.char.isalnum(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isdecimal(a):
        return np.char.isdecimal(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isdigit(a):
        return np.char.isdigit(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_islower(a):
        return np.char.islower(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isnumeric(a):
        return np.char.isnumeric(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isspace(a):
        return np.char.isspace(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_istitle(a):
        return np.char.istitle(a)

    @staticmethod
    @njit(nogil=True, cache=False)
    def char_isupper(a):
        return np.char.isupper(a)
