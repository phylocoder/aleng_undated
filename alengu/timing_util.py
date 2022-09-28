""" Timing utils for benchmarking and debugging

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from functools import wraps
from time import perf_counter


def timing(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t_start = perf_counter()
        r = fn(*args, **kwargs)
        t_finish = perf_counter()
        print(fn.__name__, (t_finish - t_start) / 1000000)
        return r

    return wrapper
