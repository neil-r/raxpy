"""TODO Explain Module """

from functools import wraps


def range_check(*a, **k):
    """
    Checks the lower and upper ranges and if none found gives values

    Arguments
    ---------
    *a : Tuple
        **Explanation**
    **k : dict[str, any]
        **Explanation**

    """
    lower = 10 if "lower" not in k else k["lower"]
    upper = 100 if "upper" not in k else k["upper"]

    def real_decorator(f):
        """
        TODO Explain the Function

        Arguments
        ---------
        f
            **Explanation**

        Returns
        -------
        real_decorator :
            **Explanation**

        """

        @wraps(f)
        def wrapper(*args, **kwargs):
            for arg in args:
                if arg < lower or arg > upper:
                    raise ValueError(f"{arg} is not in [{lower}, {upper}]")

            for k, v in kwargs.items():
                if v < lower or v > upper:
                    raise ValueError(f"{k}={v} is not in [{lower}, {upper}]")

            retval = f(*args, **kwargs)
            return retval

        return wrapper

    return real_decorator


@range_check(lower=-10, upper=100)
def add_one(a):
    """ Funciton adds 1 to input """
    return a + 1


print(add_one(10))
print(add_one(-10))
print(add_one(-100))
