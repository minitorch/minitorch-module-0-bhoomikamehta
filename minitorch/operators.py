"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.
def mul(a, b) -> float:
    """
    Multiply two values.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The product of `a` and `b`.
    """
    return a * b


def id(a) -> float:
    """
    Identity function. Returns the input value unchanged.

    Args:
        a: The input value.

    Returns:
        The same value `a`.
    """
    return a


def add(a, b) -> float:
    """
    Add two values.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The sum of `a` and `b`.
    """
    return a + b


def neg(a) -> float:
    """
    Negate a value.

    Args:
        a: The input value.

    Returns:
        The negated value of `a`.
    """
    return -a


def lt(a, b) -> bool:
    """
    Check if the first value is less than the second value.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        True if `a` is less than `b`, otherwise False.
    """
    return a < b


def eq(a, b) -> bool:
    """
    Check if two values are equal.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        True if `a` is equal to `b`, otherwise False.
    """
    return a == b


def max(a, b) -> float:
    """
    Return the maximum of two values.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The maximum of `a` and `b`.
    """
    return a if a > b else b


def is_close(a, b) -> bool:
    """
    Check if two values are close to each other within a small tolerance.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        True if the absolute difference between `a` and `b` is less than 1e-2, otherwise False.
    """
    return abs(a - b) < 1e-2


def sigmoid(a) -> float:
    """
    Compute the sigmoid of a value.

    Args:
        a: The input value.

    Returns:
        The sigmoid of `a`, defined as 1 / (1 + exp(-a)).
    """
    return 1 / (1 + math.exp(-a))


def relu(a) -> float:
    """
    Compute the ReLU (Rectified Linear Unit) of a value.

    Args:
        a: The input value.

    Returns:
        `a` if `a` is greater than 0, otherwise 0.
    """
    return a if a > 0 else 0


def log(a) -> float:
    """
    Compute the natural logarithm of a value.

    Args:
        a: The input value.

    Returns:
        The natural logarithm of `a`.
    """
    return math.log(a)


def exp(a) -> float:
    """
    Compute the exponential of a value.

    Args:
        a: The input value.

    Returns:
        The exponential of `a`, i.e., e^a.
    """
    return math.exp(a)


def inv(a) -> float:
    """
    Compute the multiplicative inverse of a value.

    Args:
        a: The input value.

    Returns:
        The multiplicative inverse of `a`, i.e., 1/a.
    """
    return 1 / a


def log_back(a, grad) -> float:
    """
    Compute the gradient of the logarithm with respect to its input.

    Args:
        a: The input value.
        grad: The gradient of the output with respect to some loss.

    Returns:
        The gradient of the input, computed as `grad / a`.
    """
    return grad / a


def inv_back(a, grad) -> float:
    """
    Compute the gradient of the multiplicative inverse with respect to its input.

    Args:
        a: The input value.
        grad: The gradient of the output with respect to some loss.

    Returns:
        The gradient of the input, computed as `-grad / (a^2)`.
    """
    return -grad / (a**2)


def relu_back(a, grad) -> float:
    """
    Compute the gradient of the ReLU function with respect to its input.

    Args:
        a: The input value.
        grad: The gradient of the output with respect to some loss.

    Returns:
        `grad` if `a` is greater than 0, otherwise 0.
    """
    return grad if a > 0 else 0


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3


# Small practice library of elementary higher-order functions.
def map(fn: Callable, lst: Iterable) -> list[float]:
    """
    Apply a function to each element in a list.

    Args:
        fn: A function to apply to each element.
        lst: An iterable of elements.

    Returns:
        A list of results after applying `fn` to each element in `lst`.
    """
    return [fn(x) for x in lst]


def custom_zip(ls1: Iterable[float], ls2: Iterable[float]) -> list[tuple[float, float]]:
    """
    Zip together multiple iterables, truncating to the length of the shortest.

    Args:
        *iterables: Any number of iterables.

    Returns:
        A list of tuples, where each tuple contains one element from each iterable at the same position.
    """
    zipped = []
    a, b = iter(ls1), iter(ls2)

    while True:
        try:
            i1 = next(a)
            i2 = next(b)
            zipped.append((i1, i2))
        except StopIteration:
            break

    return zipped


def zipWith(f: Callable, lst1: Iterable, lst2: Iterable) -> list[float]:
    """
    Apply a binary function to pairs of elements from two lists.

    Args:
        f: A function that takes two arguments.
        lst1: The first iterable.
        lst2: The second iterable.

    Returns:
        A list of results after applying `f` to each pair of elements from `lst1` and `lst2`.
    """
    return [f(x, y) for x, y in custom_zip(lst1, lst2)]


def custom_reduce(f: Callable, lst: Iterable, initial) -> float:
    """
    Reduce a list to a single value by iteratively applying a binary function.

    Args:
        f: A function that takes two arguments.
        lst: An iterable to be reduced.
        initial: The initial value to start the reduction.

    Returns:
        The reduced value after applying `f` across the elements of `lst`.
    """
    for i in lst:
        initial = f(initial, i)
    return initial


def negList(lst: Iterable) -> list[float]:
    """
    Apply the negation function to each element in a list.

    Args:
        lst: An iterable of elements.

    Returns:
        A list of negated values.
    """
    return map(neg, lst)


def addLists(lst1: Iterable, lst2: Iterable) -> list[float]:
    """
    Element-wise addition of two lists.

    Args:
        lst1: The first iterable.
        lst2: The second iterable.

    Returns:
        A list containing the sum of corresponding elements from `lst1` and `lst2`.
    """
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable) -> float:
    """
    Compute the sum of all elements in a list.

    Args:
        lst: An iterable of elements.

    Returns:
        The sum of all elements in `lst`.
    """
    return custom_reduce(add, lst, 0.0)


def prod(lst: Iterable) -> float:
    """
    Compute the product of all elements in a list.

    Args:
        lst: An iterable of elements.

    Returns:
        The product of all elements in `lst`.
    """
    return custom_reduce(mul, lst, 1.0)


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
