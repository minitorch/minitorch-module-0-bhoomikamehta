import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """
    Generate a list of N random points in 2D space.

    Args:
        N (int): The number of points to generate.

    Returns:
        List[Tuple[float, float]]: A list of tuples, where each tuple contains
        two random float values representing (x1, x2) coordinates.
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    A class to represent a 2D dataset with labels.

    Attributes:
        N (int): Number of points in the dataset.
        X (List[Tuple[float, float]]): The list of points (x1, x2).
        y (List[int]): List of labels corresponding to the points.
    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generate a simple dataset where points are classified by whether x1 < 0.5.

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points are labeled 1 if x1 < 0.5, otherwise 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generate a dataset where points are classified by whether x1 + x2 < 0.5.

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points are labeled 1 if x1 + x2 < 0.5, otherwise 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generate a dataset where points are classified based on specific x1 intervals.

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points are labeled 1 if x1 < 0.2 or x1 > 0.8, otherwise 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generate an XOR-like dataset where points are classified by their quadrant relative to (0.5, 0.5).

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points are labeled 1 if they are in opposite quadrants,
               otherwise 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generate a circular dataset where points are classified by their distance from the center (0.5, 0.5).

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points are labeled 1 if they are outside a radius of 0.1 from the center, otherwise 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generate a spiral dataset where points form two spirals, each labeled differently.

    Args:
        N (int): The number of points to generate.

    Returns:
        Graph: A dataset where points in different spirals are labeled 0 and 1.
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
