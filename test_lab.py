import numpy as np

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)

from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)


def test_count_vowels():
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("AEIOUaeiou")) == 10
    assert count_vowels(TextInput("")) == 0
    assert count_vowels(TextInput("bcdfg123!")) == 0


def test_has_unique_characters():
    assert has_unique_characters(TextInput("abcdef")) is True
    assert has_unique_characters(TextInput("hello")) is False
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("aA")) is True


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(8)) == 1
    assert count_one_bits(PositiveIntegerInput(1023)) == 10


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1


def test_mse():
    assert mse(VectorPairInput(predicted=(1, 2, 3), expected=(1, 2, 3))) == 0.0
    assert mse(VectorPairInput(predicted=(1, 2), expected=(2, 4))) == 2.5
    assert mse(VectorPairInput(predicted=(0,), expected=(5,))) == 25.0


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(16)) == "(2**4)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(7)) is True
    assert is_balanced_number(PositiveIntegerInput(12)) is True
    assert is_balanced_number(PositiveIntegerInput(1221)) is True


def test_sum_prod():
    matrices = np.array((
        ((1, 2), (3, 4)),
        ((2, 0), (0, 2)),
    ))
    vectors = np.array((
        ((1,), (1,)),
        ((3,), (4,)),
    ))
    res = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    expected = np.array(((9,), (15,)))
    assert res.shape == (2, 1)
    assert np.array_equal(res, expected)


def test_binarize():
    mat = np.array(((0.1, 0.5), (0.6, 1.0)))
    res = binarize(BinarizeInput(matrix=mat, threshold=0.5))
    expected = np.array(((0, 0), (1, 1)))
    assert np.array_equal(res, expected)
    assert mat.item(0, 0) == 0.1


def test_unique_rows_and_columns():
    mat = np.array(((1, 2, 2), (3, 1, 3)))
    assert unique_rows(MatrixInput(mat)) == [[1, 2], [1, 3]]
    assert unique_columns(MatrixInput(mat)) == [[1, 3], [1, 2], [2, 3]]


def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(rows=4, columns=5, mean=10.0, std=2.0, seed=42))
    assert stats.matrix.shape == (4, 5)
    assert stats.row_means.shape == (4,)
    assert stats.column_means.shape == (5,)
    assert np.allclose(stats.row_means, np.mean(stats.matrix, axis=1))
    assert np.allclose(stats.column_variances, np.var(stats.matrix, axis=0))


def test_chess():
    res = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array(((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1)))
    assert np.array_equal(res, expected)

    single = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    assert np.array_equal(single, np.array(((5,),)))


def test_draw_rectangle():
    img = draw_rectangle(
        RectangleInput(
            width=2,
            height=2,
            image_height=4,
            image_width=4,
            shape_color=(255, 0, 0),
            background_color=(0,0, 0),
        )
    )
    assert img.shape == (4, 4, 3)
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert tuple(img[2, 2]) == (255, 0, 0)


def test_draw_ellipse():
    img = draw_ellipse(
        EllipseInput(
            semi_axis_x=2,
            semi_axis_y=2,
            image_height=6,
            image_width=6,
            shape_color=(0, 255, 0),
            background_color=(255, 255, 255),
        )
    )
    assert img.shape == (6, 6, 3)
    assert tuple(img[0, 0]) == (255, 255, 255)
    assert tuple(img[3, 3]) == (0, 255, 0)


def test_analyze_time_series():
    ts = (1, 5, 2, 4, 1, 5)
    stats = analyze_time_series(TimeSeriesInput(values=ts, window=3))
    assert np.isclose(stats.mean, 3.0)
    assert np.array_equal(stats.local_maxima_indices, np.array((1, 3)))
    assert np.array_equal(stats.local_minima_indices, np.array((2, 4)))
    assert len(stats.moving_average) == 4
    assert np.allclose(stats.moving_average, (8 / 3, 11 / 3, 7/3, 10 / 3))


def test_one_hot():
    res1 = one_hot(OneHotInput(labels=(0, 2, 3, 0)))
    expected1 = np.array((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
    ))
    assert np.array_equal(res1, expected1)

    res2 = one_hot(OneHotInput(labels=(0, 1), class_count=4))
    assert res2.shape == (2, 4)