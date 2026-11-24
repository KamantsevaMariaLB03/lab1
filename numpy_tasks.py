import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    m_arr = np.asarray(matrices)
    v_arr = np.asarray(vectors)
    return np.sum(m_arr @ v_arr, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    m_arr = np.asarray(matrix)
    return (m_arr > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    m_arr = np.asarray(matrix)
    return [np.unique(row).tolist() for row in m_arr]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    m_arr = np.asarray(matrix)
    return [np.unique(col).tolist() for col in m_arr.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    dtype = float if isinstance(first, float) or isinstance(second, float) else int
    res = np.full((rows, columns), first, dtype=dtype)
    res[1::2, ::2] = second
    res[::2, 1::2] = second
    return res


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    center_y, center_x = image_height // 2, image_width // 2

    top = max(0, center_y - height // 2)
    bottom = min(image_height, top + height)
    left = max(0, center_x - width // 2)
    right = min(image_width, left + width)

    img[top:bottom, left:right] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    y, x = np.ogrid[:image_height, :image_width]
    center_y, center_x = image_height // 2, image_width // 2

    mask = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    img[mask] = shape_color
    return img

#провалено
def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    ts = np.asarray(values, dtype=float)

    mean = float(np.mean(ts))
    variance = float(np.var(ts))
    std = float(np.std(ts))

    if len(ts) >= 3:
        local_max = np.where((ts[1:-1] > ts[:-2]) & (ts[1:-1] > ts[2:]))[0] + 1
        local_min = np.where((ts[1:-1] < ts[:-2]) & (ts[1:-1] < ts[2:]))[0] + 1
    else:
        local_max = np.array([], dtype=int)
        local_min = np.array([], dtype=int)

    kernel = np.ones(window) / window
    moving_avg = np.convolve(ts, kernel, mode="valid")

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_max,
        local_minima_indices=local_min,
        moving_average=moving_avg,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    y = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = int(np.max(y)) + 1 if y.size > 0 else 0

    res = np.zeros((len(y), class_count), dtype=int)
    if y.size > 0:
        res[np.arange(len(y)), y] = 1
    return res