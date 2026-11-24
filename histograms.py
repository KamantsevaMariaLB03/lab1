import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_matrix_histograms() -> None:
    # Генерируем матрицу побольше (например, 200x200), чтобы гистограммы были наглядными
    data = RandomMatrixInput(rows=200, columns=200, mean=0.0, std=1.0, seed=42)
    stats = matrix_statistics(data)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Гистограммы статистик по строкам и столбцам")

    # Средние по строкам
    axes[0, 0].hist(stats.row_means, bins=20, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Средние значения строк")

    # Средние по столбцам
    axes[0, 1].hist(stats.column_means, bins=20, color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Средние значения столбцов")

    # Дисперсии по строкам
    axes[1, 0].hist(stats.row_variances, bins=20, color="salmon", edgecolor="black")
    axes[1, 0].set_title("Дисперсии строк")

    # Дисперсии по столбцам
    axes[1, 1].hist(stats.column_variances, bins=20, color="plum", edgecolor="black")
    axes[1, 1].set_title("Дисперсии столбцов")

    plt.tight_layout()
    plt.savefig("histograms.png") # Сохранит картинку в папку проекта
    plt.show()


if __name__ == "__main__":
    plot_matrix_histograms()