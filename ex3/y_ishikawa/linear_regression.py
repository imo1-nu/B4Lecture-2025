import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(  # noqa: PLR0913
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Scatter plot",
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    zlabel: str = r"$x_3$",
) -> None:
    """Show a scatter plot from a DataFrame and saves it to `output_path`.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing the data (2D or 3D) to be plotted.

    output_path : Path
        The path where the plot will be saved.

    title : str, optional, default "Scatter plot"
        The title of the scatter plot.

    xlabel : str, optional, default r"$x_1$"
        The label for the x-axis.

    ylabel : str, optional, default r"$x_2$"
        The label for the y-axis.

    zlabel : str, optional, default r"$x_3$"
        The label for the z-axis, used only for 3D plots.
    """
    # reset plt
    fig = plt.figure()

    # 2-dimensions
    if df.shape[1] == 2:  # noqa: PLR2004
        ax = fig.add_subplot(111)
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=".")
    # 3-dimensions
    elif df.shape[1] == 3:  # noqa: PLR2004
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], marker=".")

        ax.set_zlabel(zlabel)

    # set labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    # show graph
    plt.show()


class NameSpace:
    """Configuration namespace for audio processing parameters.

    Parameters
    ----------
    output_path : Path
        Path where the processed output will be saved.
    """

    output_path: Path


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including output path.
    """
    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output result directory.",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    output_path = args.output_path

    # load data
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../data/").resolve()
    DATA1_PATH = DATA_DIR / "data1.csv"
    DATA2_PATH = DATA_DIR / "data2.csv"
    DATA3_PATH = DATA_DIR / "data3.csv"
    df1 = pd.read_csv(DATA1_PATH)
    df2 = pd.read_csv(DATA2_PATH)
    df3 = pd.read_csv(DATA3_PATH)

    # pre-process data
    df1.attrs["title"] = "data1"
    df2.attrs["title"] = "data2"
    df3.attrs["title"] = "data3"
    data = [df1, df2, df3]

    # plot original data
    for df in data:
        title = df.attrs["title"]
        plot_scatter(df, output_path / "scatter_plot" / f"{title}.png", title=title)
