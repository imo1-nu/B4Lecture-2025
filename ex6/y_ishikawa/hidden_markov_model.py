"""Apply HMM to data and visualize."""

import argparse
import pickle
import time
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


class Models(TypedDict):
    """Dictionary containing parameters for a collection of models.

    Attributes
    ----------
    PI : list[np.ndarray], each with shape=(l, 1)
        List of initial state probability vectors for each model.

    A : list[np.ndarray], each with shape=(l, l)
        List of state transition matrices for each model.

    B : list[np.ndarray], each with shape=(l, n)
        List of emission probability matrices for each model.

    """

    PI: list[np.ndarray]  # shape=(k, l, 1)
    A: list[np.ndarray]  # shape=(k, l, l)
    B: list[np.ndarray]  # shape=(k, l, n)


class Data(TypedDict):
    """Data container for model outputs and associated information.

    Attributes
    ----------
    answer_models : np.ndarray, shape=(p,)
        Array indicating the index of the correct model for each data point.

    output : list[np.ndarray], each with shape=(t,)
        List of observed output sequences.

    models : Models
        Dictionary containing the model parameters.

    """

    answer_models: np.ndarray  # shape=(p,)
    output: list[np.ndarray]  # shape=(p, t)
    models: Models


class HMM:
    """Hidden Markov Model (HMM) for evaluating sequence likelihoods and decoding.

    Attributes
    ----------
    A : list[np.ndarray], each with shape=(l, l)
        Transition matrices for each model.

    B : list[np.ndarray], each with shape=(l, n)
        Emission probability matrices for each model.

    PI : list[np.ndarray], each with shape=(l, 1)
        Initial state probability vectors for each model.

    model_num : int
        Number of models.

    state_num : int
        Number of hidden states.

    symbol_num : int
        Number of observable symbols.

    """

    def __init__(self, models: Models) -> None:
        """Initialize the HMM with a given set of models.

        Parameters
        ----------
        models : Models
            Dictionary containing HMM parameters for multiple models.

        """
        self.A = models["A"]
        self.B = models["B"]
        self.PI = models["PI"]
        self.model_num = len(self.A)
        self.state_num, self.symbol_num = self.B[0].shape

    def forward(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Compute the likelihood of observation sequences using the forward algorithm.

        Parameters
        ----------
        outputs : list[np.ndarray], each with shape=(t,)
            List of observed output sequences.

        Returns
        -------
        likelihood : np.ndarray, shape=(output_num, model_num)
            Array containing the likelihood of each output sequence for each model.

        """
        output_num = len(outputs)
        likelihood = np.zeros((output_num, self.model_num))

        for output_idx, output in enumerate(outputs):
            for model_idx, (A, B, PI) in enumerate(
                zip(self.A, self.B, self.PI, strict=True)
            ):
                alpha = PI.reshape(-1)
                for i, out in enumerate(output):
                    if i != 0:
                        alpha = alpha @ A
                    alpha = alpha * B[:, out]

                likelihood[output_idx, model_idx] = np.sum(alpha)

        return likelihood

    def viterbi(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Compute the likelihood of observation sequences using the viterbi algorithm.

        Parameters
        ----------
        outputs : list[np.ndarray], each with shape=(t,)
            List of observed output sequences.

        Returns
        -------
        likelihood : np.ndarray, shape=(output_num, model_num)
            Array containing the likelihood of each output sequence for each model.

        """
        output_num = len(outputs)
        likelihood = np.zeros((output_num, self.model_num))

        for output_idx, output in enumerate(outputs):
            for model_idx, (A, B, PI) in enumerate(
                zip(self.A, self.B, self.PI, strict=True)
            ):
                delta = PI.reshape(-1)
                for i, out in enumerate(output):
                    if i != 0:
                        delta = np.max(delta.reshape(-1, 1) * A, axis=0)
                    delta = delta * B[:, out]

                likelihood[output_idx, model_idx] = np.max(delta)

        return likelihood


class NameSpace:
    """Configuration namespace for HMM processing parameters.

    Parameters
    ----------
    input_paths: list[Path]
        List of paths to input data.
    output_dir : Path
        Path where the processed output will be saved.

    """

    input_paths: list[Path]
    output_dir: Path


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths parameters.

    """
    # data path
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../").resolve()
    DATA1_PATH = DATA_DIR / "data1.pickle"
    DATA2_PATH = DATA_DIR / "data2.pickle"
    DATA3_PATH = DATA_DIR / "data3.pickle"
    DATA4_PATH = DATA_DIR / "data4.pickle"

    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output result directory.",
    )
    parser.add_argument(
        "--input_paths",
        type=Path,
        nargs="*",
        default=[DATA1_PATH, DATA2_PATH, DATA3_PATH, DATA4_PATH],
        help="Path to input data.",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    input_paths = args.input_paths
    output_dir = args.output_dir

    # make directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        # load data
        data = pickle.load(open(input_path, "rb"))
        hmm = HMM(data["models"])

        # forward algorithm
        forward_start = time.perf_counter()
        forward_likelihood = hmm.forward(data["output"])
        forward_time = time.perf_counter() - forward_start

        # viterbi algorithm
        viterbi_start = time.perf_counter()
        viterbi_likelihood = hmm.viterbi(data["output"])
        viterbi_time = time.perf_counter() - viterbi_start

        # prediction
        forward_prediction = np.argmax(forward_likelihood, axis=1)
        viterbi_prediction = np.argmax(viterbi_likelihood, axis=1)

        # calculate confusion matrix
        forward_cm = confusion_matrix(data["answer_models"], forward_prediction)
        viterbi_cm = confusion_matrix(data["answer_models"], viterbi_prediction)

        # calculate accuracy
        forward_acc = accuracy_score(data["answer_models"], forward_prediction) * 100
        viterbi_acc = accuracy_score(data["answer_models"], viterbi_prediction) * 100

        # plot forward confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=forward_cm)
        disp1.plot(ax=axes[0], cmap="Greys", colorbar=False)
        axes[0].set_title(
            f"Forward algorithm\n(Acc. {forward_acc}%, Time. {forward_time:.3f}s)"
        )

        # plot viterbi confusion matrix
        disp2 = ConfusionMatrixDisplay(confusion_matrix=viterbi_cm)
        disp2.plot(ax=axes[1], cmap="Greys", colorbar=False)
        axes[1].set_title(
            f"Viterbi algorithm\n(Acc. {viterbi_acc}%, Time. {viterbi_time:.3f}s)"
        )
        plt.savefig(output_dir / input_path.with_suffix(".png").name)
        plt.show()
        plt.close()
