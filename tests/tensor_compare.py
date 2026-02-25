import torch
import numpy as np
from typing import Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from argparse import ArgumentParser


mach_epsilon_fp32 = 2**-23
mach_epsilon_bf16 = 2**-7


def parse_args():
    parser = ArgumentParser(description="Tensor comparison script")
    parser.add_argument(
        "--tensor_file1",
        type=str,
        required=True,
        help="Path to the first tensor file",
    )
    parser.add_argument(
        "--tensor_file2",
        type=str,
        required=True,
        help="Path to the second tensor file",
    )
    parser.add_argument(
        "--tensor_file3",
        type=str,
        default=None,
        help="Path to the third tensor file (optional, for comparing three tensors)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of differences",
    )
    parser.add_argument(
        "--figure_format",
        type=str,
        default="png",
        choices=["png", "html"],
        help="Format for saving figures",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Folder to save output files",
    )
    parser.add_argument(
        "--output_prefix", type=str, default="", help="Prefix for output files"
    )

    parser.add_argument(
        "--basic_self_test",
        action="store_true",
        help="Run basic self-test for API usage",
    )
    return parser.parse_args()


def _convert_to_numpy_fp64(tensor):
    """Convert a tensor to numpy with dtype float64."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    tensor = tensor.reshape(-1)
    return tensor.astype(np.float64)


def _visualize_differences_one_series(
    diff, output_folder, figure_format, output_prefix=""
):
    """
    visualize the diff in histogram and save it to the output file.
    """
    os.makedirs(output_folder, exist_ok=True)

    title = f"min {np.min(diff):.4f}, max {np.max(diff):.4f}, mean {np.mean(diff):.4f}, std {np.std(diff):.4f}"

    fig = px.histogram(
        x=diff.flatten(),
        title=title,
        labels={"x": "Difference"},
        nbins=100,
    )

    if figure_format == "html":
        fig.write_html(
            f"{output_folder}/{output_prefix}_histogram.html",
            include_plotlyjs="cdn",
        )
    else:
        fig.write_image(
            f"{output_folder}/{output_prefix}_histogram.{figure_format}",
            format=figure_format,
            width=800,
            height=600,
        )


def _visualize_differences_two_series(
    elem_diff1,
    elem_diff2,
    output_folder,
    output_prefix,
    figure_format,
    max_xaxes_range=1,
    num_quantiles=10000,
):
    """Plot the elem_diff1 and elem_diff2 in the same histogram figure in different colors.
    Check if two series have the same distribution using the qq-plot from scipy
    """

    # plot both series in the same histogram
    os.makedirs(output_folder, exist_ok=True)
    title = (
        f"series 1: min {np.min(elem_diff1):.4f}, max {np.max(elem_diff1):.4f}, mean {np.mean(elem_diff1):.4f}, std {np.std(elem_diff1):.4f} \n"
        f"series 2: min {np.min(elem_diff2):.4f}, max {np.max(elem_diff2):.4f}, mean {np.mean(elem_diff2):.4f}, std {np.std(elem_diff2):.4f}"
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Histogram", "QQ-Plot"),
        horizontal_spacing=0.1,
    )

    # add the histogram element-wise differences
    figure.add_trace(
        go.Histogram(
            x=elem_diff1.flatten(),
            name="Error series 1",
            opacity=0.75,
            marker_color="blue",
            nbinsx=100,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=elem_diff2.flatten(),
            name="Error series 2",
            opacity=0.75,
            marker_color="red",
            nbinsx=100,
        ),
        row=1,
        col=1,
    )

    # compute the quantiles for qq-plot
    n = min(len(elem_diff1), len(elem_diff2), num_quantiles)
    p = np.linspace(0, 1, n)
    sorted_diff1 = np.sort(elem_diff1.flatten())
    sorted_diff2 = np.sort(elem_diff2.flatten())
    q1 = np.quantile(sorted_diff1, p)
    q2 = np.quantile(sorted_diff2, p)

    figure.add_trace(
        go.Scatter(
            x=q1,
            y=q2,
            mode="markers",
            name="QQ-Plot",
            marker=dict(
                color="green",
                size=5,
                opacity=0.6,
            ),
        ),
        row=1,
        col=2,
    )

    min_val = min(q1.min(), q2.min())
    max_val = max(q1.max(), q2.max())
    ref_line = np.linspace(min_val, max_val, 100)

    figure.add_trace(
        go.Scatter(
            x=ref_line,
            y=ref_line,
            mode="lines",
            name="45-degree line",
            line=dict(
                color="red",
                dash="dash",
            ),
        ),
        row=1,
        col=2,
    )

    figure.update_layout(
        title=title,
        showlegend=True,
        barmode="overlay",
        width=1200,
        height=600,
    )

    mean_val = np.mean(elem_diff1)
    std_val = np.max(
        (np.std(elem_diff1), np.std(elem_diff2))
    )  # Use the larger std for range
    figure.update_xaxes(
        title_text="Difference",
        row=1,
        col=1,
        range=[mean_val - 5 * std_val, mean_val + 5 * std_val],
    )

    if figure_format == "html":
        figure.write_html(
            f"{output_folder}/{output_prefix}_qqplot_hist.html",
            include_plotlyjs="cdn",
        )
    else:
        figure.write_image(
            f"{output_folder}/{output_prefix}_qqplot_hist.{figure_format}",
            format=figure_format,
            width=1200,
            height=600,
        )


def compare_3tensors(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
    tensor3: Union[torch.Tensor, np.ndarray],
    visualize=False,
    figure_format="png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
):
    """Consider the tensor1 is the reference tensor.
    take the difference between tensor2 and tensor1, and tensor3 and tensor1.

    """
    tensor1 = _convert_to_numpy_fp64(tensor1)
    tensor2 = _convert_to_numpy_fp64(tensor2)
    tensor3 = _convert_to_numpy_fp64(tensor3)

    diff_2_1 = tensor2 - tensor1
    diff_3_1 = tensor3 - tensor1

    fro_norm_err_2_1 = np.linalg.norm(diff_2_1) / np.linalg.norm(tensor1)
    fro_norm_err_3_1 = np.linalg.norm(diff_3_1) / np.linalg.norm(tensor1)
    inf_norm_err_2_1 = np.linalg.norm(diff_2_1, ord=np.inf) / np.linalg.norm(
        tensor1, ord=np.inf
    )
    inf_norm_err_3_1 = np.linalg.norm(diff_3_1, ord=np.inf) / np.linalg.norm(
        tensor1, ord=np.inf
    )

    element_wise_rel_err_2_1 = np.where(
        tensor1 != 0, np.abs(diff_2_1) / np.abs(tensor1), 0
    )
    element_wise_rel_err_3_1 = np.where(
        tensor1 != 0, np.abs(diff_3_1) / np.abs(tensor1), 0
    )

    if visualize:
        # plot the differences in histogram and qq-plot
        assert (
            output_folder is not None
        ), "Output folder must be specified for visualization."

        _visualize_differences_two_series(
            diff_2_1,
            diff_3_1,
            output_folder=output_folder,
            output_prefix=output_prefix + "_diff",
            figure_format=figure_format,
        )

        _visualize_differences_two_series(
            element_wise_rel_err_2_1,
            element_wise_rel_err_3_1,
            output_folder=output_folder,
            output_prefix=output_prefix + "_rel_diff",
            figure_format=figure_format,
        )

    return {
        "abs_max_elem_diff_2_1": np.max(np.abs(diff_2_1)),
        "abs_mean_elem_diff_2_1": np.mean(np.abs(diff_2_1)),
        "rel_max_elem_diff_2_1": np.max(element_wise_rel_err_2_1),
        "rel_mean_elem_diff_2_1": np.mean(element_wise_rel_err_2_1),
        "rel_matrix_fro_norm_dist_2_1": fro_norm_err_2_1,
        "rel_matrix_inf_norm_dist_2_1": inf_norm_err_2_1,
        "abs_max_elem_diff_3_1": np.max(np.abs(diff_3_1)),
        "abs_mean_elem_diff_3_1": np.mean(np.abs(diff_3_1)),
        "rel_max_elem_diff_3_1": np.max(element_wise_rel_err_3_1),
        "rel_mean_elem_diff_3_1": np.mean(element_wise_rel_err_3_1),
        "rel_matrix_fro_norm_dist_3_1": fro_norm_err_3_1,
        "rel_matrix_inf_norm_dist_3_1": inf_norm_err_3_1,
    }


def compare_2tensors(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
    visualize=False,
    figure_format="png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
):
    """
    Assume the tensor1 is the reference tensor.
    """

    tensor1 = _convert_to_numpy_fp64(tensor1)
    tensor2 = _convert_to_numpy_fp64(tensor2)

    diff = tensor1 - tensor2
    abs_diff = np.abs(diff)

    fro_norm_err = np.linalg.norm(abs_diff) / np.linalg.norm(tensor1)
    inf_norm_err = np.linalg.norm(abs_diff, ord=np.inf) / np.linalg.norm(
        tensor1, ord=np.inf
    )

    elem_wise_rel_err = np.where(tensor1 != 0, abs_diff / np.abs(tensor1), 0)

    if visualize:
        assert (
            output_folder is not None
        ), "Output folder must be specified for visualization."

        # for visualize in the one series
        _visualize_differences_one_series(
            diff=diff,
            output_folder=output_folder,
            figure_format=figure_format,
            output_prefix="diff",
        )

        _visualize_differences_one_series(
            diff=elem_wise_rel_err,
            output_folder=output_folder,
            figure_format=figure_format,
            output_prefix="rel_diff",
        )

    return {
        "abs_max_elem_diff": np.max(abs_diff),
        "abs_mean_elem_diff": np.mean(abs_diff),
        "rel_max_elem_diff": np.max(elem_wise_rel_err),
        "rel_mean_elem_diff": np.mean(elem_wise_rel_err),
        "rel_matrix_fro_norm_dist": fro_norm_err,
        "rel_matrix_inf_norm_dist": inf_norm_err,
    }


def compare_tensors_from_dicts(
    list_of_tensor_dicts: list,
    visualize: bool = False,
    figure_format: str = "png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
):
    assert len(list_of_tensor_dicts) in [
        2,
        3,
    ], "Must provide either 2 or 3 tensor dictionaries for comparison."

    nkeys = [len(t) for t in list_of_tensor_dicts]
    assert all(
        n == nkeys[0] for n in nkeys
    ), "All tensor dictionaries must have the same number of keys recording tensors."

    for key in list_of_tensor_dicts[0].keys():
        tensors = [t[key] for t in list_of_tensor_dicts]
        assert all(
            isinstance(t, (torch.Tensor, np.ndarray)) for t in tensors
        ), f"All tensors for key '{key}' must be of type torch.Tensor or np.ndarray. Found: {[type(t) for t in tensors]}"

        if len(tensors) == 3:
            result = compare_3tensors(
                tensors[0],
                tensors[1],
                tensors[2],
                visualize=visualize,
                figure_format=figure_format,
                output_folder=output_folder,
                output_prefix=output_prefix + key,
            )
        else:
            result = compare_2tensors(
                tensors[0],
                tensors[1],
                visualize=visualize,
                figure_format=figure_format,
                output_folder=output_folder,
                output_prefix=output_prefix + key,
            )

        print(f"Comparison results for {key}: {result}")


def compare_tensors_from_files(
    list_of_tensor_files: list,
    visualize: bool = False,
    figure_format: str = "png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
):
    assert len(list_of_tensor_files) in [
        2,
        3,
    ], "Must provide either 2 or 3 tensor files for comparison."

    tensor_dicts = []
    for filename in list_of_tensor_files:
        vals = torch.load(filename, map_location="cpu")
        tensor_dicts.append(vals)

    compare_tensors_from_dicts(
        list_of_tensor_dicts=tensor_dicts,
        visualize=visualize,
        figure_format=figure_format,
        output_folder=output_folder,
        output_prefix=output_prefix,
    )


def _test_api_usage():
    # Example usage
    tensor1 = torch.randn(100, 100, dtype=torch.float32)
    tensor2 = (
        tensor1 + torch.randn(100, 100, dtype=torch.float32) * 0.01
    )  # Slightly perturbed
    tensor3 = (
        tensor1 + torch.randn(100, 100, dtype=torch.float32) * 0.02
    )  # More perturbed

    result = compare_2tensors(
        tensor1,
        tensor2,
        visualize=True,
        output_folder="output",
        output_prefix="two_random_tensors",
        figure_format="png",
    )
    print(result)

    another_result = compare_3tensors(
        tensor1,
        tensor2,
        tensor3,
        visualize=True,
        output_folder="output",
        output_prefix="three_random_tensors",
        figure_format="html",
    )
    print(another_result)


if __name__ == "__main__":
    args = parse_args()

    if args.basic_self_test:
        _test_api_usage()
        exit(0)

    tensor_files = [args.tensor_file1, args.tensor_file2]
    if args.tensor_file3:
        tensor_files.append(args.tensor_file3)

    compare_tensors_from_files(
        list_of_tensor_files=tensor_files,
        visualize=args.visualize,
        figure_format=args.figure_format,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix,
    )
