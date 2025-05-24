#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

Shape = tuple[int, int, int]


class Report:
    runs: list[float]

    def __init__(self, flops: Optional[float] = None):
        if flops is not None:
            self.runs = [flops]
        else:
            self.runs = []

    def __str__(self):
        return str(self.runs)

    def append(self, flops: float):
        self.runs.append(flops)

    def extend(self, runs: list[float]):
        self.runs.extend(runs)

    def get_mean(self) -> float:
        return np.mean(self.runs).item()

    def get_std(self) -> float:
        return np.std(self.runs).item()

    def get_cv(self) -> float:
        return self.get_std() / self.get_mean()


def _parse_results(filename: Path) -> dict[str, dict[Shape, Report]]:
    def is_aggregate(name: str) -> bool:
        return name.split("/")[-1].find("_") != -1

    with open(filename, 'r') as f:
        data = json.load(f)
    results: dict[str, dict[Shape, Report]] = defaultdict(lambda: defaultdict(lambda: Report()))
    for result in data["benchmarks"]:
        name = result["name"]
        if is_aggregate(name):
            continue
        stuff = name.split("/")
        lib = stuff[0]
        shape = (int(stuff[1:][0]), int(stuff[1:][1]), int(stuff[1:][2]))
        results[lib][shape].append(result["FLOPS"])
    return results


def get_results(filenames: list[Path]) -> dict[str, dict[Shape, Report]]:
    merged_results: dict[str, dict[Shape, Report]] = defaultdict(
        lambda: defaultdict(lambda: Report()))
    for filename in filenames:
        results = _parse_results(filename)
        for lib, result in results.items():
            for shape, report in result.items():
                merged_results[lib][shape].extend(report.runs)
    return merged_results


def plot_results(
    results: dict[str, dict[Shape, Report]],
    peak: Optional[int] = None
) -> Figure:
    bar_height: float = 1.00
    num_bars: dict[Shape, int] = defaultdict(lambda: 0)
    for lib, result in results.items():
        for shape in result.keys():
            num_bars[shape] += 1
    bar_counts: NDArray = np.array(list(num_bars.values()), dtype=np.float64)
    shape_ys: NDArray = np.zeros(len(bar_counts))
    for i in range(1, len(bar_counts)):
        shape_ys[i] = shape_ys[i - 1] + bar_counts[i - 1] * bar_height + bar_height

    fig, ax = plt.subplots(figsize=(5.5, len(bar_counts)))
    for idx, lib in enumerate(results.keys()):
        ys: list[float] = []
        widths: list[float] = []
        for shape_idx, shape in enumerate(num_bars.keys()):
            if shape in results[lib]:
                ys.append(shape_ys[shape_idx] + idx * bar_height)
                widths.append(results[lib][shape].get_mean() / 1e9)
        if not ys:
            continue
        bars = ax.barh(
            y=ys,
            width=widths,
            height=bar_height,
            label=lib
        )
        ax.bar_label(bars, padding=3, fmt="{:.2f}")
        if lib == "Matmul":
            for bar in bars:
                bar.set_hatch("//")
                bar.set_hatch_linewidth(0.25)

    labels: list[str] = list(map(
        lambda shape: f"${r"\times".join(str(x) for x in shape)}$",
        num_bars.keys()))
    if peak is not None:
        ax.set_xlim(0, peak)
    ax.set_yticks(shape_ys + (bar_counts - 1) * bar_height / 2, labels)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper center', fancybox=True, ncol=3)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak", type=int, metavar="FLOPS",
                        help="theoretical peak GFLOPS")
    parser.add_argument("--results", type=Path, nargs='+', metavar="FILE...",
                        help="results files to process")
    parser.add_argument("--output", type=Path, default="figures/plot.svg",
                        metavar="PATH", help="path to output")
    return parser.parse_args()


def main():
    args = parse_args()
    results = get_results(args.results)
    fig = plot_results(results, args.peak)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
