"""
Script to analyze grid search results.
"""

import os
import argparse
import pandas as pd
import options
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

opts = options.Options()
os.makedirs(opts.output_dir())
variable_cols = opts._variety.keys()


def split_and_titlecase(text):
    return " ".join(text.split("_")).title()


def find_optimal_configuration(data):
    max_acc_row = data.loc[data["final_test_acc"].idxmax(), :]
    print("=" * 40)
    print("Optimal configuration")
    print("-" * 40)
    print(max_acc_row)
    print("=" * 40)
    return max_acc_row


def plot_column_correlations(data):
    correlations = {}
    for col in [col for col in data.columns if col != "final_test_acc"]:
        corr = data["final_test_acc"].corr(data[col])
        correlations[col] = corr
    plt.bar(range(len(correlations)), list(correlations.values()))
    plt.title("Correlation with Final Test Accuracy")
    plt.xticks(range(len(correlations)), [split_and_titlecase(label) for label in correlations.keys()])
    plt.setp(plt.xticks()[1], rotation=30, horizontalalignment='right')
    plt.ylabel("Correlation")
    plt.savefig(opts.output_path("correlations.png"), bbox_inches="tight")


def scatter_plot(data, x, y):
    data.plot.scatter(x=x, y=y)
    x_title = split_and_titlecase(x)
    y_title = split_and_titlecase(y)
    plt.title("{} vs. {}".format(x_title, y_title))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(opts.output_path("{}_vs_{}.png".format(x, y)), bbox_inches="tight")


def main(args):
    df = pd.read_csv(args.input_csv)
    df = df.drop([col for col in df.columns if not (col in variable_cols or col == "final_test_acc")], "columns")
    best_configuration = find_optimal_configuration(df)
    plot_column_correlations(df)
    for col in tqdm([col for col in df.columns if col not in ["final_test_acc", "use_batchnorm"]]):
        scatter_plot(df, col, "final_test_acc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    main(parser.parse_args())
