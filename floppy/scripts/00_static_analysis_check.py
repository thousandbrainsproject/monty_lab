import os

import pandas as pd


def main():
    path = "/Users/hlee/tbp/monty_lab/floppy/results/static_analysis/flop_analysis_20250218_114725.csv"

    df = pd.read_csv(path)

    unique_functions_and_counts = df["operation_type"].value_counts()
    print(unique_functions_and_counts.head(20))


if __name__ == "__main__":
    main()
