import numpy as np
import pandas as pd
from tabulate import tabulate
import time
from tqdm import tqdm

class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def check_pairwise_equality(arrays, labels=None):
    """
    Compares n numpy arrays for exact equality and prints a summary table.
    """
    n = len(arrays)
    if labels is None:
        labels =[f"Array {i}" for i in range(n)]
    
    headers = [""] + labels
    table_data = []

    for i in range(n):
        row = [labels[i]]
        for j in range(i+1):
            # Check for exact equality (shape and elements)
            is_equal = np.array_equal(arrays[i], arrays[j])
            
            # Format output for a quick visual scan
            result = f"{Color.GREEN}MATCH{Color.RESET}" if is_equal else f"{Color.RED}{Color.BOLD}DIFF{Color.RESET}"
            row.append(result)
            
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

def benchmark(function, input_generator, ig_args, display_progress_bar=True):
    summary = []
    if display_progress_bar:
        print("Start benchmark for", function.__name__)

    for arg in tqdm(ig_args):
        inputs = input_generator(arg)

        t_start = time.perf_counter()
        for input in inputs:
            function(*input)
        t_end = time.perf_counter()
        t_avg = (t_end - t_start) / len(inputs)
        summary.append(t_avg)

    return summary

def visualize_performance_improvement(csv_filename):
    df = pd.read_csv(csv_filename)
    # TODO

def visualize_benchmark(csv_filename):
    df = pd.read_csv(csv_filename)
    #TODO
