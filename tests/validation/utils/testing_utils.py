import numpy as np
from tabulate import tabulate

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
