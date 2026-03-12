import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_absolute_difference(csv_filename):
    # 1. Load the CSV file
    # We use df.iloc to select columns by index in case names are unknown
    df = pd.read_csv(csv_filename)
    
    if len(df.columns) < 2:
        print("Error: The CSV file must have at least two columns.")
        return

    # 2. Calculate the absolute difference between the first two columns
    # Formula: |Column1 - Column2|
    col1 = df.iloc[:, 0]
    col2 = df.iloc[:, 1]
    minimum = np.minimum(col1,col2)
    df['diff'] = (col1 - col2).abs()/minimum

    # 3. Generate the plot
    plt.rcParams.update({
        "pgf.texsystem": "lualatex",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([
            r"\usepackage{fontspec}",
            r"\usepackage[sfdefault]{newcomputermodern}", # Sets text to Sans
            r"\usepackage{unicode-math}",                 # Allows math font swapping
            r"\setmathfont{NewCMSansMath-Regular.otf}",          # Forces math/ticks to Sans
            r"\renewcommand{\familydefault}{\sfdefault}", # Global sans-serif override
        ])
    })  
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['diff'], marker='o', markersize=2, linestyle='-', color='tab:blue')
    
    # Adding labels and styling
    plt.title('euprima vs. euchar', fontsize=14)
    plt.xlabel('Matrix dimension (N x N)', fontsize=12)
    plt.ylabel('Performance difference (milliseconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Save and show the plot
    plt.savefig('improvement_binary_ec.pdf')
    print("Plot has been saved as 'improvement_binary_ec.pdf'")

# Replace 'your_file.csv' with the actual name of your CSV file
if __name__ == "__main__":
    # Example usage:
    plot_absolute_difference('benchmark_binary_ec.csv')
    pass
