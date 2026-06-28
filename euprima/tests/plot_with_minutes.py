import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmark_from_csv(csv_path, output_path=None, x_values=None, T=255):
    """
    Reads benchmark timing data from a CSV and plots it with a secondary y-axis for minutes.
    
    Args:
        csv_path (str): Path to the input .csv file.
        output_path (str, optional): Path for the output image. Defaults to replacing .csv with .png.
        x_values (iterable, optional): Values for the x-axis (Image Dimensions). 
                                       If None, it defaults to the original [1000, 2000, 3000...].
        T (int): The threshold value used in the title.
    """
    # Read the data
    df = pd.read_csv(csv_path)
    
    # If no x_values are provided, reconstruct them based on the number of rows
    # Assuming START=1000, STEPS=1000 based on the original script
    if x_values is None:
        x_values = range(1000, 1000 + (len(df) * 1000), 1000)
        
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Iterate through all columns in the CSV to make the function flexible
    for column in df.columns:
        ax.plot(x_values, df[column], label=column, marker='o', markersize=2)
        
    # Primary axis formatting
    ax.set_xlabel('Image Dimension N')
    ax.set_ylabel('Avg Execution Time (s)')
    ax.set_title(f"$T_1 = T_2 = T_3 = {T}$")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Secondary y-axis for minutes
    def seconds_to_minutes(x):
        return x / 60

    def minutes_to_seconds(x):
        return x * 60

    secax = ax.secondary_yaxis('right', functions=(seconds_to_minutes, minutes_to_seconds))
    secax.set_ylabel('Avg Execution Time (min)')
    
    fig.tight_layout()
    
    # Determine output filename
    if output_path is None:
        # Replaces the .csv extension with .png
        output_path = os.path.splitext(csv_path)[0] + ".png"
        
    # Save and clean up
    fig.savefig(output_path)
    print(f"Graph successfully saved to: {output_path}")
    
    # Close the figure to free memory if calling this in a loop
    plt.close(fig)

plot_benchmark_from_csv("benchmark_report.csv")
