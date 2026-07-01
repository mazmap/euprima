import csv
import os
import matplotlib.pyplot as plt

def plot_runtime_vs_size(csv_path, output_image="runtime_plot.png"):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            print("Error: CSV file is empty.")
            return
        
        # The first 3 columns are path, width, height. The rest are algorithm names.
        alg_names = headers[3:]
        
        # Read the rows
        rows = list(reader)

    if not rows:
        print("Warning: CSV file contains headers but no data.")
        return

    # We will collect data as tuples: (image_size, [time_alg1, time_alg2, ...])
    data_points = []
    
    for row in rows:
        try:
            w = int(row[1])
            h = int(row[2])
            size = w * h
            
            # Extract execution times as floats
            times = [float(row[3 + i]) for i in range(len(alg_names))]
            data_points.append((size, times))
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping malformed row {row} - Error: {e}")
            continue

    if not data_points:
        print("Error: No valid data points extracted.")
        return

    # Sort the data points strictly by image size (x-axis)
    # This prevents the plot lines from zig-zagging back and forth if the CSV 
    # images weren't perfectly ordered by size.
    data_points.sort(key=lambda x: x[0])

    # Unzip the sorted data into x (sizes) and y (a list of times per algorithm)
    sizes = [pt[0] for pt in data_points]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for i, alg_name in enumerate(alg_names):
        # Extract the times for the current algorithm across all sorted sizes
        alg_times = [pt[1][i] for pt in data_points]
        plt.plot(sizes, alg_times, marker='o', linestyle='-', markersize=2, label=alg_name)

    # Format the plot
    plt.title('Algorithm Execution Time vs. Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Execution Time (seconds)')
    
    # Use a logarithmic scale for the x-axis if there is a massive discrepancy 
    # in image sizes, otherwise standard linear is fine. 
    # plt.xscale('log') # Uncomment if testing very small and very large images together
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Algorithms', fontsize=10)
    plt.tight_layout()

    # Save and optionally display the plot
    plt.savefig(output_image, dpi=300)
    print(f"Plot successfully generated and saved to: {output_image}")
    
    # Uncomment the next line if you want the window to pop up when the script runs
    # plt.show()

plot_runtime_vs_size("benchmark_report.csv", "benchmark_report.png")
