import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for T in [50,100,150,175,200,225,255]: 
    csv_filename = f"benchmark_ecp_2d3c_T{T}";

    df = pd.read_csv(csv_filename+".csv")

    summary1 = df.iloc[:,0]
    summary2 = df.iloc[:,1]

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, [t*1000 for t in summary1], label='ecp_2d3c [euprima]', marker='o', markersize=2)
    plt.plot(df.index, [t*1000 for t in summary2], label='ecp_2d3c_hw_optimized2 [euprima]', marker='o', markersize=2)

    plt.xlabel('Matrix Dimension (N x N)')
    plt.ylabel('Avg Execution Time (ms)')
    plt.title(f"$T_1 = T_2 = T_3 = {T}$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"benchmark_ecp_2d3c_T{T}"+".png")
