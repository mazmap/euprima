import pandas as pd
import matplotlib.pyplot as plt
import math

df1 = pd.read_csv("benchmark_eulearning.csv")
df2 = pd.read_csv("../benchmark_hl_top_cell.csv")
df2 = df2[:len(df1)]

# ecp_2d3c_hl
summary1 = df1.iloc[:,0]
# eulearning
summary2 = df1.iloc[:,1]
# ecp_2d3c_hl_contributions
summary3 = df1.iloc[:,2]
# ecp_2d3c_hw
summary4 = df2.iloc[:,1]

plt.figure(figsize=(10, 6))

xaxis = [i for i in range(500,5000+1,500)]

plt.plot(xaxis, [t for t in summary1], label='ecp_2d3c_hl [euprima]', marker='o', markersize=2)
plt.plot(xaxis, [t for t in summary2], label='compute_contributions [euprima] + compute_euler_profile [eulearning]', marker='o', markersize=2)
plt.plot(xaxis, [t for t in summary3], label='compute_contributions [euprima] + ecp_2d3c_hl_contributions [euprima]', marker='o', markersize=2)
plt.plot(xaxis, [t for t in summary4], label='ecp_2d3c_hw [euprima]', marker='o', markersize=2)

plt.xlabel('Image Dimension $N$ (N x N)')
plt.ylabel('Execution Time (in s)')
plt.title(f"$T_1 = T_2 = T_3 = 255$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("whole-eulearning-comparison.png")
