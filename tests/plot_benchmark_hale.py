import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D # Required for custom legend handles
import matplotlib.ticker as ticker

# Load the benchmark data
df = pd.read_csv('benchmark_hale.csv')

plt.rcParams.update({
    "font.family": "sans-serif",          # Define the font family
    "font.sans-serif": ["NewComputerModernSans10"], # Specify the preferred system font
    "font.size": 14,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,                    # Embed true type fonts in the PDF (editable)
    "text.usetex": False                   # Ensure matplotlib uses its native font finder
})

# Initialize the figure and primary axis
fig, ax1 = plt.subplots(figsize=(8,5))

# Extract the independent variable X
x = df['Image Dimension N']

# Iterate through the dependent variables and plot each configuration
ax1.plot(x, df["HALE:T(2,3) [euprima]"], marker='o', label="HALE:T(2,3) [euprima]", markersize=4, linestyle="--", color="tab:blue")
ax1.plot(x, df["HALE:V(2,3) [euprima]"], marker='o', label="HALE:V(2,3) [euprima]", markersize=4, linestyle="-.", color="tab:orange")

# Configure primary axis labels and title
ax1.set_xlabel('Image width N and height N')
ax1.set_ylabel('Execution time (seconds)')
# ax1.set_title('Algorithm Performance Benchmarks')

# Impose a strict MultipleLocator to ensure major ticks occur exactly at 60-second intervals
ax1.yaxis.set_major_locator(MultipleLocator(20))
# Add minor ticks at 10-second intervals for finer granularity
ax1.yaxis.set_minor_locator(MultipleLocator(10))

# Configure gridlines to distinguish between major (minutes) and minor (10-seconds) intervals
ax1.grid(True, which='major', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', linestyle='--', alpha=0.3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2000))

# Define the bijective mapping functions for the secondary axis
def sec_to_min(sec): 
    return sec / 60.0

def min_to_sec(minute): 
    return minute * 60.0

# Instantiate the secondary y-axis using the defined transformations
ax2 = ax1.secondary_yaxis('right', functions=(sec_to_min, min_to_sec))
ax2.set_ylabel('Execution time (minutes)')
ax2.yaxis.set_major_locator(MultipleLocator(1))

# 6. Render the legend applying the custom lists
leg = ax1.legend(loc='upper left')

# Adjust layout padding
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('plot_benchmark_hale.pdf', format="pdf", bbox_inches='tight')

# Close the figure to free up memory 
plt.close(fig)
