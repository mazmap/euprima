import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D # Required for custom legend handles
import matplotlib.ticker as ticker

# Load the benchmark data
df = pd.read_csv('benchmark_all.csv')

plt.rcParams.update({
    "font.family": "sans-serif",          # Define the font family
    "font.sans-serif": ["NewComputerModernSans10"], # Specify the preferred system font
    "font.size": 14,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,                    # Embed true type fonts in the PDF (editable)
    "text.usetex": False,                   # Ensure matplotlib uses its native font finder
    "lines.markersize": 4
})

# Initialize the figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Extract the independent variable X
x = df['Image Dimension N']

# Iterate through the dependent variables and plot each configuration
ax1.plot(x, df["HALE:T(2,3) [euprima]"], marker='o', label="HALE:T(2,3) [euprima]", linestyle="--", color="tab:blue")
#ax1.plot(x, df["HALE:T(2,3) [eulearning]"], marker='o', label="HALE:T(2,3) [eulearning]", linestyle="--", color="tab:green")
#ax1.plot(x, df["HALE:T(2,3) like eulearning [euprima]"], marker='o', label="HALE:T(2,3) like eulearning [euprima]", linestyle="--")

ax1.plot(x, df["HALE:V(2,3) [euprima]"], marker='o', label="HALE:V(2,3) [euprima]", linestyle=":", color="tab:orange")

ax1.plot(x, df["HALE:I(2,3) [euprima]"], marker='o', label="HALE:I(2,3) [euprima]", linestyle="-", color="tab:purple")
ax1.plot(x, df["HEWA(2,3) [euprima]"], marker='o', label="HEWA(2,3) [euprima]", linestyle="-", color="tab:pink")
ax1.plot(x, df["BELT(2,3) [euprima]"], marker='o', label="BELT(2,3) [euprima]", linestyle="-", color="tab:brown")

# Configure primary axis labels and title
ax1.set_xlabel('Image width N and height N')
ax1.set_ylabel('Execution time (seconds)')
# ax1.set_title('Algorithm Performance Benchmarks')

# Impose a strict MultipleLocator to ensure major ticks occur exactly at 60-second intervals
ax1.yaxis.set_major_locator(MultipleLocator(5))
# Add minor ticks at 10-second intervals for finer granularity
ax1.yaxis.set_minor_locator(MultipleLocator(1))

# Configure gridlines to distinguish between major (minutes) and minor (10-seconds) intervals
ax1.grid(True, which='major', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', linestyle='--', alpha=0.3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1000))

# Define the bijective mapping functions for the secondary axis
def sec_to_min(sec): 
    return sec / 60.0

def min_to_sec(minute): 
    return minute * 60.0

# Instantiate the secondary y-axis using the defined transformations
#ax2 = ax1.secondary_yaxis('right', functions=(sec_to_min, min_to_sec))
#ax2.set_ylabel('Execution time (minutes)')
#ax2.yaxis.set_major_locator(MultipleLocator(1))

# --- Legend Grouping Logic ---

# 3. Retrieve the automatically generated handles (the line icons) and labels (the text)
handles, labels = ax1.get_legend_handles_labels()

# 4. Create an invisible "blank" handle to act as the visual placeholder for our headers
blank_handle = Line2D([], [], linestyle='none', marker='none')

# 5. Manually assemble new lists, inserting the blank handles and header text 
# (Indices 0,1,2 are your dashed lines; Indices 3,4,5 are your solid lines)
custom_handles = [
    blank_handle, handles[0],  # Group 1: Header + Dashed
    blank_handle, handles[1],   # Group 2: Header + Dashed-dot 
    blank_handle, handles[2], handles[3], handles[4]   # Group 2: Header + Solid
]

custom_labels = [
    "Top-cell filtration", labels[0],  # Group 1: Header + Dashed
    "Vertex filtration", labels[1],   # Group 2: Header + Dashed-dot
    "Filtration induced by top-cells", labels[2], labels[3], labels[4]   # Group 2: Header + Solid
]

# 6. Render the legend applying the custom lists
leg = ax1.legend(custom_handles, custom_labels, loc='upper left')

# 3. Iterate through the text objects in the generated legend
for text in leg.get_texts():
    if text.get_text() in ["Top-cell filtration", "Vertex filtration", "Filtration induced by top-cells"]:
        # Apply italic styling
        text.set_fontstyle('italic')
        
        # Shift the text to the left by 35 points to cancel out the handle indentation
        # Note: If you change your font size later, you may need to tweak this -35 value
        text.set_position((-29, 0))

# Adjust layout padding
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('plot_benchmark_all_without_eulearning.pdf', format="pdf", bbox_inches='tight')

# Close the figure to free up memory 
plt.close(fig)

