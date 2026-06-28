import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D # Required for custom legend handles

# Load the benchmark data
df = pd.read_csv('benchmark_all.csv')

# Initialize the figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Extract the independent variable X
x = df['Image Dimension N']

# Iterate through the dependent variables and plot each configuration
ax1.plot(x, df["HALE3t [euprima]"], marker='o', label="HALE3t [euprima]", markersize=2, linestyle="--")
ax1.plot(x, df["HALE3t [eulearning]"], marker='o', label="HALE3t [eulearning]", markersize=2, linestyle="--")
ax1.plot(x, df["HALE3t with contribs [euprima]"], marker='o', label="HALE3t with contribs [euprima]", markersize=2, linestyle="--")

ax1.plot(x, df["HALE3i [euprima]"], marker='o', label="HALE3i [euprima]", markersize=2, linestyle="-")
ax1.plot(x, df["HAWE3 [euprima]"], marker='o', label="HAWE3 [euprima]", markersize=2, linestyle="-")
ax1.plot(x, df["BELT3 [euprima]"], marker='o', label="BELT3 [euprima]", markersize=2, linestyle="-")

# Configure primary axis labels and title
ax1.set_xlabel('Image width N and height N')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Algorithm Performance Benchmarks')

# Impose a strict MultipleLocator to ensure major ticks occur exactly at 60-second intervals
ax1.yaxis.set_major_locator(MultipleLocator(60))
# Add minor ticks at 10-second intervals for finer granularity
ax1.yaxis.set_minor_locator(MultipleLocator(10))

# Configure gridlines to distinguish between major (minutes) and minor (10-seconds) intervals
ax1.grid(True, which='major', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', linestyle='--', alpha=0.3)

# Define the bijective mapping functions for the secondary axis
def sec_to_min(sec): 
    return sec / 60.0

def min_to_sec(minute): 
    return minute * 60.0

# Instantiate the secondary y-axis using the defined transformations
ax2 = ax1.secondary_yaxis('right', functions=(sec_to_min, min_to_sec))
ax2.set_ylabel('Time (minutes)')
ax2.yaxis.set_major_locator(MultipleLocator(1))

# --- Legend Grouping Logic ---

# 3. Retrieve the automatically generated handles (the line icons) and labels (the text)
handles, labels = ax1.get_legend_handles_labels()

# 4. Create an invisible "blank" handle to act as the visual placeholder for our headers
blank_handle = Line2D([], [], linestyle='none', marker='none')

# 5. Manually assemble new lists, inserting the blank handles and header text 
# (Indices 0,1,2 are your dashed lines; Indices 3,4,5 are your solid lines)
custom_handles = [
    blank_handle, handles[0], handles[1], handles[2],  # Group 1: Header + Dashed
    blank_handle, handles[3], handles[4], handles[5]   # Group 2: Header + Solid
]

custom_labels = [
    'Top-cell constr.', labels[0], labels[1], labels[2], 
    'Induced filtration', labels[3], labels[4], labels[5]
]

# 6. Render the legend applying the custom lists
leg = ax1.legend(custom_handles, custom_labels, loc='upper left')

# 3. Iterate through the text objects in the generated legend
for text in leg.get_texts():
    if text.get_text() in ['Top-cell constr.', 'Induced filtration']:
        # Apply italic styling
        text.set_fontstyle('italic')
        
        # Shift the text to the left by 35 points to cancel out the handle indentation
        # Note: If you change your font size later, you may need to tweak this -35 value
        text.set_position((-120, 0))

# Adjust layout padding
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig('plot_benchmark_all.png', dpi=300, bbox_inches='tight')

# Close the figure to free up memory 
plt.close(fig)
