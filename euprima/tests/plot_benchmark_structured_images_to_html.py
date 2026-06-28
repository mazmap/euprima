import csv
import os

def generate_html_from_csv(csv_path, output_file="transposed_benchmark_report.html"):
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
            
        rows = list(reader)

    if not rows:
        print("Warning: CSV file contains headers but no data.")
        return

    # Extract algorithm names dynamically (everything after path, width, height)
    algorithm_names = headers[3:]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transposed Algorithm Benchmark</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 40px; background: #f9f9f9; }
        .table-container { overflow-x: auto; max-width: 100%; background: white; }
        table { border-collapse: collapse; width: max-content; min-width: 100%; }
        th, td { border: 1px solid #ddd; padding: 16px; text-align: center; vertical-align: middle; }
        th { background-color: #f2f2f2; font-weight: 600; color: #333; text-align: right; position: sticky; left: 0; z-index: 1;}
        /* Add a right border to the sticky header to separate it from scrolling content */
        th::after { content: ''; position: absolute; top: 0; right: 0; bottom: 0; border-right: 2px solid #ccc; }
        .img-preview { width: 150px; height: auto; object-fit: contain; background: #fff; }
        .filename { font-size: 0.9em; font-weight: bold; color: #444; margin: 8px auto 0; max-width: 150px; word-wrap: break-word; text-align: center; }
        .dimensions { font-size: 0.85em; color: #666; }
    </style>
</head>
<body>
    <h2>Structured Images Benchmark</h2>
    <div class="table-container">
        <table>
            <tbody>
"""

    # --- Row 1: Preview Images ---
    html += "                <tr>\n"
    html += "                    <th>Preview</th>\n"
    for row in rows:
        path = row[0]
        html += f"""                    <td>
                        <img class="img-preview" src="{path}" alt="{path}">
                        <div class="filename">{path}</div>
                    </td>\n"""
    html += "                </tr>\n"

    # --- Row 2: Dimensions ---
    html += "                <tr>\n"
    html += "                    <th>Dimensions</th>\n"
    for row in rows:
        w, h = row[1], row[2]
        # Calculate pixel count safely
        try:
            pixels = int(w) * int(h)
        except ValueError:
            pixels = 0
            
        html += f"""                    <td>
                        <strong>{w} &times; {h}</strong><br>
                        <span class="dimensions">({pixels} px)</span>
                    </td>\n"""
    html += "                </tr>\n"

    # --- Rows 3+: Algorithm Execution Times ---
    for i, alg_name in enumerate(algorithm_names):
        html += "                <tr>\n"
        html += f"                    <th>{alg_name}</th>\n"
        for row in rows:
            # The algorithm times start at index 3
            # Use a safe fallback if the row is somehow missing columns
            time_val = row[3 + i] if len(row) > 3 + i else "N/A"
            html += f"                    <td>{int(round(float(time_val),0))}</td>\n"
        html += "                </tr>\n"

    html += """            </tbody>
        </table>
    </div>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Transposed HTML report successfully generated: {output_file}")

generate_html_from_csv("benchmark_report.csv")
