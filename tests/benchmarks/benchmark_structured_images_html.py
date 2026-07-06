from PIL import Image
import numpy as np
import time
import csv
import os

import euprima

def benchmark_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_c1 = img_np[:,:,0]
    img_c2 = img_np[:,:,1]
    img_c3 = img_np[:,:,2]

    # In NumPy, shape is (rows, columns, channels), mapping to (height, width)
    height, width = img_np.shape[:2]

    print(f"Benchmarking {img_path} ({width}x{height})...")

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hw_optimized(img_c1, img_c2, img_c3, 255, 255, 255)
    t_dur_1 = time.perf_counter() - t_start

    t_start = time.perf_counter()
    euler_changes = euprima.euler_changes_2d()
    euprima.ecp_2d3c_optimized(img_c1, img_c2, img_c3, euler_changes, 255, 255, 255)
    t_dur_2 = time.perf_counter() - t_start

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hl(img_c1, img_c2, img_c3, 255, 255, 255)
    t_dur_3 = time.perf_counter() - t_start

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hl_mc(img_c1, img_c2, img_c3, 255, 255, 255)
    t_dur_4 = time.perf_counter() - t_start

    return {
        "path": img_path,
        "width": width,
        "height": height,
        "HEWA3": round(t_dur_1, 4),
        "BELT3": round(t_dur_2, 4),
        "HALE3t": round(t_dur_3, 4),
        "HALE3i": round(t_dur_4, 4),
    }

def generate_csv_report(results, output_file="benchmark_report.csv"):
    if not results:
        return
        
    # Map the dictionary keys to the fieldnames
    fieldnames = ['path', 'width', 'height', "HEWA3", "BELT3", "HALE3t", "HALE3i"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write descriptive headers instead of just the dictionary keys
        writer.writerow({
            'path': 'Image Path',
            'width': 'Width (px)',
            'height': 'Height (px)',
            'HEWA3': 'HEWA3 (s)',
            'BELT3': 'BELT3 (s)',
            'HALE3t': 'HALE3t (s)',
            'HALE3i': 'HALE3i (s)'
        })
        
        # Write the actual benchmark data
        for res in results:
            writer.writerow(res)
            
    print(f"CSV successfully generated: {output_file}")

def generate_html_report(results, output_file="benchmark_report.html"):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Algorithm Benchmark Results</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 40px; background: #f9f9f9;}
        table { border-collapse: collapse; width: 100%; max-width: 1000px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 16px; text-align: center; vertical-align: middle; }
        th { background-color: #f2f2f2; font-weight: 600; color: #333; }
        .img-preview { width: auto; height: 150px; object-fit: contain; border-radius: 4px; border: 1px solid #e0e0e0; background: #fff;}
        .filename { font-size: 0.9em; font-weight: bold; color: #444; margin-top: 8px; }
        .dimensions { font-size: 0.85em; color: #666; }
    </style>
</head>
<body>
    <h2>Execution Time Benchmark</h2>
    <table>
        <thead>
            <tr>
                <th>Preview</th>
                <th>Dimensions</th>
                <th>HEWA3 (s)</th>
                <th>BELT3 (s)</th>
                <th>HALE3t (s)</th>
                <th>HALE3i (s)</th>
            </tr>
        </thead>
        <tbody>
"""
    for res in results:
        html += f"""
            <tr>
                <td>
                    <img class="img-preview" src="{res['path']}" alt="{res['path']}">
                    <div class="filename">{res['path']}</div>
                </td>
                <td>
                    <strong>{res['width']} &times; {res['height']}</strong><br>
                    <span class="dimensions">({res['width'] * res['height']} px)</span>
                </td>
                <td>{res['HEWA3']}</td>
                <td>{res['BELT3']}</td>
                <td>{res['HALE3t']}</td>
                <td>{res['HALE3i']}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport successfully generated: {output_file}")

IMG_PATHS = [
    "tigger.jpg",
    "ship.jpg",
    "bird.jpg",
    "mountains.jpg",
    "night-sky-lines.jpg",
    "minarett.jpg",
    "mountains-silhouette.jpg",
    "desert.jpg",
    "white-flower-red-background.jpg"
]

IMG_FOLDER = "test-images/"

if __name__ == "__main__":
    for img_path in IMG_PATHS:
        # Check if the file exists before attempting to open it
        if not os.path.exists(IMG_FOLDER+img_path):
            print(f"Warning: Image file '{IMG_FOLDER+img_path}' not found.")

    results = []
    for img_path in IMG_PATHS:
        results.append(benchmark_img(IMG_FOLDER+img_path))
    
    print()
    generate_csv_report(results)
    generate_html_report(results)
