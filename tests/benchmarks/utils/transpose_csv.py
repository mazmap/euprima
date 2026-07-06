import csv
import sys


def transpose_csv(input_file: str, output_file: str) -> None:
    """Reads a CSV file, transposes its rows and columns, and writes to a new file."""
    try:
        # Read the original CSV data
        with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            print(f"Warning: '{input_file}' is empty.")
            return

        # Transpose the data using zip
        # zip(*rows) pairs the first elements together, second elements together, etc.
        transposed_rows = list(zip(*rows))

        # Write the transposed data to the output CSV file
        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerows(transposed_rows)

        print(
            f"Success: Transposed '{input_file}' and saved to '{output_file}'."
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Example usage:
    # Replace 'input.csv' and 'output.csv' with your actual file paths
    input_filename = "benchmark_all.csv"
    output_filename = "benchmark_all_transposed.csv"

    transpose_csv(input_filename, output_filename)
