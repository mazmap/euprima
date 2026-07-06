import csv
import sys


def format_value(val: str) -> str:
    """Helper function to format floats to 2 decimal places.

    Leaves text headers and integers unchanged.
    """
    # Strip whitespace to avoid parsing issues
    val_stripped = val.strip()

    # If it's an integer, preserve it as-is
    if val_stripped.isdigit() or (
        val_stripped.startswith("-") and val_stripped[1:].isdigit()
    ):
        return val_stripped

    # Try to convert to float and round to 2 decimal places
    try:
        float_val = float(val_stripped)
        return f"{float_val:.2f}"
    except ValueError:
        # If it's a header or string, return the original value
        return val


def transpose_and_round_csv(input_file: str, output_file: str) -> None:
    """Reads a CSV file, rounds floats to 2 decimal places, transposes

    rows and columns, and writes the result to a new file.
    """
    try:
        # Read the original CSV data
        with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            print(f"Warning: '{input_file}' is empty.")
            return

        # Format each cell: apply rounding to floats, preserve headers/integers
        formatted_rows = [
            [format_value(cell) for cell in row] for row in rows
        ]

        # Transpose the formatted data matrix
        transposed_rows = list(zip(*formatted_rows))

        # Write the transposed data to the output CSV file
        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerows(transposed_rows)

        print(
            f"Success: Transposed '{input_file}' with 2-decimal rounding and saved to '{output_file}'."
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Example usage:
    # Replace 'input.csv' and 'output.csv' with your actual file paths
    input_filename = "benchmark_all_2.csv"
    output_filename = "benchmark_all_2_transposed_rounded.csv"

    transpose_and_round_csv(input_filename, output_filename)
