import csv
import sys


def format_value(val: str) -> str:
    """Helper function to format floats to 2 decimal places.

    Leaves text headers and integers unchanged.
    """
    val_stripped = val.strip()

    # If it's an integer, preserve it as-is (e.g., 1000, 2000)
    if val_stripped.isdigit() or (
        val_stripped.startswith("-") and val_stripped[1:].isdigit()
    ):
        return val_stripped

    # Try to convert to float and round to 2 decimal places
    try:
        float_val = float(val_stripped)
        return f"{float_val:.2f}"
    except ValueError:
        # If it's a header text, return the original value
        return val


def round_csv_values(input_file: str, output_file: str) -> None:
    """Reads a CSV file, rounds floats to 2 decimal places, and saves the

    result in the same layout.
    """
    try:
        # Read the original CSV data
        with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            print(f"Warning: '{input_file}' is empty.")
            return

        # Format each cell without transposing the overall structure
        formatted_rows = [
            [format_value(cell) for cell in row] for row in rows
        ]

        # Write the formatted data straight to the output CSV file
        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerows(formatted_rows)

        print(
            f"Success: Rounded floats in '{input_file}' and saved to '{output_file}'."
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Example usage:
    # Replace 'input.csv' and 'output.csv' with your actual file paths
    input_filename = "benchmark_hale.csv"
    output_filename = "benchmark_hale_rounded.csv"

    round_csv_values(input_filename, output_filename)
