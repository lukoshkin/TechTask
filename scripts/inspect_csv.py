import os

import pandas as pd


def analyze_csv(file_path: str) -> bool:
    """Analyze a CSV file and return info about its structure and content."""
    print(f"\nAnalyzing file: {file_path}")

    try:
        # Try to detect if there is a header row
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        # Check if first row looks like headers (no numeric values, shorter than typical data)
        has_header = False
        first_cols = first_line.split(",")
        if first_cols and all(not col.strip().isdigit() for col in first_cols):
            has_header = True

        # Read the file with pandas for easier analysis
        df = pd.read_csv(file_path, header=0 if has_header else None)

        # Store original column names to use later
        original_column_names = df.columns.tolist()

        # Basic file stats
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        num_rows = len(df)
        num_columns = len(df.columns)

        print("\nBasic Statistics:")
        print(f"File size: {file_size:.2f} MB")
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_columns}")
        if has_header:
            print(f"Column names: {', '.join(original_column_names)}")

        # Infer column meanings based on content (first 5 rows)
        print("\nSample data (first 5 rows):")
        for i in range(min(5, len(df))):
            print(f"Row {i + 1}: {df.iloc[i].tolist()}")

        # Analyze data types and content for each column
        print("\nColumn Analysis:")
        for col_idx, col_name in enumerate(df.columns):
            non_empty_values = df[col_name].dropna().astype(str)
            value_counts = non_empty_values.value_counts()
            unique_values_count = len(value_counts)
            sample_values = (
                value_counts.index[:3].tolist()
                if unique_values_count > 0
                else []
            )

            # Try to infer if column contains IDs, language codes, categories, etc.
            column_type = "Unknown"

            # If mostly numeric, could be IDs
            if non_empty_values.str.isdigit().mean() > 0.9:
                column_type = "Numeric ID/Value"

            # Check for language codes (2-3 characters)
            if (
                non_empty_values.str.len().mean() < 4
                and unique_values_count < 100
            ):
                if all(len(val) == 2 for val in sample_values[:10]):
                    column_type = "Possible Language Code"

            # Detect text content (articles, descriptions)
            if non_empty_values.str.len().mean() > 100:
                column_type = "Long Text Content"
            elif non_empty_values.str.len().mean() > 30:
                column_type = "Text/Description"

            print(f"  Column {col_idx} ({col_name}):")
            print(f"    Values: {len(non_empty_values)} non-empty")
            print(f"    Unique values: {unique_values_count}")
            print(f"    Sample values: {sample_values[:3]}")
            print(f"    Inferred type: {column_type}")

        # Language distribution if applicable
        if "lang" in df.columns:
            lang_dist = df["lang"].value_counts()
            if (
                len(lang_dist) < 100
            ):  # If there are reasonably few distinct values
                print("\nLanguage Distribution:")
                for lang, count in lang_dist.items():
                    print(f"  {lang}: {count} ({count / num_rows * 100:.1f}%)")

        return True

    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    if os.path.isfile(file_path):
        analyze_csv(file_path)
    else:
        print(f"File not found: {file_path}")
