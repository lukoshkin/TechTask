#!/usr/bin/env python3

"""
Knowledge Base Sample Viewer.

This script generates samples from a knowledge base CSV file,
showing a specified number of samples for each language or a specific language.
With the --html flag, it can show HTML content instead of plain text.
"""

import argparse
import os

import pandas as pd


def show_available_categories(file_path: str) -> bool:
    """
    Show all available category IDs in the knowledge base.

    Args:
        file_path: Path to the CSV file

    Returns
    -------
        bool: True if categories were displayed successfully, False otherwise
    """
    print(f"\nAvailable category IDs in: {file_path}")

    try:
        # Read the file with pandas
        df = pd.read_csv(file_path)

        # Get unique category IDs and sort them
        categories = sorted(df["categoryId"].unique().tolist())

        # Display the categories
        for category_id in categories:
            # Count samples in this category
            count = len(df[df["categoryId"] == category_id])
            print(f"Category ID: {category_id} ({count} samples)")

        return True

    except Exception as ex:
        print(f"Error processing file: {str(ex)}")
        return False


def show_kb_samples(
    file_path: str,
    num_samples: int = 3,
    lang: str | None = None,
    category_id: int | None = None,
    show_html: bool = False,
    seed: int | None = None,
    min_text_length: int | None = None,
) -> bool:
    """
    Show samples from a knowledge base CSV file.

    Args:
        file_path: Path to the CSV file
        num_samples: Number of samples to show per language
        lang: Optional language code to filter by
        category_id: Optional category ID to filter by
        show_html: Whether to show HTML content instead of plain text
        seed: Random seed for consistent sample selection
        min_text_length: Optional minimum length of plain text to filter by

    Returns
    -------
        bool: True if samples were displayed successfully, False otherwise
    """
    print(f"\nShowing samples from knowledge base: {file_path}")

    try:
        # Read the file with pandas
        df = pd.read_csv(file_path)

        # Apply category filter if specified
        if category_id is not None:
            if category_id not in df["categoryId"].unique():
                print(f"Error: Category '{category_id}' not found in the KB.")
                return False
            df = df[df["categoryId"] == category_id]
            if df.empty:
                print(f"No samples found for category ID: {category_id}")
                return False

        # Apply minimum text length filter if specified
        if min_text_length is not None:
            # Calculate text length for each row
            df["text_length"] = df["description_text"].apply(
                lambda x: len(str(x))
            )
            # Filter by minimum length
            original_count = len(df)
            df = df[df["text_length"] >= min_text_length]
            filtered_count = len(df)

            if df.empty:
                print(
                    "No samples found with text length >= "
                    f"{min_text_length} characters."
                )
                return False
            elif filtered_count < original_count:
                print(
                    f"Filtered {original_count - filtered_count} samples "
                    f"with text length < {min_text_length} characters."
                )

        # Get list of languages to process
        if lang:
            if lang not in df["lang"].unique():
                print(f"Error: Language '{lang}' not found in the KB.")
                return False
            languages = [lang]
        else:
            languages = sorted(df["lang"].unique().tolist())

        # Show samples for each language
        for language in languages:
            print(f"\n=== Samples for language: {language} ===")

            # Filter dataframe for the current language
            lang_df = df[df["lang"] == language]

            # Limit to the specified number of samples with fixed seed if provided
            sample_size = min(num_samples, len(lang_df))
            samples = lang_df.sample(sample_size, random_state=seed)

            # Display each sample
            for i, (_, row) in enumerate(samples.iterrows()):
                print(f"\nSample {i + 1}:")
                print(f"  Title: {row['title']}")
                print(f"  Category ID: {row['categoryId']}")
                print(f"  Folder ID: {row['folderId']}")

                if show_html:
                    html_content = str(row.get("description", "")).strip()
                    print(f"  HTML Content: {html_content}")
                else:
                    text_preview = str(row.get("description_text", "")).strip()
                    print(f"  Text Preview: {text_preview}")

                print(
                    f"  HTML Length: {len(str(row.get('description', '')))} characters"
                )
                print(
                    f"  Text Length: {len(str(row.get('description_text', '')))} characters"
                )

        return True

    except Exception as ex:
        print(f"Error processing file: {str(ex)}")
        return False


def main() -> None:
    """Parse command line arguments and run the sample viewer."""
    parser = argparse.ArgumentParser(
        description="Display samples from a knowledge base CSV file."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to show per language (default: 3)",
    )
    parser.add_argument(
        "-l", "--lang", type=str, help="Filter by specific language code"
    )
    parser.add_argument(
        "-c", "--categoryId", type=int, help="Filter by specific category ID"
    )
    parser.add_argument(
        "--show-categories",
        action="store_true",
        help="Show available category IDs and exit",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Display HTML content instead of plain text",
    )
    parser.add_argument(
        "-s",
        "--seed",
        nargs="?",
        const=0,  # Default to 0 if flag is used without value
        type=int,
        help="Set random seed for consistent sample selection (default: 0 if flag provided without value)",
    )
    parser.add_argument(
        "--tl",
        type=int,
        help="Filter samples by minimum text length (in characters)",
    )
    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        default="data/knowledge_base.csv",
        help="Path to the knowledge base CSV file (default: data/knowledge_base.csv)",
    )
    args = parser.parse_args()

    # Check if file exists
    if not os.path.isfile(args.file_path):
        print(f"File not found: {args.file_path}")
        return

    # Show categories if requested and exit
    if args.show_categories:
        show_available_categories(args.file_path)
        return

    # Run the sample viewer
    show_kb_samples(
        file_path=args.file_path,
        num_samples=args.num_samples,
        lang=args.lang,
        category_id=args.categoryId,
        show_html=args.html,
        seed=args.seed,
        min_text_length=args.tl,
    )


if __name__ == "__main__":
    main()
