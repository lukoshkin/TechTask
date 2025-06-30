#!/usr/bin/env python3

"""
Knowledge Base CSV Analysis Tool.

This script analyzes a CSV file containing knowledge base articles,
providing statistics and insights about the content.
"""

import argparse
from collections import Counter
import os
import re
from typing import Any

import pandas as pd


def load_data(file_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load the CSV file and prepare the data for analysis.

    Args:
        file_path: Path to the CSV file

    Returns
    -------
        Tuple of DataFrame and dict with basic file info
    """
    print(f"\nAnalyzing knowledge base: {file_path}")

    # Read the file with pandas - assuming first row is header
    df = pd.read_csv(file_path)

    # Basic file information
    info = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path) / (1024 * 1024),  # Size in MB
        "num_rows": len(df),
        "num_columns": len(df.columns),
    }

    # Calculate text length for later use
    df["text_length"] = df["description_text"].astype(str).apply(len)

    return df, info


def analyze_basic_stats(df: pd.DataFrame, info: dict[str, Any]) -> None:
    """
    Analyze and print basic file statistics.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information
    """
    print("\nBasic Statistics:")
    print(f"File size: {info['file_size']:.2f} MB")
    print(f"Number of rows: {info['num_rows']}")
    print(f"Number of columns: {info['num_columns']}")
    print(f"Column names: {', '.join(df.columns.tolist())}")


def analyze_columns(df: pd.DataFrame) -> None:
    """
    Analyze data types and content for each column.

    Args:
        df: DataFrame with knowledge base data
    """
    print("\nColumn Analysis:")
    for col_idx, col_name in enumerate(df.columns):
        non_empty_values = df[col_name].dropna().astype(str)
        value_counts = non_empty_values.value_counts()
        unique_values_count = len(value_counts)
        sample_values = (
            value_counts.index[:4].tolist() if unique_values_count > 0 else []
        )

        # Try to infer if column contains IDs, language codes, categories, etc.
        column_type = "Unknown"

        # If mostly numeric, could be IDs
        if non_empty_values.str.isdigit().mean() > 0.9:
            column_type = "Numeric ID/Value"

        # Check for language codes (2-3 characters)
        if non_empty_values.str.len().mean() < 4 and unique_values_count < 100:
            if all(
                len(val) == 2 for val in sample_values[:10] if not pd.isna(val)
            ):
                column_type = "Possible Language Code"

        # Detect text content (articles, descriptions)
        if non_empty_values.str.len().mean() > 100:
            column_type = "Long Text Content"
        elif non_empty_values.str.len().mean() > 30:
            column_type = "Text/Description"

        print(f"  Column {col_idx} ({col_name}):")
        print(f"    Values: {len(non_empty_values)} non-empty")
        print(f"    Unique values: {unique_values_count}")
        print(f"    Sample values: {sample_values}")
        print(f"    Inferred type: {column_type}")


def analyze_languages(
    df: pd.DataFrame, info: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze language distribution in the knowledge base.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information

    Returns
    -------
        Dict with language distribution information
    """
    lang_dist = df["lang"].value_counts()

    print("\nLanguage Distribution:")
    for lang, count in lang_dist.items():
        print(f"  {lang}: {count} ({count / info['num_rows'] * 100:.1f}%)")

    return {"lang_dist": lang_dist}


def analyze_categories(
    df: pd.DataFrame, info: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze category distribution in the knowledge base.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information

    Returns
    -------
        Dict with category distribution information
    """
    category_dist = df["categoryId"].value_counts()

    print(f"\nCategory Distribution (Top 10 of {len(category_dist)}):")
    for cat_id, count in category_dist.head(10).items():
        print(
            f"  Category {cat_id}: {count} ({count / info['num_rows'] * 100:.1f}%)"
        )

    return {"category_dist": category_dist}


def analyze_folders(df: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze folder distribution in the knowledge base.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information

    Returns
    -------
        Dict with folder distribution information
    """
    folder_dist = df["folderId"].value_counts()

    print(f"\nFolder Distribution (Top 10 of {len(folder_dist)}):")
    for folder_id, count in folder_dist.head(10).items():
        print(
            f"  Folder {folder_id}: {count} ({count / info['num_rows'] * 100:.1f}%)"
        )

    return {"folder_dist": folder_dist}


def analyze_category_folders(df: pd.DataFrame) -> None:
    """
    Analyze folder distribution by category.

    Args:
        df: DataFrame with knowledge base data
    """
    print("\nFolder Distribution by Category (Top 5 categories):")
    top_categories = df["categoryId"].value_counts().head(5).index
    for cat_id in top_categories:
        cat_df = df[df["categoryId"] == cat_id]
        print(f"\n  Category {cat_id} ({len(cat_df)} articles):")
        folder_in_cat = cat_df["folderId"].value_counts()
        for folder_id, count in folder_in_cat.head(5).items():
            print(
                f"    Folder {folder_id}: {count} ({count / len(cat_df) * 100:.1f}%)"
            )


def analyze_cross_language_categories(df: pd.DataFrame) -> None:
    """
    Analyze whether samples from different languages have the same category IDs.

    Args:
        df: DataFrame with knowledge base data
    """
    print("\nCross-Language Category Analysis:")

    # Group by language and get category sets
    languages = df["lang"].unique()
    category_by_lang = {}

    for lang in languages:
        lang_df = df[df["lang"] == lang]
        category_by_lang[lang] = set(lang_df["categoryId"])
        print(f"  {lang}: {len(category_by_lang[lang])} unique categories")

    # Find categories present in all languages (universal categories)
    if len(languages) > 1:
        universal_categories = set.intersection(*category_by_lang.values())

        print(
            f"\n  Universal categories (present in all languages): {len(universal_categories)}"
        )
        if len(universal_categories) > 0:
            print(f"  Examples: {list(universal_categories)[:5]}")

        # Analyze category overlap between languages
        print("\n  Category overlap between languages:")
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i + 1 :]:
                shared = category_by_lang[lang1].intersection(
                    category_by_lang[lang2]
                )
                total = len(
                    category_by_lang[lang1].union(category_by_lang[lang2])
                )
                overlap_pct = len(shared) / total * 100 if total > 0 else 0

                print(
                    f"    {lang1}-{lang2}: {len(shared)} shared categories "
                    f"({overlap_pct:.1f}% overlap)"
                )

        # Categories unique to specific languages
        print("\n  Categories unique to specific languages:")
        for lang in languages:
            other_langs = [l for l in languages if l != lang]
            other_lang_cats = set()
            for l in other_langs:
                other_lang_cats.update(category_by_lang[l])

            unique_cats = category_by_lang[lang] - other_lang_cats
            print(f"    {lang} - Σ(others): {len(unique_cats)} unique cat-s")
            if len(unique_cats) > 0:
                print(f"      Examples: {list(unique_cats)[:3]}")
    else:
        print(
            "  Only one language found in the data."
            " Cross-language analysis not applicable."
        )


def analyze_text_content(df: pd.DataFrame) -> None:
    """
    Analyze text content statistics.

    Args:
        df: DataFrame with knowledge base data
    """
    print("\nPlain Text Content Length Statistics:")
    print(f"  Average length: {df['text_length'].mean():.1f} characters")
    print(f"  Median length: {df['text_length'].median():.1f} characters")
    print(f"  Min length: {df['text_length'].min()} characters")
    print(f"  Max length: {df['text_length'].max()} characters")


def analyze_html_content(
    df: pd.DataFrame, info: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze HTML content in the knowledge base.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information

    Returns
    -------
        Dict with HTML analysis information
    """
    # Calculate HTML content length
    df["html_length"] = df["description"].astype(str).apply(len)

    print("\nHTML Content Length Statistics:")
    print(f"  Average length: {df['html_length'].mean():.1f} characters")
    print(f"  Median length: {df['html_length'].median():.1f} characters")
    print(f"  Min length: {df['html_length'].min()} characters")
    print(f"  Max length: {df['html_length'].max()} characters")

    # HTML structure analysis
    # Detect HTML elements used
    html_tags = []
    for desc in (
        df["description"].dropna().astype(str).sample(min(100, len(df)))
    ):
        tags = re.findall(r"<([a-zA-Z0-9]+)[^>]*>", desc)
        html_tags.extend(tags)

    # Count occurrences of each tag
    tag_counter = Counter(html_tags)
    print(
        f"\nCommon HTML Elements (from sample of {min(100, len(df))} articles):"
    )
    for tag, count in tag_counter.most_common(10):
        print(f"  <{tag}> tag: {count} occurrences")

    # Analyze links in HTML content
    has_links = df["description"].str.contains(r"<a\s+[^>]*href=[^>]*>").sum()
    print(
        f"\nArticles with links: {has_links} ({has_links / info['num_rows'] * 100:.1f}%)"
    )

    # Analyze images in HTML content - count per article
    df["image_count"] = (
        df["description"]
        .astype(str)
        .apply(lambda x: len(re.findall(r"<img\s+[^>]*>", x)))
    )
    has_images = (df["image_count"] > 0).sum()

    # Calculate average and maximum images for articles with images
    articles_with_images = df[df["image_count"] > 0]
    avg_images = 0
    max_images = 0
    if not articles_with_images.empty:
        avg_images = articles_with_images["image_count"].mean()
        max_images = articles_with_images["image_count"].max()

    print(
        f"Articles with images: {has_images} ({has_images / info['num_rows'] * 100:.1f}%)"
    )
    print(f"  Average images per article with images: {avg_images:.2f}")
    print(f"  Maximum images in a single article: {max_images}")

    # Analyze videos/embeds in HTML content - count per article
    df["video_count"] = (
        df["description"]
        .astype(str)
        .apply(lambda x: len(re.findall(r"<iframe\s+[^>]*>", x)))
    )
    has_videos = (df["video_count"] > 0).sum()

    # Calculate average and maximum videos for articles with videos
    articles_with_videos = df[df["video_count"] > 0]
    avg_videos = 0
    max_videos = 0
    if not articles_with_videos.empty:
        avg_videos = articles_with_videos["video_count"].mean()
        max_videos = articles_with_videos["video_count"].max()

    print(
        f"Articles with videos/embeds: {has_videos} ({has_videos / info['num_rows'] * 100:.1f}%)"
    )
    print(f"  Average videos per article with videos: {avg_videos:.2f}")
    print(f"  Maximum videos in a single article: {max_videos}")

    # Calculate HTML to text ratio
    valid_rows = (df["html_length"] > 0) & (df["text_length"] > 0)
    avg_ratio = 0.0
    if valid_rows.any():
        avg_ratio = (
            df.loc[valid_rows, "text_length"]
            / df.loc[valid_rows, "html_length"]
        ).mean()
        print("\nHTML to Text Ratio:")
        print(
            f"  Average text content is {avg_ratio:.2f} of HTML content size"
        )

    return {
        "has_links": has_links,
        "has_images": has_images,
        "has_videos": has_videos,
        "avg_ratio": avg_ratio,
        "avg_images_per_article": avg_images,
        "max_images": max_images,
        "avg_videos_per_article": avg_videos,
        "max_videos": max_videos,
    }


def generate_summary(
    df: pd.DataFrame, info: dict[str, Any], analyses: dict[str, Any]
) -> None:
    """
    Generate and print a summary of the knowledge base.

    Args:
        df: DataFrame with knowledge base data
        info: Dict with basic file information
        analyses: Dict with analysis results from other functions
    """
    print("\n========== SUMMARY ==========")
    print(f"Knowledge Base Size: {info['num_rows']} articles")

    # Language summary
    if "lang_dist" in analyses:
        lang_dist = analyses["lang_dist"]
        main_lang = lang_dist.idxmax()
        print(
            f"Primary Language: {main_lang} "
            f"({lang_dist[main_lang]} articles, "
            f"{lang_dist[main_lang] / info['num_rows'] * 100:.1f}%)"
        )

    # Categories and folders summary
    if "category_dist" in analyses and "folder_dist" in analyses:
        print(f"Number of Categories: {len(analyses['category_dist'])}")
        print(f"Number of Folders: {len(analyses['folder_dist'])}")

    # Content format summary
    has_html = 0
    has_text = (df["text_length"] > 100).sum()
    has_images = 0
    has_videos = 0

    if "html_length" in df.columns:
        has_html = (df["html_length"] > 100).sum()

    if "has_images" in analyses:
        has_images = analyses["has_images"]
        has_videos = analyses.get("has_videos", 0)

    print(
        f"Content Format: {has_html} articles with rich HTML content, "
        f"{has_text} with substantial plain text"
    )

    if has_images > 0 or has_videos > 0:
        print(
            f"Rich Media: {has_images} articles with images,"
            f" {has_videos} with videos/embeds"
        )


def analyze_knowledge_base(file_path: str, stats: str = "b") -> bool:
    """
    Analyze a knowledge base CSV file based on requested statistics.

    Args:
        file_path: Path to the CSV file
        stats: String containing letters for which analyses to perform:
               b - Basic statistics (default)
               c - Column analysis
               l - Language distribution
               g - Category distribution
               f - Folder distribution
               x - Cross-language category analysis
               r - Text-category correlation analysis
               t - Text content statistics
               h - HTML content analysis
               s - Summary
               a - All analyses

    Returns
    -------
        bool: True if analysis was successful, False otherwise
    """
    try:
        # If 'a' is present, perform all analyses
        if "a" in stats:
            stats = "bclgfxrthws"  # All analysis flags combined

        # Load data once for all analyses
        df, info = load_data(file_path)

        # Store results from analyses that might be needed in the summary
        analyses = {}

        # Perform requested analyses
        if "b" in stats:
            analyze_basic_stats(df, info)

        if "c" in stats:
            analyze_columns(df)

        if "l" in stats:
            analyses.update(analyze_languages(df, info))

        if "g" in stats:
            analyses.update(analyze_categories(df, info))

        if "f" in stats:
            analyses.update(analyze_folders(df, info))

        if "x" in stats:
            analyze_cross_language_categories(df)

        if "t" in stats:
            analyze_text_content(df)

        if "h" in stats:
            analyses.update(analyze_html_content(df, info))

        if "s" in stats:
            generate_summary(df, info, analyses)

        return True

    except Exception as ex:
        print(f"Error analyzing file: {str(ex)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a knowledge base CSV file and provide statistics."
    )
    parser.add_argument(
        "--stats",
        default="b",
        help="String containing flags for analyses to perform: "
        "b - Basic statistics (default), "
        "c - Column analysis, "
        "l - Language distribution, "
        "g - Category distribution, "
        "f - Folder distribution, "
        "x - Cross-language category analysis, "
        "t - Text content statistics, "
        "h - HTML content analysis, "
        "s - Summary, "
        "a - All analyses",
    )
    args = parser.parse_args()
    file_path = "data/knowledge_base.csv"
    if os.path.isfile(file_path):
        analyze_knowledge_base(file_path, args.stats)
    else:
        print(f"File not found: {file_path}")
