import os
import re
from typing import List

import feedparser


def filter_papers(feed: feedparser.FeedParserDict, keywords: list) -> List[dict]:
    """
    Filters papers based on the provided keywords.

    Args:
        feed (feedparser.FeedParserDict): The list of papers fetched from the Arxiv API.
        keywords (list): The list of keywords to filter the papers.
    Returns:
        List[dict]: The list of papers that contain the keywords in their titles or abstracts.
    """
    papers = []
    for entry in feed.get("entries", []):
        title = entry.get("title", "").strip()
        # clean the title to remove unnecessary whitespace
        title = re.sub(r"\n+", " ", title)
        title = re.sub(r"\s{2,}", " ", title)
        print(title)
        # Clean the abstract to remove unnecessary blank lines
        abstract = entry.get("summary", "").strip()
        abstract = re.sub(
            r"\n+", " ", abstract
        )  # Replace multiple newlines with a single space
        abstract = re.sub(
            r"\s{2,}", " ", abstract
        )  # Replace multiple spaces with a single space

        if any(keyword.lower() in (title + abstract).lower() for keyword in keywords):
            papers.append(
                {
                    "title": title,
                    "authors": [author["name"] for author in entry["authors"]],
                    "abstract": abstract,
                    "link": entry["id"],
                    "category": entry["arxiv_primary_category"]["term"],
                }
            )
    return papers


def select_top_papers(papers: List[dict], n: int = 10) -> List[dict]:
    """
    Selects the top N papers based on their scores.

    Args:
        papers (List[dict]): The list of papers with scores.
        n (int): The number of top papers to select.
    Returns:
        List[dict]: The top N papers based on their scores.
    """
    top_papers = []
    last_score = 0

    sorted_papers = sorted(papers, key=lambda x: x["score"], reverse=True)

    for paper in sorted_papers:
        # if the score is the same as the last one, we can include it
        if len(top_papers) < n or paper["score"] == last_score:
            top_papers.append(paper)
            last_score = paper["score"]
        else:
            break

    return top_papers


def duplicate_papers(papers: List[dict], base_path: str = "summaries") -> List[dict]:
    """
    Deduplicates papers based on the link.

    Args:
        papers (List[dict]): The list of papers to deduplicate.
        base_path (str): The base path to the summary directory.
    Returns:
        List[dict]: The deduplicated list of papers.
    """
    all_papers_file = os.path.join(base_path, "all_papers.md")

    if not os.path.exists(all_papers_file):
        return papers

    # Read the existing papers from the all_papers.md file and extract the latest link
    latest_link = None
    with open(all_papers_file, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            match = re.search(r"\[.*?\]\((.*?)\)", line)
            if match:
                latest_link = match.group(1)
                break

    unique_papers = []

    for paper in papers:
        if paper["link"] > latest_link:
            unique_papers.append(paper)
        else:
            break

    return unique_papers
