import os
import re
from typing import List


def filter_papers(feed: list, keywords: list) -> List[dict]:
    """
    Filters papers based on the provided keywords.

    Args:
        feed (list): The list of papers fetched from the Arxiv API.
        keywords (list): The list of keywords to filter the papers.
    Returns:
        List[dict]: The list of papers that contain the keywords in their titles or abstracts.
    """
    papers = []
    for entry in feed:
        title = entry.get("title", "").strip()
        # clean the title to remove unnecessary whitespace
        title = re.sub(r"\n+", " ", title)
        title = re.sub(r"\s{2,}", " ", title)
        # print(title)
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


def duplicate_papers(
    papers: List[dict], base_path: str = "summaries", latest_num: int = 100
) -> List[dict]:
    """
    Deduplicates papers based on the link.
    Checks if the paper link is already present in the all_papers.md file.
        To avoid summarizing the same paper multiple times.
        Load the 100 most recent papers from all_papers.md and compare the links.

    Args:
        papers (List[dict]): The list of papers to deduplicate.
        base_path (str): The base path to the summary directory.
        latest_num (int): The number of latest papers to load from all_papers.md.
    Returns:
        List[dict]: The deduplicated list of papers.
    """
    all_papers_file = os.path.join(base_path, "all_papers.md")

    if not os.path.exists(all_papers_file):
        return papers

    # Read the existing papers from the all_papers.md file and extract the 100 most recent links
    exists_link = []  # descending order
    with open(all_papers_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(reversed(lines)):
            match = re.search(r"\[.*?\]\((.*?)\)", line)
            if match:
                latest_link = match.group(1)
                exists_link.append(latest_link)
                i = i + 1
            if i >= latest_num:
                break

    unique_papers = []

    for paper in papers:
        if paper["link"] not in exists_link:
            unique_papers.append(paper)
        else:
            print(f"- Paper already exists: {paper['title']} -")

    return unique_papers
