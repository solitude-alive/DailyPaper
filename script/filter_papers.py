import os
import re


def filter_papers(feed, keywords: list):
    """
    Filters papers based on the provided keywords.
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


def select_top_papers(papers, n=10):
    """
    Selects the top N papers based on their scores.
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


def duplicate_papers(papers, base_path="summaries"):
    """
    Deduplicates papers based on the link.
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
