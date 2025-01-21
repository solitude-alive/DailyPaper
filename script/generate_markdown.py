import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_markdown_for_day(papers, date, base_path="summaries"):
    """
    Generates the daily summary Markdown file.
    """
    daily_path = os.path.join(base_path, date[:7])
    create_directory(daily_path)

    # If no papers are found, return
    if len(papers) == 0:
        return

    daily_file = os.path.join(daily_path, f"{date}.md")
    with open(daily_file, "w") as f:
        f.write(f"# Daily Summary: {date}\n\n")
        for paper in papers:
            f.write(f"## {paper['title']}\n")
            f.write(f"- **Link**: [Link to Paper]({paper['link']})\n")
            f.write(f"- **Authors**: {', '.join(paper['authors'])}\n")
            f.write(f"- **Abstract**: {paper['abstract']}\n")
            f.write(f"- **Summary**: {paper['summary']}\n")
            f.write(f"- **Classification**: {paper['category']}\n")
            f.write(f"- **Score**: {paper['score']}/10\n\n")


def update_all_papers(papers, base_path="summaries"):
    """
    Appends new papers to the all_papers.md file.
    """
    all_papers_file = os.path.join(base_path, "all_papers.md")
    with open(all_papers_file, "a") as f:
        for paper in papers:
            f.write(f"- **[Title: {paper['title']}]({paper['link']})**\n")
            f.write(f"  - **Classification**: {paper['category']}\n")
            f.write(f"  - **Summary**: {paper['summary']}\n")
            f.write(f"  - **Score**: {paper['score']}/10\n\n")
