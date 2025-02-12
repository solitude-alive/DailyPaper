import datetime
import time

from script.fetch_papers import fetch_papers
from script.filter_papers import duplicate_papers, filter_papers, select_top_papers
from script.generate_markdown import generate_markdown_for_day, update_all_papers
from script.git_operations import create_pull_request, git_commit_and_push
from script.summarize_papers import summarize_and_score
from script.update_daily import update_daily_papers


def main():
    # Step 1: Define parameters
    query = "cat:cs.*"  # Fetch computer science papers
    keywords = [
        "Large Language Models",
        "transformers",
        "Watermarking",
        "Diffusion Model",
        "Generative Adversarial Networks",
        "Chain-of-Thought",
        "Image Generation",
    ]  # Keywords to filter papers
    max_results = 500  # Maximum number of papers to fetch
    highlight_number = 5  # Number of papers to highlight

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    print("Fetching papers from Arxiv...")
    raw_feed = fetch_papers(query, max_results)
    print("Number of papers fetched:", len(raw_feed))

    print("Filtering papers based on keywords...")
    filtered_papers = filter_papers(raw_feed, keywords)
    print("Number of related papers fetched:", len(filtered_papers))

    # Deduplicate papers based on the title, the latest paper are at the top
    print("Deduplicating papers already stored in the all_papers.md file...")
    filtered_papers = duplicate_papers(filtered_papers)
    print("Number of papers after deduplication:", len(filtered_papers))

    # sort the papers by date and time, so that the latest papers are at the bottom
    filtered_papers = sorted(filtered_papers, key=lambda x: x["link"], reverse=False)

    if len(filtered_papers) == 0:
        print("No new papers found. Exiting.")
        return
    elif len(filtered_papers) > 150:
        print("Too many papers found. Only use the latest 150 papers.")
        filtered_papers = filtered_papers[:150]

    print("Summarizing and scoring papers...")

    for paper in filtered_papers:
        summary, score = summarize_and_score(paper)
        paper["summary"] = summary
        paper["score"] = score
        time.sleep(4)  # Sleep for 4 seconds to avoid rate limiting,
        # 15 requests per minute allowed by GitHub Models, and Gemini API
        # 1M tokens per minute allowed by Gemini API

    print("Selecting top papers...")
    highlight_papers = select_top_papers(filtered_papers, highlight_number)

    print("Generating daily markdown file...")
    generate_markdown_for_day(filtered_papers, date)

    print("Updating highlight papers to DailyPaper.md ...")
    update_daily_papers(highlight_papers, filtered_papers, date)

    print("Updating all papers file...")
    update_all_papers(filtered_papers, date)

    print("Committing and pushing changes to Git...")
    git_commit_and_push(date)

    print("Creating pull request on GitHub...")
    create_pull_request(date)

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
