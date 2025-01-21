import datetime
import time

from script.fetch_papers import fetch_papers
from script.filter_papers import filter_papers, select_top_papers, duplicate_papers
from script.summarize_papers import summarize_and_score
from script.generate_markdown import generate_markdown_for_day, update_all_papers
from script.update_readme import update_readme_with_highlight_papers
from script.git_operations import git_commit_and_push, create_pull_request


def main():
    # Step 1: Define parameters
    query = "cat:cs.*"  # Fetch computer science papers
    keywords = ["Large Language Models", "transformers", "Watermarking", "Diffusion Model", "Generative Adversarial Networks"]  # Keywords to filter papers
    max_results = 100  # Maximum number of papers to fetch
    highlight_number = 5  # Number of papers to highlight

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    print("Fetching papers from Arxiv...")
    raw_feed = fetch_papers(query, max_results)

    print("Filtering papers based on keywords...")
    filtered_papers = filter_papers(raw_feed, keywords)
    print("Number of papers fetched:", len(filtered_papers))

    print("Deduplicating papers already stored in the all_papers.md file...")
    # Deduplicate papers based on the title
    filtered_papers = duplicate_papers(filtered_papers)
    print("Number of papers after deduplication:", len(filtered_papers))

    if len(filtered_papers) == 0:
        print("No new papers found. Exiting.")
        return

    print("Summarizing and scoring papers...")

    # sort the papers by date and time, so that the latest papers are at the bottom
    filtered_papers = sorted(filtered_papers, key=lambda x: x['link'], reverse=False)

    for paper in filtered_papers:
        summary, score = summarize_and_score(paper)
        paper["summary"] = summary
        paper["score"] = score
        time.sleep(4) # Sleep for 4 seconds to avoid rate limiting, 15 requests per minute allowed by GitHub Models

    print("Selecting top papers...")
    highlight_papers = select_top_papers(filtered_papers, highlight_number)

    print("Generating daily markdown file...")
    generate_markdown_for_day(filtered_papers, date)

    print("Updating highlight papers to README.md ...")
    update_readme_with_highlight_papers(highlight_papers, filtered_papers,date)

    print("Updating all papers file...")
    update_all_papers(filtered_papers)

    print("Committing and pushing changes to Git...")
    git_commit_and_push(date)

    print("Creating pull request on GitHub...")
    create_pull_request()

    print("Workflow completed successfully.")

if __name__ == "__main__":
    main()