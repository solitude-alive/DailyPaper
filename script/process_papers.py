import asyncio
import time
from typing import Dict, List

from script.fetch_papers import download_pdf
from script.summarize_papers import summarize_and_score
from script.utils import create_directory, remove_directory


async def process_papers(papers: List[Dict], num_works: int = 5) -> None:
    """
    Process the papers by summarizing and scoring them.
    Async function to download the PDFs, and summarize the papers.

    Args:
        papers (List[Dict]): The list of papers to process.
        num_works (int): The number of works to process concurrently. Default 5
    """
    # set the semaphore to limit the number of concurrent works
    semaphore = asyncio.Semaphore(num_works)

    paper_dir = "papers"
    create_directory(paper_dir)

    async def paper_downloader(paper: Dict, save_dir: str) -> None:
        """
        Async function to download the PDF of a paper.

        Args:
            paper (Dict): The paper details.
            save_dir (str): The directory to save the PDF.
        """
        async with semaphore:
            pdf_path = await download_pdf(paper["link"], save_dir)
            paper["pdf_path"] = pdf_path

    time_current = time.time()

    # download the PDFs concurrently
    await asyncio.gather(*[paper_downloader(paper, paper_dir) for paper in papers])

    total_time = time.time() - time_current
    print(
        f"==Total download time: {int(total_time // 60)} minutes and {(total_time % 60):.2f} seconds.=="
    )

    # sort the papers by date and time, so that the latest papers are at the bottom
    papers = sorted(papers, key=lambda x: x["link"], reverse=False)

    time_current = time.time()

    # summarize and score the papers, API calls are made sequentially
    for paper in papers:
        summary, score = summarize_and_score(paper)
        paper["summary"] = summary
        paper["score"] = score

    total_time = time.time() - time_current
    print(
        f"==Total summarize time: {int(total_time // 60)} minutes and {(total_time % 60):.2f} seconds.=="
    )

    remove_directory(paper_dir)

    # sort the papers by date and time, so that the latest papers are at the bottom
    papers = sorted(papers, key=lambda x: x["link"], reverse=False)
