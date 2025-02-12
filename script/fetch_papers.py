import os
import time

import feedparser
import fitz
import requests


def fetch_papers(
    query: str, max_results: int = 50, retries: int = 5, wait_time: int = 30
) -> list:
    """
    Fetches papers from Arxiv and retries on errors, ensuring up to max_results are retrieved.

    Args:
        query (str): The query string for searching papers on Arxiv.
        max_results (int): The maximum number of papers to fetch.
        retries (int): Number of times to retry the request on failure.
        wait_time (int): Wait time (in seconds) between retries.

    Returns:
        list: A list of paper entries.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the request.
        feedparser.NonXMLContentType: If the response content is not XML.
    """
    base_url = "http://export.arxiv.org/api/query"
    all_entries = []
    start = 0

    while len(all_entries) < max_results:
        attempt = 0

        while attempt < retries:
            try:
                # Define request parameters, https://info.arxiv.org/help/api/user-manual.html
                # (max_results) is limited to 30000 in slices of at most 2000 at a time
                params = {
                    "search_query": query,
                    "start": start,
                    "max_results": min(
                        max_results - len(all_entries), 2000
                    ),  # Fetch in batches
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }

                # Send request to Arxiv API
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()  # Raise HTTP errors if they occur

                # Parse the Atom feed response
                feed = feedparser.parse(response.text)

                if feed.bozo:
                    raise feed.bozo_exception

                # Add new entries to the all_entries list
                entries = feed.entries
                if not entries:
                    attempt += 1
                    if attempt < retries:
                        print(
                            f"Already fetched {len(all_entries)} papers. No new papers found."
                        )
                        print(
                            f"Retrying in {wait_time} seconds... (Attempt {attempt}/{retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        print("Max retries reached.")
                        if len(all_entries) == 0:
                            raise Exception(
                                "Failed to fetch papers after multiple retries."
                            )
                        else:
                            print("Returning the fetched papers.")
                            return all_entries
                else:
                    all_entries.extend(entries)
                    start += len(entries)
                    break  # Successfully fetched, exit retry loop

            except (
                requests.exceptions.RequestException,
                feedparser.NonXMLContentType,
            ) as e:
                # Handle HTTP or parsing errors
                print(f"Error occurred: {e}")
                attempt += 1
                if attempt < retries:
                    print(
                        f"Retrying in {wait_time} seconds... (Attempt {attempt}/{retries})"
                    )
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Exiting.")
                    raise Exception(
                        "Failed to fetch papers after multiple retries."
                    ) from e

    return all_entries[:max_results]  # Return exactly max_results


def fetch_pdf(url: str, max_retries: int = 3, wait_time: int = 15) -> str:
    """
    Fetch and extract text from a PDF file hosted on arXiv.

    Args:
        url (str): The arXiv URL (e.g., "https://arxiv.org/abs/2401.12345").
        max_retries (int): Number of retry attempts in case of failure. Default is 3.
        wait_time (int): Initial wait time (in seconds) before retrying. Default is 15s.

    Returns:
        str: Extracted text from the PDF, or an error message if extraction fails.
    """
    # Extract the arXiv ID from the URL
    try:
        arxiv_id = url.strip().split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except IndexError:
        return "Invalid arXiv URL format."

    save_path = "./tmp.pdf"

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Downloading {pdf_url} ...")
            response = requests.get(
                pdf_url, timeout=15
            )  # Set a timeout to prevent hanging

            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"PDF downloaded successfully: {save_path}")
                break  # Exit loop on success
            else:
                print(
                    f"Error: HTTP {response.status_code} - Retrying in {wait_time} seconds..."
                )
        except requests.RequestException as e:
            print(f"Request failed: {e} - Retrying in {wait_time} seconds...")

        time.sleep(wait_time)  # Wait before retrying
    else:
        return "Failed to download PDF after multiple attempts."

    # Extract text from the PDF
    try:
        doc = fitz.open(save_path)
        text = "\n".join([page.get_text("text") for page in doc])

        # Cleanup: remove the temporary PDF file
        os.remove(save_path)

        return text if text.strip() else "PDF extracted but contains no readable text."
    except Exception as e:
        return f"PDF extraction failed: {e}"
