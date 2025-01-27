import time

import feedparser
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
