import requests
import time
import feedparser


def fetch_papers(query, max_results=50, retries=3, wait_time=30):
    """
    Fetches papers from Arxiv and retries on errors.

    Args:
        query (str): The query string for searching papers on Arxiv.
        max_results (int): The maximum number of papers to fetch.
        retries (int): Number of times to retry the request on failure.
        wait_time (int): Wait time (in seconds) between retries.

    Returns:
        List[dict]: A list of dictionaries containing paper details.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    attempt = 0
    while attempt < retries:
        try:
            # Send request to Arxiv API
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()  # Raise HTTP errors if they occur

            # Parse the Atom feed response
            feed = feedparser.parse(response.text)

            return feed

        except (requests.exceptions.RequestException, feedparser.NonXMLContentType) as e:
            # Handle HTTP or parsing errors
            print(f"Error occurred: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in {wait_time} seconds... (Attempt {attempt}/{retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Exiting.")
                raise

