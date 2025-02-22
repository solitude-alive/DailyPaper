import os
import pathlib

import re
import time
from abc import ABC, abstractmethod

from google import genai

from openai import OpenAI

from script.fetch_papers import download_pdf, remove_pdf


def extract_score(response_content: str):
    """
    Extracts the score (1-10) from the response content where the score is preceded by any content
    and followed directly by a number.

    Args:
        response_content (str): The LLM response containing the summary and score.

    Returns:
        int: The extracted score, or 0 if not found.
    """
    # Regular expression to find 'Score:' preceded by any content and followed by a number between 1 and 10
    match = re.search(r"Score:.*?\s*(\b([1-9]|10)\b)", response_content)
    if match:
        return int(match.group(1))  # Extract and convert to an integer
    return 0


class Query(ABC):
    def __init__(self):
        self.client = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


# OpenAI Query
class OpenAIQuery(Query):
    def __init__(self):
        # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
        # Create your PAT token by following instructions here:
        # https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
        super().__init__()
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )

    def __call__(self, paper: dict) -> tuple[str, int]:
        prompt = (
            f"Please provide a concise summary of the following paper:\n\n"
            f"Title: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
            "Additionally, conduct a **rigorous and critical evaluation** of the paper's "
            "novelty and significance within the field. "
            "Assign a score between 1 and 10, where 1 indicates a paper with negligible novelty or impact, "
            "and 10 represents an exceptional contribution. "
            "Be **critical** in your assessment, ensuring that your score is based on clear, well-reasoned arguments. "
            "Provide a thorough justification for the score, discussing the strengths and weaknesses of the paper, "
            "as well as its potential influence on the field. "
            "Include the score at the end in the format 'Score: X', "
            "with an emphasis on providing a **rigorous rationale** for the score assigned."
        )
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a famous researcher, helping your students read a paper.",
                },
                {
                    "role": "user",
                    "content": "{}".format(prompt),
                },
            ],
            model="gpt-4o-mini",
            temperature=1,
            max_tokens=4096,
            top_p=1,
        )

        summary = response.choices[0].message.content

        # Clean the summary to remove unnecessary blank lines
        summary = summary.strip()
        summary = re.sub(
            r"\n+", " ", summary
        )  # Replace multiple newlines with a single space

        # Extract the score from the response content
        score = extract_score(summary)

        return summary, score


# Gemini Query
class GeminiQuery(Query):
    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=os.environ["GOOGLE_TOKEN"])

    def __call__(self, paper: dict):
        paper_pdf = download_pdf(paper["link"])
        prompt = (
            "Please provide a concise summary of the paper.\n"
            "Additionally, conduct a **rigorous and critical evaluation** of the paper's "
            "novelty and significance within the field. "
            "Assign a score between 1 and 10, where 1 indicates a paper with negligible novelty or impact, "
            "and 10 represents an exceptional contribution. "
            "Be **critical** in your assessment, ensuring that your score is based on clear, well-reasoned arguments. "
            "Provide a thorough justification for the score, discussing the strengths and weaknesses of the paper, "
            "as well as its potential influence on the field. "
            "Include the score at the end in the format 'Score: X', "
            "with an emphasis on providing a **rigorous rationale** for the score assigned."
        )

        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                if paper_pdf:
                    file_path = pathlib.Path("./tmp.pdf")
                    # Upload the PDF using the File API
                    sample_file = self.client.files.upload(
                        file=file_path,
                    )
                    response = self.client.models.generate_content(
                        model="gemini-1.5-flash", contents=[sample_file, prompt]
                    )
                    summary = response.text
                    score = extract_score(summary)
                    remove_pdf("./tmp.pdf")
                else:
                    print("Failed to download the PDF. Using the abstract instead.")
                    paper_abstract = paper["abstract"]
                    prompt = f"{paper_abstract}\n{prompt}"
                    response = self.client.models.generate_content(
                        model="gemini-1.5-flash", contents=prompt
                    )
                    summary = response.text
                    score = extract_score(summary)
                return summary, score
            except Exception as e:
                print(f"Error occurred while generating content: {e}")
                print("Trying again ... in 10 seconds")
                attempt += 1
                if attempt < retries:
                    time.sleep(10)

        print("Using OpenAI as a fallback")
        return OpenAIQuery()(paper)


def summarize_and_score(paper: dict, query: str = "Gemini") -> tuple[str, int]:
    """
    Summarizes the paper and assigns a score using an LLM.

    Args:
        paper (dict): The paper details containing the title and abstract.
        query (str): The query to use for summarization and scoring. ["OpenAI", "Gemini"]
    Returns:
        tuple[str, int]: The summary of the paper, and the assigned score.
    Raises:
        ValueError: If an invalid query type is provided.
    """
    print(f"Summarizing and scoring paper with {query}")
    if query == "OpenAI":
        query = OpenAIQuery()
    elif query == "Gemini":
        query = GeminiQuery()
    else:
        raise ValueError("Invalid query type. Please select 'OpenAI' or 'Gemini'.")
    summary, score = query(paper)
    return summary, score
