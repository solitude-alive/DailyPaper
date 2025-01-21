import os
from openai import OpenAI

import re


# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)


def extract_score(response_content):
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


def summarize_and_score(paper):
    """
    Summarizes the paper and assigns a score using an LLM.
    """
    prompt = (f"Please provide a detailed summary of the following paper:\n\n"
              f"Title: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
              "Additionally, conduct a **rigorous and critical evaluation** of the paper's novelty and significance within the field. "
              "Assign a score between 1 and 10, where 1 indicates a paper with negligible novelty or impact, and 10 represents an exceptional, game-changing contribution. "
              "Be **exceptionally critical** in your assessment, ensuring that your score is based on clear, well-reasoned arguments. "
              "Provide a thorough justification for the score, discussing the strengths and weaknesses of the paper, as well as its potential influence on the field. "
              "Include the score at the end in the format 'Score: X', with an emphasis on providing a **rigorous rationale** for the score assigned.")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a famous researcher, helping your students read a paper.",
            },
            {
                "role": "user",
                "content": "{}".format(prompt),
            }
        ],
        model="gpt-4o-mini",
        temperature=1,
        max_tokens=4096,
        top_p=1
    )

    summary = response.choices[0].message.content

    # Clean the summary to remove unnecessary blank lines
    summary = summary.strip()
    summary = re.sub(r'\n+', ' ', summary)  # Replace multiple newlines with a single space

    # Extract the score from the response content
    score = extract_score(summary)
    return summary, score