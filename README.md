# Daily Paper
This project automates the process of fetching, filtering, and summarizing research papers from ArXiv based on user-defined keywords. Using GitHub Actions and an external LLM for summarization and scoring, it generates daily markdown reports highlighting the most relevant papers.

# [The Latest Daily Paper](./DailyPaper.md)

### How it works

1. **Fetch**: The pipeline fetches the latest papers from ArXiv using the ArXiv API.
2. **Filter**: The pipeline filters the papers based on user-defined keywords.
3. **Summarize**: The pipeline summarizes the papers using an external LLM.
4. **Score**: The pipeline scores the papers based on the summaries.
5. **Highlight**: The pipeline highlights the most important papers based on the scores.
6. **Report**: The pipeline generates a daily markdown report of the papers.

### Acknowledgements
