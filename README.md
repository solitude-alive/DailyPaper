# Daily Paper
This project automates the process of fetching, filtering, and summarizing research papers from ArXiv based on user-defined keywords. Using GitHub Actions and an external LLM for summarization and scoring, it generates daily markdown reports highlighting the most relevant papers.

# Paper

- [Daily Papers](./DailyPaper.md)
- [All Papers](./summaries/all_papers.md)

### How it works

1. **Fetch**: The pipeline fetches the latest papers from ArXiv using the ArXiv API.
2. **Filter**: The pipeline filters the papers based on user-defined keywords.
3. **Summarize**: The pipeline summarizes the papers using an external LLM.
4. **Score**: The pipeline scores the papers based on the summaries.
5. **Highlight**: The pipeline highlights the most important papers based on the scores.
6. **Report**: The pipeline generates a daily markdown report of the papers.
7. **Commit, Push, and Pull Request**: The pipeline automatically commits the report to the repository, pushes the changes, and creates a pull request.

### Acknowledgements

This project draws inspiration from the following open-source projects:

- [DailyArxiv](https://github.com/zezhishao/DailyArXiv)

ChatGPT also helped in the development of this project.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
