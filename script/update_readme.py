def update_readme_with_highlight_papers(hl_papers, papers, date):
    """
    Updates the README.md file with the highlight papers.
    Markdown format:
        ...
        # The Latest Daily Papers
    Target format:
        ...
        # The Latest Daily Papers
        ## Date: {date}
        - **[Title: {paper['title']}]({paper['link']})**
          - **Classification**: {paper['category']}
          - **Summary**: {paper['summary']}
          - **Score**: {paper['score']}/10
        ...
    """
    readme_file = "README.md"
    with open(readme_file, "r") as f:
        lines = f.readlines()

    # Find the index of the line with "# The Latest Daily Papers"
    index = -1
    for i, line in enumerate(lines):
        if line.startswith("# The Latest Daily Papers"):
            index = i
            break

    # If the line is found, insert the highlight papers
    if index != -1:
        for paper in hl_papers:
            lines.insert(index + 1, f"## Date: {date}\n")
            lines.insert(index + 2, f"- **[Title: {paper['title']}]({paper['link']})**\n")
            lines.insert(index + 4, f"  - **Summary**: {paper['summary']}\n")
            lines.insert(index + 5, f"  - **Score**: {paper['score']}/10\n\n")
            index += 5
        # Insert the remaining papers below the highlight papers

        for paper in papers:
            if paper not in hl_papers:
                lines.insert(index + 1, f"- **[Title: {paper['title']}]({paper['link']})**\n")

    # Write the updated content back to README.md
    with open(readme_file, "w") as f:
        f.writelines(lines)
