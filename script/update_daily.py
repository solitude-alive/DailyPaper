def update_daily_papers(hl_papers, papers, date):
    """
    Updates the DailyPaper.md file with the highlight papers.
    Target format:
        # The Latest Daily Papers - Date: {date}
        ## Highlight Papers
        ### **[Title: {paper['title']}]({paper['link']})**
        - **Summary**: {paper['summary']}
        - **Score**: {paper['score']}/10
        ...
        ## Other Papers
    """
    file = "DailyPaper.md"

    with open(file, "w") as f:
        f.write(f"# The Latest Daily Papers - Date: {date}\n")
        f.write("## Highlight Papers\n")
        for paper in hl_papers:
            f.write(f"### **[Title: {paper['title']}]({paper['link']})**\n")
            f.write(f"- **Summary**: {paper['summary']}\n")
            f.write(f"- **Score**: {paper['score']}/10\n\n")
        f.write("## Other Papers\n")
        for paper in papers:
            f.write(f"### **[Title: {paper['title']}]({paper['link']})**\n")
