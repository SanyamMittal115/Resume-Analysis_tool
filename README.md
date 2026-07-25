---
title: Resume Analysis Tool
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---

# AI Resume Analysis Tool

Upload a resume as a PDF and get:

1. **Professional Summary** — AI-generated summary of the resume (`sshleifer/distilbart-cnn-12-6`)
2. **Role Fit Scoring** — zero-shot classification against target roles: Product Manager,
   AI Engineer, Business Analyst, Solutions Engineer (`facebook/bart-large-mnli`)
3. **Skill/Keyword Extraction** — named entity recognition to surface tools, technologies,
   and organizations mentioned in the resume (`dslim/bert-base-NER`)

The resume text is broken into ~800-character chunks before being run through each model.

## Why Gradio instead of Docker

This Space runs on the **Gradio SDK**, which is free and doesn't require a paid Docker
plan. Gradio Spaces run your Python app directly on HF's managed CPU containers — no
`Dockerfile` needed. If you already have Docker access, you can still containerize this app;
just install `requirements.txt` and run `python app.py` (or use gradio's built-in `launch()`
server) instead of `streamlit run`.

## Local development

```bash
pip install -r requirements.txt
python app.py
```
