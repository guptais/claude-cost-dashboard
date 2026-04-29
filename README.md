# Claude Cost Dashboard

A local Streamlit app that estimates what your Claude.ai conversations would cost at API rates, broken down by conversation, date, and model.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.x-red)

## What it does

- Parses your Claude.ai conversation export
- Estimates token counts using `tiktoken`
- Calculates cost using Anthropic's published pricing (Sonnet / Opus / Haiku)
- Visualises cost over time, token split, and top conversations
- No API calls are made.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/guptais/claude-cost-dashboard.git
cd claude-cost-dashboard
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
Note: Run `pip freeze > requirements.txt` whenever new package is installed. this is to keep the requirements.txt updated.

**4. Export your Claude.ai data**

Claude.ai → top-right avatar → Settings → Account → Export Data. You'll receive an email with a ZIP file. Extract `conversations.json` and place it in the same folder as `dashboard.py`.

**5. Run the app**

> **Note:** Make sure your venv is active (`source venv/bin/activate`) each time before running.

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`.

## Notes

- Token counts are estimates using `tiktoken` (`cl100k_base` encoding) — accurate to ~3-5%
- Cost figures are hypothetical API equivalents, not actual charges
- `conversations.json` is excluded from the repo via `.gitignore` — your chat history stays local
