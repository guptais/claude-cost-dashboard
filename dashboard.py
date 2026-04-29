import streamlit as st
import json
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
import tiktoken
import plotly.express as px

st.set_page_config(page_title="Claude Cost Dashboard", layout="wide")
st.title("Claude Session Cost Dashboard")


DATA_FILE = Path("/Users/isha/Downloads/data-f2915dd0-12fc-4a66-ba67-0fed48d71c42-1777464899-da689454-batch-0000/conversations.json")

# -- Load Data --
if not DATA_FILE.exists():
    st.error("conversations.json file not found")
    st.stop()

with open(DATA_FILE) as f:
    raw = json.load(f)

# st.header("Step 1: Raw data explorer")

# st.metric("Total conversations", len(raw))

# st.subheader("What one conversation looks like")
# st.json(raw[0])

# -- Parse into flat rows --
rows = []

for convo in raw:
    convo_id = convo.get("uuid")
    convo_name = convo.get("name") or "Untitled"
    created_at = convo.get("created_at")

    for msg in convo.get("chat_messages", []):
        text = msg.get("text") or ""
        sender = msg.get("sender")

        rows.append(
            {
                "convo_id": convo_id,
                "convo_name": convo_name,
                "convo_date": created_at,
                "msg_id": msg.get("uuid"),
                "sender": sender,
                "text": text,
                "char_count": len(text),
                "msg_date": msg.get("created_at"),
            }
        )

df = pd.DataFrame(rows)

df["convo_date"] = pd.to_datetime(df["convo_date"], utc=True)
df["msg_date"]   = pd.to_datetime(df["msg_date"],   utc=True)

# --- Step 6: Sidebar filters ---
st.sidebar.header("Filters")

# -- Model selector (changes pricing) --
MODEL_PRICING = {
    "Claude Sonnet 4.5": {"input": 3.00,  "output": 15.00},
    "Claude Opus 4":     {"input": 15.00, "output": 75.00},
    "Claude Haiku 4.5":  {"input": 0.80,  "output": 4.00},
}

selected_model = st.sidebar.selectbox(
    "Model (for pricing)",
    options=list(MODEL_PRICING.keys()),
    index=0,
)

# Pricing per million tokens (Claude Sonnet 4.5)
INPUT_PRICE_PER_M  = MODEL_PRICING[selected_model]["input"]   # human messages
OUTPUT_PRICE_PER_M = MODEL_PRICING[selected_model]["output"]  # assistant messages

# -- Date range filter --
min_date = df["convo_date"].dt.date.min()
max_date = df["convo_date"].dt.date.max()

date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# -- Conversation name search --
search_term = st.sidebar.text_input("Search conversation name", "")

# -- Apply filters to df --
mask = (
    (df["convo_date"].dt.date >= date_from) &
    (df["convo_date"].dt.date <= date_to)
)
if search_term:
    mask &= df["convo_name"].str.contains(search_term, case=False, na=False)

df = df[mask]

# --- Step 3: Token estimation ---

# Anthropic's tokenizer counts tokens the same way the API does.
# We use a dummy client purely for the count_tokens helper — no API calls made
# client = Anthropic(api_key="dummy")

# @st.cache_data(show_spinner="Counting tokens...")
# def estimate_tokens(texts: tuple) -> list[int]:
#     # st.cache_data stores the result so this only runs once,
#     # not on every Streamlit re-render. We pass a tuple (not list)
#     # because cache keys must be hashable.
#     counts = []
#     # for text in texts:
#     #     if not text.strip():
#     #         counts.append(0)
#     #     else:
#     #         resp = client.messages.count_tokens(
#     #             model="claude-sonnet-4-5",
#     #             messages=[{"role": "user", "content": text}]
#     #         )
#     #         counts.append(resp.input_tokens)
#     # return counts

enc = tiktoken.get_encoding("cl100k_base") # GPT-4 encoding, close to Claude

def estimate_tokens(text: str) -> int:
    return len(enc.encode(text=text)) if text.strip() else 0

# Only run on non-empty texts — skip blanks
df["tokens"] = df["text"].fillna("").apply(estimate_tokens)

# st.header("Step 2: Parsed messages")

# st.metric("Total conversations", df["convo_id"].nunique())
# st.metric("Total messages", len(df))
# st.metric("Human messages", len(df[df["sender"] =="human"]))
# st.metric("Assistant messages", len(df[df["sender"] =="assistant"]))

# st.subheader("Message Table")
# st.dataframe(
#     df[["convo_name", "sender", "char_count","msg_date"]],
#     width='stretch'
# )

# --- Display ---
# st.header("Step 3: Token Counts")

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total conversations", df["convo_id"].nunique())
# col2.metric("Total messages", len(df))
# col3.metric("Total Tokens", f"{df["tokens"].sum():,}")
# col4.metric("Assistant messages", f"{df["tokens"].mean():,.0f}")

# st.subheader("Message Table")
# st.dataframe(
#     df[["convo_name", "sender", "tokens", "char_count","msg_date"]],
#     width='stretch'
# )


# --- Step 4: Cost calculation ---
df["cost_usd"] = df.apply(
    lambda row: (row["tokens"] / 1_000_000) * (
        OUTPUT_PRICE_PER_M if row["sender"] == "assistant" else INPUT_PRICE_PER_M
    ),
    axis=1
)

# Roll up to conversation level for summary view
convo_df = (
    df.groupby(["convo_id", "convo_name", "convo_date"])
    .agg(
        total_tokens = ("tokens", "sum"),
        total_cost   = ("cost_usd", "sum"),
        message_count = ("msg_id", "count"),
    )
    .reset_index()
    .sort_values("convo_date", ascending=False)
    .reset_index(drop=True)
)

# --- Display ---
st.header("Claude Cost Dashboard")
st.caption(f"Pricing: {selected_model}  ·  Input ${INPUT_PRICE_PER_M}/M  ·  Output ${OUTPUT_PRICE_PER_M}/M")

# st.header("Step 4: Cost Breakdown")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversations", df["convo_id"].nunique())
col2.metric("Messages",      len(df))
col3.metric("Total tokens:", f"{df["tokens"].sum():,.0f}")
col4.metric("Estimated Total Cost", f"${df["cost_usd"].sum():.2f}")

# -- Daily stats --
daily_df = (
    df.groupby(df["convo_date"].dt.date)
    .agg(daily_cost=("cost_usd", "sum"))
    .reset_index()
    .rename(columns={"convo_date":"date"})
) 

st.subheader("Daily cost summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg cost / day", f"${daily_df["daily_cost"].mean():.2f}")
col2.metric("Avg cost / day", f"${daily_df["daily_cost"].max():.2f}")
col3.metric("Avg cost / day", f"${daily_df["daily_cost"].min():.2f}")

st.subheader("Cost by conversation")
st.dataframe(
    convo_df[["convo_name", "convo_date", "message_count", "total_tokens", "total_cost"]],
    column_config={
        "convo_name": st.column_config.TextColumn("Conversation"),
        "convo_date": st.column_config.DatetimeColumn("Date",format="DD-MM-YYYY"),
        "message_count": st.column_config.NumberColumn("Messages"),
        "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
        "total_cost": st.column_config.NumberColumn("Cost (USD)", format="$%.4f"),
    },
    width='stretch',
    hide_index=True,
)

st.header("Step 5: Charts")

# -- Chart 1: Cost over time --
# Group by date (day level) and sum cost
daily_df = (
    df.groupby(df["convo_date"].dt.date) # converts a datetime column to just the date par
    .agg(daily_cost=("cost_usd","sum"))
    .reset_index()
    .rename(columns={"convo_date":"date"})
)

fig_timeline = px.bar(
    daily_df,
    x="date",
    y="daily_cost",
    title="Estimated cost over time",
    labels={"date": "Date", "daily_cost": "Cost(USD)"},
)
fig_timeline.update_layout(yaxis_tickprefix="$")
st.plotly_chart(fig_timeline, width='stretch')

# -- Chart 2: Token split — input vs output --
# Human messages = input tokens, assistant = output tokens
token_split = (
    df.groupby("sender")["tokens"].sum()
    .reset_index()
    .replace({"human": "Input(human)", "assistant": "Output ( assistant)"})
)

fig_split = px.pie(
    token_split,
    names="sender",
    values="tokens",
    title="Token split - input vs output",
    color_discrete_sequence=["#636EFA", "#EF553B"],
)
st.plotly_chart(fig_split, width='stretch')

# -- Chart 3: Top 10 conversations by cost --
top10 = convo_df.head(10)  # already sorted by date, sort by cost instead
top10 = convo_df.nlargest(10, "total_cost") # Pandas shorthand for sort + head, picks the 10 rows with highest cost.

fig_top = px.bar(
    top10,
    x="total_cost",
    y="convo_name",
    orientation="h", # flips the bar chart horizontal, much easier to read when the labels are long conversation names.
    title="Top 10 conversations by estimated cost",
    labels={"total_cost": "Cost (USD)", "convo_name": "Conversation"},
)

fig_top.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_tickprefix="$",
)
st.plotly_chart(fig_top, width='stretch')