"""
ClientPulse — AI-Powered Account Insights Dashboard
-----------------------------------------------------
Reads mock client/pipeline data and surfaces upsell / retention actions
based on a weighted health-scoring model, with an optional LLM layer
(Anthropic Claude) for narrative recommendations.

Run locally:
    pip install streamlit pandas anthropic
    streamlit run app.py

Deploy free & public:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="ClientPulse — AI Account Insights", layout="wide")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path="mock_clients.csv"):
    return pd.read_csv(path)

df = load_data()

# ---------------------------------------------------------------------------
# 2. SCORING ENGINE
# ---------------------------------------------------------------------------
def health_score(row):
    """
    Weighted 0-100 health score. Lower = more urgent.
    Signals:
      - recency of activity (heavier weight, most predictive of churn)
      - documents pending (friction / stalled deals)
      - proximity to renewal (urgency window)
      - communication responsiveness score (0-10 given)
      - past renewal loyalty (small positive weight)
    """
    recency_penalty = min(row["days_since_last_activity"], 100) * 0.8
    docs_penalty = row["documents_pending"] * 7
    renewal_urgency_penalty = max(0, (30 - row["days_to_renewal"])) * 1.2 if row["days_to_renewal"] <= 30 else 0
    comms_adjustment = (row["communication_score"] - 5) * 3  # centered so avg comms is neutral
    loyalty_bonus = row["past_renewals"] * 2

    score = 100 - recency_penalty - docs_penalty - renewal_urgency_penalty + comms_adjustment + loyalty_bonus
    return round(max(0, min(100, score)), 1)


def classify(row):
    score = row["health_score"]
    near_renewal = row["days_to_renewal"] <= 30
    stalled = row["file_status"] in ("Pending Docs", "Stalled")

    if score < 40 and near_renewal:
        return "🔴 At-Risk — Renewal in danger"
    if score < 40 and stalled:
        return "🔴 At-Risk — File stalled"
    if score < 55:
        return "🟠 Needs Attention"
    if score >= 75 and row["commission_value"] > 4000:
        return "🟢 Upsell Opportunity"
    return "🟡 Healthy — Monitor"


def reason(row):
    reasons = []
    if row["days_since_last_activity"] > 30:
        reasons.append(f"no contact in {row['days_since_last_activity']} days")
    if row["documents_pending"] > 2:
        reasons.append(f"{row['documents_pending']} documents still pending")
    if row["days_to_renewal"] <= 30:
        reasons.append(f"renewal due in {row['days_to_renewal']} days")
    if row["communication_score"] <= 3:
        reasons.append("low responsiveness")
    if row["commission_value"] > 4000 and row["health_score"] >= 75:
        reasons.append("high-value file in good standing — cross-sell candidate")
    if not reasons:
        reasons.append("no red flags detected")
    return "; ".join(reasons)


df["health_score"] = df.apply(health_score, axis=1)
df["segment"] = df.apply(classify, axis=1)
df["reason"] = df.apply(reason, axis=1)

# ---------------------------------------------------------------------------
# 3. OPTIONAL LLM NARRATIVE LAYER (Anthropic Claude)
# ---------------------------------------------------------------------------
def generate_llm_action(row, api_key):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a mortgage brokerage retention advisor. Given this client file, write ONE
short, specific action the broker should take today (max 2 sentences, no preamble):

Client: {row['client_name']}
File type: {row['file_type']}
Status: {row['file_status']}
Days since last activity: {row['days_since_last_activity']}
Documents pending: {row['documents_pending']}
Days to renewal: {row['days_to_renewal']}
Communication score (0-10): {row['communication_score']}
Health score: {row['health_score']}
Segment: {row['segment']}
"""
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"(LLM call failed: {e})"

# ---------------------------------------------------------------------------
# 4. UI
# ---------------------------------------------------------------------------
st.title("📊 ClientPulse")
st.caption("AI-powered account insights — reviews client/pipeline data and flags upsell or retention actions based on behavioral patterns.")

with st.sidebar:
    st.header("⚙️ Options")
    segment_filter = st.multiselect(
        "Filter by segment",
        options=sorted(df["segment"].unique()),
        default=sorted(df["segment"].unique()),
    )
    st.divider()
    st.subheader("Optional: AI Narrative Layer")
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                             help="Leave blank to use the rule-based engine only. Nothing is stored.")
    st.caption("This app runs fully without a key. The key, if entered, is only used for this session's requests.")

filtered = df[df["segment"].isin(segment_filter)].sort_values("health_score")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clients", len(df))
col2.metric("At-Risk", (df["segment"].str.contains("At-Risk")).sum())
col3.metric("Upsell Opportunities", (df["segment"].str.contains("Upsell")).sum())
col4.metric("Avg Health Score", round(df["health_score"].mean(), 1))

st.divider()

for _, row in filtered.iterrows():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 3])
        with c1:
            st.markdown(f"**{row['client_name']}**  \n{row['file_type']} • {row['file_status']}")
        with c2:
            st.metric("Health", f"{row['health_score']}")
            st.markdown(row["segment"])
        with c3:
            st.markdown(f"**Signal:** {row['reason']}")
            if api_key:
                if st.button("Generate AI action", key=f"btn_{row['client_id']}"):
                    with st.spinner("Thinking..."):
                        action = generate_llm_action(row, api_key)
                    st.info(action)

st.divider()
st.caption(
    "Scoring model: weighted combination of activity recency, document friction, renewal urgency, "
    "communication responsiveness, and loyalty history. Segments and reasons are rule-derived; the "
    "optional LLM layer adds a natural-language recommendation per client on demand."
)
