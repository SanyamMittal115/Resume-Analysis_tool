import re
import streamlit as st
from transformers import pipeline
from pypdf import PdfReader

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="centered")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_CHUNK_LEN = 800
TARGET_ROLES = ["Product Manager", "AI Engineer", "Business Analyst", "Solutions Engineer"]

# Lighter models than the original bart-large-mnli / bart-large-cnn so the
# app fits comfortably in Streamlit Community Cloud's free-tier RAM.
CLASSIFIER_MODEL = "valhalla/distilbart-mnli-12-3"   # ~440MB, vs 1.6GB for bart-large-mnli
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"    # ~305MB, vs 1.6GB for bart-large-cnn
NER_MODEL = "dslim/bert-base-NER"                     # ~430MB


@st.cache_resource(show_spinner=False)
def load_models():
    """Cached so models load once per app session. Forces full (non-meta)
    weight materialization on CPU to avoid the 'Tensor on device cpu is not
    on the expected device meta' error."""
    classifier = pipeline(
        "zero-shot-classification",
        model=CLASSIFIER_MODEL,
        device=-1,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        device=-1,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    ner = pipeline(
        "ner",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=-1,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier, summarizer, ner


# ---------------------------------------------------------------------------
# Display-safety helpers
# ---------------------------------------------------------------------------
def escape_markdown(text):
    """Escape characters that Streamlit's markdown/LaTeX renderer would
    otherwise interpret on their own (e.g. '$12K' turning into a LaTeX math
    expression, or '*' / '_' triggering bold/italic mid-sentence)."""
    for ch in ["$", "*", "_", "`", "#"]:
        text = text.replace(ch, "\\" + ch)
    return text


def fix_pdf_spacing(text):
    """Collapse stray spaces that pypdf sometimes inserts inside a single
    word when extracting justified/wrapped PDF text (e.g. 'initia tives'
    instead of 'initiatives')."""
    # Join a lowercase fragment of <=3 letters back onto the word before it
    # when it's immediately followed by more lowercase letters, which is the
    # signature of a broken word rather than two real words.
    text = re.sub(r'\b([a-z]{2,})\s+([a-z]{1,3})\b(?=[a-z])', r'\1\2', text)
    return text


def summary_to_bullets(summary_text):
    """Split a summary paragraph into individual sentences so it can be
    rendered as a clean bullet list instead of one dense block of text."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', summary_text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def text_extract(file_obj):
    reader = PdfReader(file_obj)
    txt = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            txt += page_text + "\n"

    if not txt.strip():
        return None
    return txt


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunksplit(txt, max_length):
    chunks = []
    txtsum = ""
    for line in txt.split("\n"):
        if not line.strip():
            continue
        if len(txtsum) + len(line) < max_length:
            txtsum += line + " "
        else:
            if txtsum.strip():
                chunks.append(txtsum.strip())
            txtsum = line + " "
    if txtsum.strip():
        chunks.append(txtsum.strip())
    return chunks


# ---------------------------------------------------------------------------
# Role-fit analysis
# ---------------------------------------------------------------------------
def analyze_role_fit(chunks, classifier):
    scores = {role: 0.0 for role in TARGET_ROLES}
    for chunk in chunks:
        result = classifier(chunk, TARGET_ROLES)
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] += score

    total = sum(scores.values())
    if total > 0:
        for role in scores:
            scores[role] = round(scores[role] / total, 3)
    return scores


# ---------------------------------------------------------------------------
# Skill / keyword extraction
# ---------------------------------------------------------------------------
def extract_skills(chunks, ner):
    skills = set()
    for chunk in chunks:
        entities = ner(chunk)
        for ent in entities:
            word = ent["word"].strip()
            if word.startswith("##") or len(word) <= 1:
                continue
            # PER = person names -> not a skill, so skip those
            if ent["entity_group"] in ["ORG", "MISC"]:
                skills.add(word.replace(" ", ""))
    return sorted(skills)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize_resume(chunks, summarizer):
    summaries = []
    for chunk in chunks[:3]:
        if len(chunk.split()) < 15:
            continue
        result = summarizer(chunk, max_length=100, min_length=25, do_sample=False)
        summaries.append(result[0]["summary_text"])
    return " ".join(summaries) if summaries else "Not enough resume text to summarize."


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("📄 AI Resume Analysis Tool")
    st.write(
        "Upload a resume as a PDF to get an AI-generated summary, role-fit "
        "scoring against common job targets, and a list of detected skills."
    )

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file is None:
        st.info("👆 Upload a PDF resume to get started.")
        return

    txt = text_extract(uploaded_file)
    if txt is None:
        st.error("PDF is not readable or appears to be an empty scanned image.")
        return

    with st.expander("🔍 View Raw Extracted Resume Text"):
        st.text_area("Plain Text Content", txt, height=250)

    with st.spinner("🤖 Loading AI models and running analysis... this can take a minute on first run."):
        classifier, summarizer, ner = load_models()
        chunks = chunksplit(txt, MAX_CHUNK_LEN)
        role_scores = analyze_role_fit(chunks, classifier)
        skills = extract_skills(chunks, ner)
        summary = summarize_resume(chunks, summarizer)

    st.subheader("📝 Professional Summary")
    clean_summary = fix_pdf_spacing(summary)
    bullets = summary_to_bullets(clean_summary)
    summary_html = "".join(
        f'<li style="font-family: \'Source Sans Pro\', sans-serif; font-size: 16px; '
        f'line-height: 1.6; margin-bottom: 8px;">{escape_markdown(b)}</li>'
        for b in bullets
    )
    st.markdown(f'<ul style="padding-left: 20px;">{summary_html}</ul>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📊 Target Role Match Rating")
    for role, score in sorted(role_scores.items(), key=lambda x: -x[1]):
        st.write(f"**{role}**: `{int(score * 100)}% Match`")
    st.markdown("---")

    st.subheader("🛠️ Technical Capabilities & Keywords")
    if skills:
        badges_html = "".join(
            f'<span style="background-color: #1e1e1e; color: #00f2fe; padding: 5px 12px; '
            f'margin: 5px; border-radius: 15px; display: inline-block; font-size: 14px; '
            f'font-weight: 500; border: 1px solid #00f2fe;">{skill}</span>'
            for skill in skills[:25]
        )
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.write("No direct capability entities isolated.")


if __name__ == "__main__":
    main()
