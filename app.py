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
    st.write(summary)
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
