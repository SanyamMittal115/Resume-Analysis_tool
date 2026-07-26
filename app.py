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
def html_escape(text):
    """Escape real HTML metacharacters only. We render summary/skill text
    inside raw HTML (unsafe_allow_html=True), so markdown/LaTeX escaping is
    irrelevant there -- it just produced visible backslashes. The only thing
    that actually needs escaping in that path is HTML syntax itself."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Common suffixes that pypdf sometimes splits off into their own "word" when
# extracting justified/wrapped PDF text (e.g. "initia tives" instead of
# "initiatives"). Deliberately excludes any entry that is itself a real
# standalone English word (e.g. "full", "less", "ally", "ships", "wards") --
# those caused false merges like "the full product" -> "thefull product".
_BROKEN_SUFFIXES = (
    "tions?|ments?|ties|tives?|ing|ings|ness|able|ible|ously?|ances?|ences?|"
    "ives?|ies|ers?"
)
_SUFFIX_BREAK_RE = re.compile(rf'\b([a-z]{{3,}})\s+({_BROKEN_SUFFIXES})\b')
_SHORT_FRAGMENT_RE = re.compile(r'\b([a-z]{2,})\s+([a-z]{1,3})\b(?=[a-z])')


def fix_pdf_spacing(text):
    """Collapse stray spaces pypdf inserts inside a single word during
    extraction. Applied to the RAW extracted text, before chunking/
    classification/NER, since that's where the broken words originate --
    fixing it only in the summary left the classifier and NER model reading
    the same broken words (which is why 'Auto', 'Des', 'Deskt' etc. showed
    up as bogus skills)."""
    text = _SUFFIX_BREAK_RE.sub(r'\1\2', text)
    text = _SHORT_FRAGMENT_RE.sub(r'\1\2', text)
    return text


def summary_to_bullets(summary_text):
    """Split a summary paragraph into individual sentences so it can be
    rendered as a clean bullet list instead of one dense block of text.
    Drops a trailing fragment that doesn't end in sentence-ending
    punctuation, since that means the summarizer's max_length cut it off
    mid-sentence rather than it being a complete thought."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', summary_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences and not re.search(r'[.!?]"?$', sentences[-1]):
        sentences = sentences[:-1]
    return sentences


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
_BULLET_CHARS = "•●▪◦‣∙"


def chunksplit(txt, max_length):
    """Break extracted text into chunks for the models. A new chunk starts
    whenever a bullet marker or an ALL-CAPS section header is hit, even if
    we're under max_length -- otherwise unrelated content (e.g. a job title
    line followed by an unrelated bullet point) gets packed into the same
    chunk and the summarizer produces one run-on sentence mixing both."""
    # Bullets often land mid-line rather than at the start of a line, so
    # force each one onto its own line first.
    txt = re.sub(rf'\s*([{_BULLET_CHARS}])\s*', r'\n\1 ', txt)

    chunks = []
    current = ""
    for line in txt.split("\n"):
        line = line.strip()
        if not line:
            continue
        starts_new_unit = line[0] in _BULLET_CHARS or (line.isupper() and len(line) > 3)
        if starts_new_unit and current.strip():
            chunks.append(current.strip())
            current = line + " "
        elif len(current) + len(line) < max_length:
            current += line + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = line + " "
    if current.strip():
        chunks.append(current.strip())
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
MIN_SKILL_CONFIDENCE = 0.85


def extract_skills(chunks, ner):
    skills = set()
    for chunk in chunks:
        entities = ner(chunk)
        for ent in entities:
            word = ent["word"].strip()
            if word.startswith("##") or len(word) < 3:
                continue
            # Low-confidence hits are disproportionately fragments left
            # over from broken words (e.g. "Face" from "Facilitated"),
            # not real entities, so require the model to be fairly sure.
            if ent["score"] < MIN_SKILL_CONFIDENCE:
                continue
            # PER = person names -> not a skill, so skip those
            if ent["entity_group"] in ["ORG", "MISC"]:
                skills.add(word.replace(" ", ""))

    # Drop any skill that's just a prefix/fragment of a longer one we also
    # detected (e.g. "Auto" when "Automation" is already in the set).
    by_length = sorted(skills, key=len, reverse=True)
    kept = []
    for s in by_length:
        if any(s != k and s.lower() in k.lower() for k in kept):
            continue
        kept.append(s)
    return sorted(kept)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize_resume(chunks, summarizer, max_chunks=5):
    substantial_chunks = [c for c in chunks if len(c.split()) >= 15]
    if not substantial_chunks:
        return "Not enough resume text to summarize."

    if len(substantial_chunks) <= max_chunks:
        selected = substantial_chunks
    else:
        # Evenly spaced picks across the whole document, so early sections
        # (education) don't crowd out later ones (projects, skills) just
        # because they happen to come first in the resume.
        step = len(substantial_chunks) / max_chunks
        indices = sorted(set(int(i * step) for i in range(max_chunks)))
        selected = [substantial_chunks[i] for i in indices]

    summaries = []
    for chunk in selected:
        result = summarizer(chunk, max_length=100, min_length=25, do_sample=False)
        summaries.append(result[0]["summary_text"])
    return " ".join(summaries)


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

    # Fix word-splitting here, on the raw text, so the classifier, NER
    # model, and summarizer all read clean words -- not just the summary.
    txt = fix_pdf_spacing(txt)

    with st.expander("🔍 View Raw Extracted Resume Text"):
        st.text_area("Plain Text Content", txt, height=250)

    with st.spinner("🤖 Loading AI models and running analysis... this can take a minute on first run."):
        classifier, summarizer, ner = load_models()
        chunks = chunksplit(txt, MAX_CHUNK_LEN)
        role_scores = analyze_role_fit(chunks, classifier)
        skills = extract_skills(chunks, ner)
        summary = summarize_resume(chunks, summarizer)

    st.subheader("📝 Professional Summary")
    bullets = summary_to_bullets(summary)
    summary_html = "".join(
        f'<li style="font-family: \'Source Sans Pro\', sans-serif; font-size: 16px; '
        f'line-height: 1.6; margin-bottom: 8px;">{html_escape(b)}</li>'
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
            f'font-weight: 500; border: 1px solid #00f2fe;">{html_escape(skill)}</span>'
            for skill in skills[:25]
        )
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.write("No direct capability entities isolated.")


if __name__ == "__main__":
    main()
