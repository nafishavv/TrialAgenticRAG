"""Streamlit UI for RAGTrial chatbot.

Thin presentation layer — imports only ChatSession from the conversation layer.
Zero RAG logic here. When migrating to a backend (FastAPI), only the
`get_session()` factory and `session.ask()` callsite change.

Run:
    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ragtrial.chat import ChatSession

st.set_page_config(
    page_title="Chatbot Layanan Publik Kab. Batang",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============ Session factory ============
def _new_session() -> ChatSession:
    return ChatSession(
        mode=st.session_state.get("mode", "enhanced"),
        max_history_turns=st.session_state.get("max_turns", 5),
        rewrite_followups=st.session_state.get("rewrite", True),
    )


def get_session() -> ChatSession:
    """Singleton ChatSession per Streamlit session; rebuilds if mode changed."""
    sess = st.session_state.get("chat_session")
    if sess is None or sess.mode != st.session_state.get("mode", "enhanced"):
        st.session_state.chat_session = _new_session()
    return st.session_state.chat_session


def reset_session() -> None:
    """Rebuild ChatSession with current sidebar settings."""
    st.session_state.chat_session = _new_session()
    st.session_state.messages = []


# ============ Styling ============
_CSS = """
<style>
:root {
  --navy: #1E3A8A;
  --ink: #0F172A;
  --slate: #334155;
  --muted: #64748B;
  --border: #E2E8F0;
  --surface-side: #F1F5F9;
  --surface-main: #F8FAFC;
  --teal: #0D9488;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 12px;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', 'Segoe UI', 'Roboto', system-ui, sans-serif !important;
  color: var(--ink);
}

/* Hide default Streamlit chrome to feel less template-y */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

[data-testid="stAppViewContainer"] > .main { background: var(--surface-main); }
.block-container { padding-top: 1.25rem !important; padding-bottom: 6rem !important; max-width: 100% !important; }

/* ---------- Header ---------- */
.gov-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; margin: 0 auto 18px auto; max-width: 880px;
  background: #FFFFFF; border: 1px solid var(--border); border-radius: var(--radius-lg);
  box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.gov-header .brand-title { font-size: 1.15rem; font-weight: 600; color: var(--ink); letter-spacing: -0.01em; margin: 0; }
.gov-header .brand-sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; letter-spacing: 0.01em; }
.gov-status { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; color: var(--slate);
  background: #ECFDF5; border: 1px solid #A7F3D0; padding: 4px 10px; border-radius: 999px; font-weight: 500; }
.gov-status .dot { width: 7px; height: 7px; border-radius: 50%; background: #10B981; box-shadow: 0 0 0 3px rgba(16,185,129,0.15); }

/* ---------- Chat container width ---------- */
[data-testid="stChatMessage"], .chat-scope { max-width: 880px; margin-left: auto; margin-right: auto; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 4px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  border-radius: var(--radius-md);
  padding: 12px 16px;
  border: 1px solid var(--border);
  background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
  font-size: 0.94rem;
  line-height: 1.55;
  color: var(--ink);
}
/* user bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
  background: #E2E8F0;
  border-color: #CBD5E1;
}
/* avatars monochrome */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
  background: var(--navy) !important; color: #FFFFFF !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
  background: var(--slate) !important;
}

/* ---------- Metric badges ---------- */
.metric-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px 0; max-width: 880px; margin-left: auto; margin-right: auto; }
.metric-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface-side); border: 1px solid var(--border); color: var(--slate);
  font-size: 0.72rem; font-weight: 500; padding: 4px 10px; border-radius: 999px;
  font-variant-numeric: tabular-nums; letter-spacing: 0.01em;
}
.metric-badge .ico { color: var(--navy); font-weight: 600; }

/* ---------- Service chips ---------- */
.chips-header { max-width: 880px; margin: 8px auto 8px auto; }
.chips-header h3 { font-size: 0.8rem; font-weight: 600; color: var(--slate); text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 4px 0; }
.chips-header p { font-size: 0.85rem; color: var(--muted); margin: 0 0 12px 0; }

.chip-row [data-testid="stHorizontalBlock"] { max-width: 880px; margin-left: auto; margin-right: auto; gap: 8px !important; }
.chip-row .stButton > button {
  width: 100%;
  background: #FFFFFF;
  color: var(--navy);
  border: 1px solid #CBD5E1;
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 0.82rem;
  font-weight: 500;
  transition: all 0.15s ease;
  box-shadow: none;
}
.chip-row .stButton > button:hover {
  background: var(--navy);
  color: #FFFFFF;
  border-color: var(--navy);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface-side); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

.side-section-label {
  font-size: 0.7rem; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin: 18px 0 8px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border);
}
.side-section-label:first-child { margin-top: 4px; }

.kb-item {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 12px; background: #FFFFFF; border: 1px solid var(--border);
  border-radius: var(--radius-md); margin-bottom: 6px;
}
.kb-item .kb-icon { color: var(--navy); font-size: 0.9rem; line-height: 1.2; flex-shrink: 0; }
.kb-item .kb-title { font-size: 0.8rem; font-weight: 600; color: var(--ink); }
.kb-item .kb-desc { font-size: 0.72rem; color: var(--muted); margin-top: 2px; line-height: 1.4; }

.side-footer { font-size: 0.7rem; color: var(--muted); text-align: center; margin-top: 24px; padding-top: 12px; border-top: 1px solid var(--border); }

[data-testid="stSidebar"] .stButton > button {
  background: #FFFFFF; color: var(--slate); border: 1px solid #CBD5E1;
  border-radius: var(--radius-md); font-weight: 500; transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover { border-color: var(--navy); color: var(--navy); }

/* ---------- Input area ---------- */
[data-testid="stChatInput"] {
  max-width: 880px; margin: 0 auto;
  background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: var(--radius-lg);
  box-shadow: 0 -1px 12px rgba(15,23,42,0.04);
}
[data-testid="stChatInput"] textarea { font-family: inherit !important; font-size: 0.92rem !important; }
[data-testid="stChatInput"] button { color: var(--navy) !important; }
[data-testid="stBottom"] { background: linear-gradient(to bottom, rgba(248,250,252,0), var(--surface-main) 30%); }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
  max-width: 880px; margin: 0 auto 8px auto;
  border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important;
  background: #FFFFFF;
}
[data-testid="stExpander"] summary { font-size: 0.8rem !important; color: var(--slate) !important; font-weight: 500; }
[data-testid="stExpander"] summary:hover { color: var(--navy) !important; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  .gov-header { flex-direction: column; align-items: flex-start; gap: 8px; }
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
"""


def _inject_css() -> None:
    try:
        st.html(_CSS)
    except AttributeError:
        st.markdown(_CSS, unsafe_allow_html=True)


# ============ Header ============
def _render_header() -> None:
    st.markdown(
        """
        <div class="gov-header">
          <div>
            <div class="brand-title">Chatbot Layanan Publik</div>
            <div class="brand-sub">Sistem Informasi Pemerintahan Kabupaten Batang</div>
          </div>
          <div class="gov-status"><span class="dot"></span>System Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============ Sidebar ============
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="side-section-label">Konfigurasi RAG</div>', unsafe_allow_html=True)
        st.selectbox(
            "RAG mode",
            options=["naive", "enhanced", "agentic"],
            index=2,
            key="mode",
            help="naive=baseline fan-out | enhanced=semantic+dense pipeline | agentic=tool-calling loop",
        )
        st.slider(
            "Max history turns",
            min_value=1,
            max_value=20,
            value=5,
            key="max_turns",
            help="Jumlah turn terakhir yang diingat (1 turn = user + bot)",
        )
        st.checkbox(
            "LLM query rewriting",
            value=True,
            key="rewrite",
            help="Rewrite follow-up questions jadi standalone query sebelum retrieve",
        )

        st.markdown('<div class="side-section-label">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="kb-item">
              <div class="kb-icon">📄</div>
              <div>
                <div class="kb-title">Dukcapil</div>
                <div class="kb-desc">Buku Saku Dafduk Capil Kab. Batang 2023</div>
              </div>
            </div>
            <div class="kb-item">
              <div class="kb-icon">📄</div>
              <div>
                <div class="kb-title">OPD</div>
                <div class="kb-desc">Direktori Nama & Alamat OPD Kab. Batang</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-label">Sesi</div>', unsafe_allow_html=True)
        if st.button("Clear chat", use_container_width=True):
            reset_session()
            st.rerun()

        st.markdown(
            '<div class="side-footer">RAGTrial · Tugas Akhir 2026</div>',
            unsafe_allow_html=True,
        )


# ============ Service chips ============
_SERVICE_CHIPS: list[tuple[str, str]] = [
    ("Perizinan", "Bagaimana cara mengurus izin usaha mikro?"),
    ("Kependudukan", "Apa syarat membuat KTP baru?"),
    ("Pajak Daerah", "Informasi pajak daerah Kabupaten Batang?"),
    ("Hukum & Peraturan", "Peraturan daerah terkait perizinan usaha?"),
    ("Informasi OPD", "Alamat dan kontak Dinas Dukcapil?"),
]


def _render_service_chips() -> str | None:
    """Render clickable service chips. Returns the template query if a chip was clicked."""
    st.markdown(
        """
        <div class="chips-header">
          <h3>Mulai Percakapan</h3>
          <p>Pilih topik layanan untuk pertanyaan cepat, atau ketik pertanyaanmu di bawah.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(_SERVICE_CHIPS))
    clicked: str | None = None
    for col, (label, query) in zip(cols, _SERVICE_CHIPS):
        with col:
            if st.button(label, key=f"chip_{label}", use_container_width=True):
                clicked = query
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


# ============ Metrics ============
def _render_metrics(meta: dict) -> None:
    t = meta["timings"]
    n_docs = len(meta["documents"])
    badges = (
        f'<div class="metric-row">'
        f'<span class="metric-badge"><span class="ico">⏱</span>Retrieve {t["retrieve"]:.2f}s</span>'
        f'<span class="metric-badge"><span class="ico">⚙</span>Generate {t["generate"]:.2f}s</span>'
        f'<span class="metric-badge"><span class="ico">Σ</span>Total {t["total"]:.2f}s</span>'
        f'<span class="metric-badge"><span class="ico">📄</span>Docs {n_docs}</span>'
        f"</div>"
    )
    st.markdown(badges, unsafe_allow_html=True)

    with st.expander("Lihat detail retrieval"):
        if meta["rewritten_query"] != meta["original_query"]:
            st.markdown(f"**Query rewritten:** `{meta['rewritten_query']}`")
        st.markdown("**Retrieved docs:**")
        if not meta["documents"]:
            st.caption("_Tidak ada dokumen yang diretrieve._")
        for i, doc in enumerate(meta["documents"], 1):
            src = doc.metadata.get("_source", "?")
            st.markdown(f"`[{i}] {src}` — {doc.page_content[:200]}...")


# ============ Main ============
_inject_css()
_render_header()
_render_sidebar()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and "meta" in msg:
        _render_metrics(msg["meta"])

# Service chips (only when no conversation yet)
chip_query: str | None = None
if not st.session_state.messages:
    chip_query = _render_service_chips()

# Resolve user input from chat_input OR chip click
typed = st.chat_input("Tanyakan mengenai KTP, perizinan usaha, atau informasi OPD Kabupaten Batang…")
user_input = typed or chip_query

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    session = get_session()
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban…"):
            try:
                result = session.ask(user_input)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(result["answer"])

    _render_metrics(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": result,
        }
    )

    # If chip was used, rerun to clear chip row (messages now non-empty)
    if chip_query and not typed:
        st.rerun()
