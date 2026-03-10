"""
Code Insight Tool — Streamlit Entry Point
"""

import streamlit as st

from config import settings


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load custom CSS
    _load_css()

    # Sidebar navigation
    with st.sidebar:
        st.title("🔍 Code Insight")
        st.caption(f"v{settings.app_version}")
        st.divider()

        page = st.radio(
            "Navigation",
            options=["📥 Ingest", "💬 Chat", "🗺️ Explore", "📊 Insights"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Powered by UniXcoder + Qdrant + Qwen")

    # Page routing
    if page == "📥 Ingest":
        from ui.pages.ingest import render_ingest_page
        render_ingest_page()
    elif page == "💬 Chat":
        st.info("🚧 Chat page — coming in Phase 2")
    elif page == "🗺️ Explore":
        st.info("🚧 Explore page — coming in Phase 3")
    elif page == "📊 Insights":
        st.info("🚧 Insights page — coming in Phase 3")


def _load_css():
    """Load custom CSS styling."""
    css = """
    <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1a1a2e;
        }

        /* Code blocks */
        .stCodeBlock {
            border-radius: 8px;
        }

        /* Stats cards */
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        .stats-card h3 {
            margin: 0;
            font-size: 2rem;
        }
        .stats-card p {
            margin: 0;
            opacity: 0.8;
            font-size: 0.9rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
