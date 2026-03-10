"""
Ingest Page — UI for indexing code repositories.
"""

import streamlit as st

from core.ingestion.pipeline import IngestionPipeline
from storage.metadata_store import MetadataStore


def render_ingest_page():
    """Render the ingestion page."""
    st.header("📥 Code Ingestion")
    st.markdown(
        "Index a codebase to enable AI-powered code search & understanding."
    )

    # --- Input Section ---
    col1, col2 = st.columns([3, 1])

    with col1:
        source = st.text_input(
            "Source",
            placeholder="Git URL (https://...) or local folder path (C:\\...)",
            help="Enter a Git repository URL or a local folder path to index.",
        )

    with col2:
        project_name = st.text_input(
            "Project Name",
            placeholder="(auto-detect)",
            help="Optional. Auto-detected from URL or folder name.",
        )

    # --- Start Indexing ---
    start_button = st.button(
        "🚀 Start Indexing",
        type="primary",
        use_container_width=True,
        disabled=not source,
    )

    st.divider()

    # --- Indexing Process ---
    if start_button and source:
        _run_ingestion(source, project_name or None)

    # --- Existing Projects ---
    _show_existing_projects()


def _run_ingestion(source: str, project_name: str = None):
    """Run the ingestion pipeline with progress display."""
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    error_container = st.empty()

    def progress_callback(message: str, percent: float):
        progress_bar.progress(min(percent, 1.0), text=message)
        status_text.text(message)

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(
            source=source,
            project_name=project_name,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0, text="✅ Indexing complete!")

        # Show results
        st.success(
            f"Successfully indexed **{result.project_name}**!"
        )

        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""<div class="stats-card">
                    <h3>{result.total_files}</h3>
                    <p>Files</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="stats-card">
                    <h3>{result.total_chunks}</h3>
                    <p>Chunks</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""<div class="stats-card">
                    <h3>{result.total_errors}</h3>
                    <p>Errors</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""<div class="stats-card">
                    <h3>{result.duration_seconds:.1f}s</h3>
                    <p>Duration</p>
                </div>""",
                unsafe_allow_html=True,
            )

        # Show errors if any
        if result.errors:
            with st.expander(f"⚠️ {len(result.errors)} errors during indexing"):
                for err in result.errors:
                    st.warning(f"**{err['file']}**: {err['error']}")

    except Exception as e:
        progress_bar.progress(1.0, text="❌ Indexing failed!")
        error_container.error(f"Error: {str(e)}")


def _show_existing_projects():
    """Show list of previously indexed projects."""
    st.subheader("📂 Indexed Projects")

    try:
        store = MetadataStore()
        projects = store.get_all_projects()

        if not projects:
            st.info("No projects indexed yet. Start by entering a source above.")
            return

        for project in projects:
            stats = store.get_latest_stats(project["id"])
            with st.expander(f"📁 {project['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files", stats["total_files"] if stats else "—")
                with col2:
                    st.metric("Chunks", stats["total_chunks"] if stats else "—")
                with col3:
                    st.metric("Indexed", project["indexed_at"][:10])

                st.caption(f"Source: `{project['path']}`")
                if project.get("commit_hash"):
                    st.caption(f"Commit: `{project['commit_hash'][:8]}`")

    except Exception:
        st.caption("Unable to load project history.")
