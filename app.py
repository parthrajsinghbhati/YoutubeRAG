import streamlit as st
import main

st.set_page_config(page_title="YouTube Video RAG", page_icon="🎥")

st.title("🎥 YouTube Video RAG")
st.caption("Ask anything about a specific YouTube video or channel")

# Cache resources to prevent re-loading on every interaction
@st.cache_resource
def load_embedding_function():
    return main.get_embedding_function()

@st.cache_resource
def load_chroma_client():
    return main.get_chroma_client()

# Initialize resources
load_embedding_function()
load_chroma_client()

video_url = st.text_input(
    "YouTube URL (Video or Channel)",
    placeholder="https://www.youtube.com/watch?v=... or https://www.youtube.com/@channel/videos"
)

if st.button("Index Content") and video_url:
    with st.spinner("Indexing content... This may take a minute."):
        status = main.index_video(video_url)
    
    if status["indexed_videos"] > 0:
        st.success(f"Indexed {status['indexed_videos']} video(s) with {status['total_chunks']} total segments!")
    else:
        st.error("Failed to index any videos.")
        if status["errors"]:
            for err in status["errors"]:
                st.warning(err)
        st.info("Note: YouTube transcripts are often blocked by hosting providers like Streamlit Cloud. You might need to use a proxy or run this locally.")

st.divider()

question = st.text_input("Ask a question about the indexed content")

if st.button("Ask") and question:
    with st.spinner("Searching and generating answer..."):
        result = main.query_video(question)

    st.write("**Answer:**")
    st.write(result["answer"])
    
    if result["sources"]:
        st.write("**Sources:**")
        for url in result["sources"]:
            st.markdown(f"- [{url}]({url})")