import streamlit as st
import main

st.set_page_config(page_title="YouTube RAG (FAISS)", page_icon="🎥")

st.title("🎥 YouTube Video RAG")
st.caption("Now using FAISS for faster indexing on hosted environments")

# Cache resources
@st.cache_resource
def load_embedding_model():
    return main.get_embedding_model()

# Initialize resources
load_embedding_model()

video_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Index Content") and video_url:
    with st.spinner("Indexing..."):
        status = main.index_video(video_url, clear_existing=False)
    
    if status["indexed_videos"] > 0:
        st.success(f"Successfully indexed {status['indexed_videos']} video(s)!")
    else:
        st.error("Indexing failed on the server.")
        for err in status["errors"]:
            st.warning(err)
        st.info("Try the 'Local Indexing' fallback if the server is blocked.")

st.divider()

question = st.text_input("Ask a question")

if st.button("Ask") and question:
    with st.spinner("Searching..."):
        result = main.query_video(question)

    st.write("**Answer:**")
    st.write(result["answer"])
    
    if result["sources"]:
        st.write("**Sources:**")
        for url in result["sources"]:
            st.markdown(f"- [{url}]({url})")