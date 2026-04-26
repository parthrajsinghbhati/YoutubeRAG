import streamlit as st
from main import index_video, query_video

st.title("🎥 YouTube Video RAG")
st.caption("Ask anything about a specific YouTube video")

video_url = st.text_input(
    "Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Index Video") and video_url:
    with st.spinner("Indexing video..."):
        index_video(video_url)
    st.success("Video indexed! Start asking questions.")

question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Searching..."):
        result = query_video(question)

    st.write("**Answer:**", result["answer"])
    st.write("**Sources:**")

    for url in result["sources"]:
        st.markdown(f"- [{url}]({url})")