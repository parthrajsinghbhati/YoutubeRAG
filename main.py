import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# File paths for FAISS persistence
FAISS_INDEX_PATH = "faiss_index.bin"
DOCS_PATH = "docs.pkl"

# Global variables for caching
_groq_client = None
_embedding_model = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        cache_folder = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_folder, exist_ok=True)
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder)
    return _embedding_model

# -------------------- TRANSCRIPT HELPERS -------------------- #
def get_channel_video_ids(channel_url: str) -> list[str]:
    ydl_opts = {'extract_flat': True, 'quiet': True, 'playlistend': 100}
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            if 'entries' in info:
                return [entry['id'] for entry in info['entries'] if entry]
            elif 'id' in info:
                return [info['id']]
            return []
        except Exception:
            return []

def get_transcript(video_id: str) -> str:
    try:
        # Use the fetch method which is confirmed to work in this environment
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join([chunk.text for chunk in transcript])
    except Exception as e:
        print(f"⚠️ Transcript failed for {video_id}: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# -------------------- INDEXING -------------------- #
def index_video(video_url: str, clear_existing: bool = False):
    model = get_embedding_model()
    
    # Load existing or create new
    if not clear_existing and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "rb") as f:
            all_docs = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        all_docs = []
        # dim = 384 for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(384)

    video_ids = get_channel_video_ids(video_url)
    indexed_count = 0
    errors = []

    for video_id in video_ids:
        transcript = get_transcript(video_id)
        if not transcript:
            errors.append(f"Could not fetch transcript for {video_id}. (YouTube often blocks cloud IPs)")
            continue

        chunks = chunk_text(transcript)
        if not chunks:
            continue

        # Generate embeddings
        embeddings = model.encode(chunks)
        index.add(np.array(embeddings).astype('float32'))
        
        # Store metadata
        for chunk in chunks:
            all_docs.append({
                "text": chunk,
                "url": f"https://youtube.com/watch?v={video_id}"
            })

        indexed_count += 1

    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(all_docs, f)

    return {
        "videos_found": len(video_ids),
        "indexed_videos": indexed_count,
        "total_chunks": len(all_docs),
        "errors": errors
    }

# -------------------- QUERY -------------------- #
def query_video(question: str) -> dict:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PATH):
        return {"answer": "No videos indexed yet.", "sources": []}

    model = get_embedding_model()
    groq = get_groq_client()
    
    # Load index and docs
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        all_docs = pickle.load(f)

    # Search
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding).astype('float32'), k=5)

    # Combine results
    context_parts = []
    sources = []
    for idx in I[0]:
        if idx != -1 and idx < len(all_docs):
            context_parts.append(all_docs[idx]["text"])
            sources.append(all_docs[idx]["url"])

    if not context_parts:
        return {"answer": "No relevant info found.", "sources": []}

    context = "\n\n".join(context_parts)
    prompt = f"Answer the question using ONLY the YouTube transcript context below.\n\nContext:\n{context}\n\nQuestion: {question}"

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources))
    }