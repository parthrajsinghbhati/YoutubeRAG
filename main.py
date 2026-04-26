import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# Load environment variables
load_dotenv()

# Global variables for laziness (will be managed by app.py's cache)
_groq_client = None
_embedding_function = None
_chroma_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        # Using a specific cache folder to avoid permission issues in hosted environments
        cache_folder = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_folder, exist_ok=True)
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            cache_folder=cache_folder
        )
    return _embedding_function

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        # Use PersistentClient for disk persistence
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client

def get_collection():
    client = get_chroma_client()
    emb_fn = get_embedding_function()
    return client.get_or_create_collection(
        name="youtube_channel",
        embedding_function=emb_fn
    )

# -------------------- STEP 1: GET VIDEO IDS -------------------- #
def get_channel_video_ids(channel_url: str) -> list[str]:
    """Fetch all video IDs from a YouTube channel or a single video."""
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'playlistend': 100
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' in info:
                return [entry['id'] for entry in info['entries'] if entry]
            elif 'id' in info:
                return [info['id']]
            else:
                return []
        except Exception as e:
            print(f"❌ Error fetching video IDs: {e}")
            return []


# -------------------- STEP 2: GET TRANSCRIPT -------------------- #
def get_transcript(video_id: str) -> str:
    """Fetch transcript for a single video."""
    try:
        # Note: In hosted environments, this might fail due to IP blocking.
        transcript_api = YouTubeTranscriptApi()
        transcript = transcript_api.fetch(video_id)
        return " ".join([chunk.text for chunk in transcript])
    except Exception as e:
        print(f"⚠️ Could not fetch transcript for {video_id}: {e}")
        return None


# -------------------- STEP 3: CHUNK TEXT -------------------- #
def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split transcript into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------- STEP 4: EMBEDDINGS -------------------- #
def get_embedding(text: str) -> list:
    """Get embedding using local Sentence-Transformer."""
    emb_fn = get_embedding_function()
    return emb_fn([text])[0]


# -------------------- STEP 5: INDEX VIDEO -------------------- #
def index_video(video_url: str):
    """Indexes a video or channel. Returns status info."""
    client = get_chroma_client()
    
    print("🧹 Clearing existing database for a fresh start...")
    try:
        client.delete_collection("youtube_channel")
    except Exception:
        pass
        
    collection = get_collection()

    print("🎥 Fetching video ID(s)...")
    video_ids = get_channel_video_ids(video_url)
    print(f"Found {len(video_ids)} video(s)")

    indexed_videos = 0
    total_chunks = 0
    errors = []

    for video_id in video_ids:
        transcript = get_transcript(video_id)

        if not transcript:
            errors.append(f"Could not fetch transcript for {video_id} (likely blocked or no captions)")
            continue

        chunks = chunk_text(transcript)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{video_id}_chunk_{i}"],
                metadatas=[{
                    "video_id": video_id,
                    "url": f"https://youtube.com/watch?v={video_id}"
                }]
            )
            total_chunks += 1

        indexed_videos += 1
        print(f"✅ Indexed video {indexed_videos}/{len(video_ids)}: {video_id}")

    return {
        "videos_found": len(video_ids),
        "indexed_videos": indexed_videos,
        "total_chunks": total_chunks,
        "errors": errors
    }


# -------------------- STEP 6: QUERY -------------------- #
def query_video(question: str) -> dict:
    """Search across indexed video chunks and generate answer."""
    collection = get_collection()
    groq = get_groq_client()

    q_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=5
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return {
            "answer": "I couldn't find any relevant information in the indexed videos. Please make sure the video was indexed successfully.",
            "sources": []
        }

    context = "\n\n".join(results["documents"][0])
    sources = [m["url"] for m in results["metadatas"][0]]

    prompt = f"""
Answer the question using ONLY the YouTube transcript context below.
Also mention which video(s) the answer came from.

Context:
{context}

Question: {question}
"""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources))
    }


# -------------------- TEST -------------------- #
if __name__ == "__main__":
    # Corrected function names for testing
    status = index_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"Status: {status}")

    if status["indexed_videos"] > 0:
        result = query_video("What is this video about?")
        print(result["answer"])
        print("Sources:", result["sources"])
    else:
        print("Failed to index video.")