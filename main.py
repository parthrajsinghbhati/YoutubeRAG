import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# Load environment variables
load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
transcript_api = YouTubeTranscriptApi()

# Initialize embeddings (local & free)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma = chromadb.Client()
collection = chroma.get_or_create_collection(
    name="youtube_channel",
    embedding_function=embedding_function
)


# -------------------- STEP 1: GET VIDEO IDS -------------------- #
def get_channel_video_ids(channel_url: str) -> list[str]:
    """Fetch all video IDs from a YouTube channel."""
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'playlistend': 100  # increase if needed
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
        transcript = transcript_api.fetch(video_id)
        return " ".join([chunk.text for chunk in transcript])
    except Exception as e:
        print(f"⚠️ Could not fetch transcript for {video_id}: {e}")
        return None  # skip videos without transcript


# -------------------- STEP 3: CHUNK TEXT -------------------- #
def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split transcript into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - 50):  # 50-word overlap
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------- STEP 4: EMBEDDINGS -------------------- #
def get_embedding(text: str) -> list:
    """Get embedding using local Sentence-Transformer."""
    return embedding_function([text])[0]


# -------------------- STEP 5: INDEX VIDEO -------------------- #
def index_video(video_url: str):
    global collection
    
    print("🧹 Clearing existing database for a fresh start...")
    try:
        chroma.delete_collection("youtube_channel")
    except Exception:
        pass
        
    collection = chroma.get_or_create_collection(
        name="youtube_channel",
        embedding_function=embedding_function
    )

    print("🎥 Fetching video ID...")
    video_ids = get_channel_video_ids(video_url)
    print(f"Found {len(video_ids)} video(s)")

    indexed = 0

    for video_id in video_ids:
        transcript = get_transcript(video_id)

        if not transcript:
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

        indexed += 1
        print(f"✅ Indexed video {indexed}/{len(video_ids)}: {video_id}")

    print(f"\n✅ Indexed {indexed} videos successfully")


# -------------------- STEP 6: QUERY -------------------- #
def query_video(question: str) -> dict:
    """Search across indexed video chunks and generate answer."""

    q_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=5
    )

    context = "\n\n".join(results["documents"][0])
    sources = [m["url"] for m in results["metadatas"][0]]

    prompt = f"""
Answer the question using ONLY the YouTube transcript context below.
Also mention which video(s) the answer came from.

Context:
{context}

Question: {question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources))
    }


# -------------------- TEST -------------------- #
if __name__ == "__main__":
    index_channel("https://www.youtube.com/@AndrejKarpathy/videos")

    result = query_channel("What does this creator say about transformers?")
    print(result["answer"])
    print("Sources:", result["sources"])