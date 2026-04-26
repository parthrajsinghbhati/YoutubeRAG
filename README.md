# YouTube Video RAG (Retrieval-Augmented Generation)

A powerful AI-driven application that allows you to "talk" to any YouTube video. Using local embeddings and the Groq LLM, this app indexes video transcripts and provides accurate, context-aware answers to your questions.

## Features

- **Instant Indexing**: Paste any YouTube video URL and index it in seconds.
- **Privacy & Performance**: Uses local `Sentence-Transformers` for embeddings—no data sent to embedding APIs and completely free to run.
- **Lightning Fast AI**: Powered by **Groq** (`llama-3.3-70b-versatile`) for near-instant responses.
- **Context-Aware**: Only answers based on the video transcript, reducing hallucinations.
- **Source Citation**: Automatically cites the specific video source for every answer.

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd YoutubeRAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_key_here
```

## Usage

Run the application locally:
```bash
streamlit run app.py
```

1. Enter a **YouTube Video URL**.
2. Click **"Index Video"**.
3. Ask questions about the video content in the text box.
