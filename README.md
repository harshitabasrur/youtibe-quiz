# YouTube Quiz Generator Extension

A simple Chrome extension + Flask backend that generates interactive multiple-choice quizzes from YouTube videos.

## Features
- Load transcript of any YouTube video
- Generate context-based multiple-choice questions
- Check answers with explanations
- Supports translated transcripts

## Tech Stack
- Python, Flask
- LangChain & Google Generative AI Embeddings
- YouTube Transcript API
- Deep Translator (for translation)
- FAISS (vector storage)

## How it works
1. User enters a YouTube video ID in the extension.
2. Transcript is fetched and translated (if needed).
3. Transcript is split into chunks, embedded, and stored in a vector store.
4. When generating a question:
   - Relevant chunks are retrieved using semantic search.
   - LLM generates a question with options.
   - Correct answer and explanation are stored server-side.
5. User can check their answer via the extension.

