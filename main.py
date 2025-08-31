# Modified backend for extension - separates question generation from answer checking

import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()  # Load keys from .env

google_key = os.getenv('GOOGLE_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)  # Allow extension to call your API


class QuizBackend:
    def __init__(self):

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )

        self.llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            model="openai/gpt-4o",
        )

        self.vector_store = None
        self.retriever = None
        self.quiz_answers = {}  # Store correct answers server-side

    def load_and_process_video(self, video_id):
        """Load transcript and create vector store"""
        try:
            # Your existing transcript loading code
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)

            try:
                transcript_obj = transcript_list.find_transcript(['en'])
            except:
                transcript_obj = transcript_list.find_generated_transcript(['hi'])

            fetched = transcript_obj.fetch()
            transcript = " ".join(piece.text for piece in fetched)

            if transcript_obj.language_code != 'en':
                transcript = GoogleTranslator(source='hi', target='en').translate(transcript)

            # Create vector store
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            texts = [doc.page_content for doc in chunks]

            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15}
            )

            return True

        except Exception as e:
            print(f"Error processing video: {e}")
            return False


# SEPARATE PROMPTS FOR QUESTIONS AND ANSWERS

# 1. Question Generation Prompt (for extension display)
question_prompt = PromptTemplate(
    template="""
Generate 1 multiple choice question based on the video transcript context.

IMPORTANT: 
- Only return the question and options
- DO NOT include the correct answer or explanation
- Make it a good test of understanding

Format EXACTLY like this:
What is the main concept explained in this section?

A) First option
B) Second option  
C) Third option
D) Fourth option

Context from video:
{context}

Focus on: {topic}
""",
    input_variables=["context", "topic"]
)

# 2. Answer Generation Prompt (for server-side storage)
answer_prompt = PromptTemplate(
    template="""
For this multiple choice question about the video content, determine:
1. The correct answer (A, B, C, or D)
2. A brief explanation

Question: {question}
Options: {options}

Video context: {context}

Respond in this format:
ANSWER: [A/B/C/D]
EXPLANATION: [Brief explanation]
""",
    input_variables=["question", "options", "context"]
)

quiz_backend = QuizBackend()


# API Endpoints for Extension

@app.route('/load_video', methods=['POST'])
def load_video():
    """Load and process YouTube video for quiz generation"""
    data = request.json
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400

    success = quiz_backend.load_and_process_video(video_id)

    if success:
        return jsonify({"status": "success", "message": "Video processed successfully"})
    else:
        return jsonify({"error": "Failed to process video"}), 500


@app.route('/generate_question', methods=['POST'])
def generate_question():
    """Generate a single quiz question (without answer)"""
    data = request.json
    topic = data.get('topic', 'main concepts')

    if not quiz_backend.retriever:
        return jsonify({"error": "No video loaded"}), 400

    try:
        # Get relevant context
        docs = quiz_backend.retriever.invoke(topic)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Generate question (without answer)
        question_response = quiz_backend.llm.invoke(
            question_prompt.format(context=context, topic=topic)
        )

        question_text = question_response.content.strip()

        # Parse question and options
        lines = question_text.split('\n')
        question_line = lines[0]
        options = {}

        for line in lines[1:]:
            line = line.strip()
            if line and line[0] in ['A', 'B', 'C', 'D']:
                key = line[0]
                value = line[3:].strip()  # Remove "A) " part
                options[key] = value

        # Generate answer separately (store server-side)
        answer_response = quiz_backend.llm.invoke(
            answer_prompt.format(
                question=question_line,
                options=str(options),
                context=context
            )
        )

        # Parse answer
        answer_text = answer_response.content.strip()
        correct_answer = None
        explanation = ""

        for line in answer_text.split('\n'):
            if line.startswith('ANSWER:'):
                correct_answer = line.replace('ANSWER:', '').strip()
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()

        # Store answer server-side with unique ID
        question_id = f"q_{len(quiz_backend.quiz_answers)}"
        quiz_backend.quiz_answers[question_id] = {
            "correct_answer": correct_answer,
            "explanation": explanation
        }

        # Return only question to extension
        return jsonify({
            "question_id": question_id,
            "question": question_line,
            "options": options
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate question: {str(e)}"}), 500


@app.route('/check_answer', methods=['POST'])
def check_answer():
    """Check if user's answer is correct"""
    data = request.json
    question_id = data.get('question_id')
    user_answer = data.get('user_answer')

    if question_id not in quiz_backend.quiz_answers:
        return jsonify({"error": "Question not found"}), 404

    stored_answer = quiz_backend.quiz_answers[question_id]
    is_correct = user_answer.upper() == stored_answer["correct_answer"].upper()

    return jsonify({
        "correct": is_correct,
        "correct_answer": stored_answer["correct_answer"],
        "explanation": stored_answer["explanation"],
        "user_answer": user_answer
    })


@app.route('/generate_full_quiz', methods=['POST'])
def generate_full_quiz():
    """Generate multiple questions at once"""
    data = request.json
    num_questions = data.get('num_questions', 5)
    topics = data.get('topics', ['main concepts', 'key points', 'important details'])

    if not quiz_backend.retriever:
        return jsonify({"error": "No video loaded"}), 400

    questions = []

    for i in range(num_questions):
        topic = topics[i % len(topics)]  # Cycle through topics

        try:
            # Generate question
            docs = quiz_backend.retriever.invoke(topic)
            context = "\n\n".join(doc.page_content for doc in docs)

            question_response = quiz_backend.llm.invoke(
                question_prompt.format(context=context, topic=topic)
            )

            question_text = question_response.content.strip()
            lines = question_text.split('\n')
            question_line = lines[0]
            options = {}

            for line in lines[1:]:
                line = line.strip()
                if line and line[0] in ['A', 'B', 'C', 'D']:
                    key = line[0]
                    value = line[3:].strip()
                    options[key] = value

            # Generate and store answer
            answer_response = quiz_backend.llm.invoke(
                answer_prompt.format(
                    question=question_line,
                    options=str(options),
                    context=context
                )
            )

            answer_text = answer_response.content.strip()
            correct_answer = None
            explanation = ""

            for line in answer_text.split('\n'):
                if line.startswith('ANSWER:'):
                    correct_answer = line.replace('ANSWER:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()

            question_id = f"q_{len(quiz_backend.quiz_answers)}"
            quiz_backend.quiz_answers[question_id] = {
                "correct_answer": correct_answer,
                "explanation": explanation
            }

            questions.append({
                "question_id": question_id,
                "question": question_line,
                "options": options
            })

        except Exception as e:
            print(f"Error generating question {i + 1}: {e}")

    return jsonify({"questions": questions})


if __name__ == "__main__":
    # For testing locally
    print("🚀 Starting Quiz Backend Server...")
    print("Test it by running the server and making API calls")

    # Test with your video
    video_id = "gl5sd_AXdK4"
    if quiz_backend.load_and_process_video(video_id):
        print("✅ Video loaded successfully!")
        print("🌐 Server ready for extension requests...")
        app.run(debug=True, port=5000)
    else:
        print("❌ Failed to load video")