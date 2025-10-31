# backend/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from openai import OpenAI
from langgraph.graph import StateGraph

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# Initialize OpenAI / OpenRouter client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
MODEL = os.getenv("OPENAI_MODEL", "z-ai/glm-4.5-air:free")


# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI(title="AI Study Assistant (FastAPI + LangGraph)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (safe for local dev / hackathon)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------
class StudyRequest(BaseModel):
    topic: str

class EvalRequest(BaseModel):
    topic: str
    summary: Optional[str] = None
    quiz_questions: Optional[List[str]] = None
    user_answers: Optional[List[str]] = None

# -------------------------------------------------------------------
# Helper to call the model
# -------------------------------------------------------------------
def llm_chat(prompt: str) -> str:
    print("🔹 Sending prompt to model:", prompt[:80])
    try:
        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        text = resp.choices[0].message.content.strip()
        print("🔹 Got response:", text[:80])
        return text
    except Exception as e:
        print("❌ LLM Error:", e)
        return ""

# -------------------------------------------------------------------
# Node functions for LangGraph
# -------------------------------------------------------------------
def summarizer_node(state, config):
    topic = state.get("topic", "")
    prompt = (
        f"Explain the topic '{topic}' in 3–4 clear sentences for a student. "
        "Make it easy to understand and include one simple example."
    )
    summary = llm_chat(prompt)
    return {"summary": summary}

def quiz_generator_node(state, config):
    summary = state.get("summary", "")
    prompt = (
        f"Based on this summary:\n{summary}\n\n"
        "Create 3 simple quiz questions to test understanding. "
        "Return them as numbered lines 1., 2., 3."
    )
    quiz_text = llm_chat(prompt)
    questions = []
    for line in quiz_text.splitlines():
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            q = line.split('.', 1)[-1].strip()
            if q:
                questions.append(q)
    if not questions:
        questions = [quiz_text]
    return {"quiz_text": quiz_text, "quiz_questions": questions}

def evaluator_node(state, config):
    questions = state.get("quiz_questions", [])
    answers = state.get("user_answers", [])
    summary = state.get("summary", "")
    prompt = (
        "You are a tutor grading short answers.\n"
        "Given the topic summary, the quiz questions, and the student's answers, "
        "grade each as Correct / Partially Correct / Incorrect and give one-sentence feedback.\n\n"
        f"Summary:\n{summary}\n\nQuestions & Answers:\n"
    )
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        prompt += f"{i}. Q: {q}\n   A: {a}\n"
    feedback = llm_chat(prompt)
    return {"feedback": feedback}

# -------------------------------------------------------------------
# Build LangGraph (compatible with latest version)
# -------------------------------------------------------------------
from langgraph.graph import StateGraph

builder = StateGraph(dict)

# Add all nodes
builder.add_node("summarizer", summarizer_node)
builder.add_node("quiz_generator", quiz_generator_node)
builder.add_node("evaluator", evaluator_node)

# Connect the flow between nodes
builder.add_edge("summarizer", "quiz_generator")
builder.add_edge("quiz_generator", "evaluator")

# ✅ Explicitly mark the entry point using the special START edge
builder.add_edge("__start__", "summarizer")

# ✅ Mark the finish node
builder.set_finish_point("evaluator")

# Compile final graph
graph = builder.compile()

#---------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------
@app.post("/api/study")
def study(req: StudyRequest):
    """Run full graph to generate summary + quiz."""
    initial = {"topic": req.topic}
    result = graph.invoke(initial)
    return {
        "topic": req.topic,
        "summary": result.get("summary"),
        "quiz_questions": result.get("quiz_questions"),
        "quiz_text": result.get("quiz_text"),
        "feedback": result.get("feedback", ""),
    }

@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    """Grade user's quiz answers only."""
    state = {
        "topic": req.topic,
        "summary": req.summary or "",
        "quiz_questions": req.quiz_questions or [],
        "user_answers": req.user_answers or [],
    }
    feedback = evaluator_node(state, None).get("feedback", "")
    return {"feedback": feedback}

# -------------------------------------------------------------------
# Root route (optional)
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running successfully!"}
