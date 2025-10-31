# frontend/app.py
import streamlit as st
import requests

BACKEND = "http://127.0.0.1:8000"

st.title("🧠 AI Study Assistant")

topic = st.text_input("Enter a topic (e.g., Photosynthesis):")

if st.button("Generate Summary & Quiz"):
    if not topic.strip():
        st.warning("Type a topic first")
    else:
        with st.spinner("Generating..."):
            res = requests.post(f"{BACKEND}/api/study", json={"topic": topic})
            data = res.json()
            summary = data.get("summary", "")
            questions = data.get("quiz_questions") or []
            st.subheader("📘 Summary")
            st.write(summary)
            st.subheader("❓ Quiz Questions")
            # present questions and capture answers
            answers = []
            for i, q in enumerate(questions):
                st.write(f"{i+1}. {q}")
                ans = st.text_input(f"Your answer to question {i+1}", key=f"a{i}")
                answers.append(ans)
            if st.button("Submit Answers"):
                with st.spinner("Grading..."):
                    eval_payload = {
                        "topic": topic,
                        "summary": summary,
                        "quiz_questions": questions,
                        "user_answers": answers
                    }
                    eval_res = requests.post(f"{BACKEND}/api/evaluate", json=eval_payload)
                    feedback = eval_res.json().get("feedback", "")
                    st.subheader("📝 Feedback")
                    st.write(feedback)
