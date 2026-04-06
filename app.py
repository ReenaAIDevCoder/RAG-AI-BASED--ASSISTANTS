import streamlit as st
import speech_recognition as sr
import sys, os
from PIL import Image

# backend import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI Assistant", layout="wide")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "new_chat_started" not in st.session_state:
    st.session_state.new_chat_started = False

if "show_options" not in st.session_state:
    st.session_state.show_options = False

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("💬 Chat History")

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.new_chat_started = True

    # 👉 ONLY show history if new chat started
    if st.session_state.new_chat_started:
        for i, chat in enumerate(st.session_state.history):
            if len(chat) > 0:
                title = chat[0]["content"][:25] + "..."
            else:
                title = f"Chat {i+1}"

            if st.button(title, key=f"chat_{i}"):
                st.session_state.messages = chat

# ---------------- TITLE ----------------
st.title(" AI Chat Assistant (RAG)")

rag = RAGPipeline()

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- VOICE FUNCTION ----------------
def get_voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Speak now...")
            audio = recognizer.listen(source, timeout=5)

        return recognizer.recognize_google(audio)

    except Exception as e:
        return f"Voice error: {str(e)}"

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns([6,1,1])

with col1:
    user_input = st.chat_input(" Type your question...")

with col2:
    if st.button("➕"):
        st.session_state.show_options = not st.session_state.show_options

with col3:
    voice_btn = st.button("🎤")

# ---------------- OPTIONS (ON + CLICK) ----------------
uploaded_file = None

if st.session_state.show_options:
    uploaded_file = st.file_uploader(
        "📂 Upload (Image / PDF / Video / Text)",
        type=["png", "jpg", "jpeg", "pdf", "mp4", "txt"]
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image)

        elif uploaded_file.type.startswith("video"):
            st.video(uploaded_file)

        elif uploaded_file.type == "application/pdf":
            st.write("📄 PDF uploaded")

        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", content)

# ---------------- VOICE ----------------
if voice_btn:
    user_input = get_voice_input()

# ---------------- CHAT FLOW ----------------
if user_input:

    # mark chat started
    st.session_state.new_chat_started = True

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = rag.run(user_input)
    except Exception as e:
        response = f"Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 👉 SAVE ONLY AFTER FIRST MESSAGE
    if st.session_state.messages not in st.session_state.history:
        st.session_state.history.append(st.session_state.messages.copy())