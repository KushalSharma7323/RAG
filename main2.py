import streamlit as st
import subprocess
import threading
import queue
import time

st.title("🤖 RAG Chat UI (Stable Version)")

# ---------------- START BACKEND ONLY ONCE ----------------
def start_process():
    return subprocess.Popen(
        ["python", "RAG/rag.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

if "process" not in st.session_state:
    process = start_process()
    st.session_state.process = process

    st.session_state.queue = queue.Queue()

    # ---------------- STDOUT READER THREAD ----------------
    def reader(proc, q):
        try:
            for line in proc.stdout:
                q.put(line)
        except Exception:
            pass

    threading.Thread(
        target=reader,
        args=(process, st.session_state.queue),
        daemon=True
    ).start()

    # ---------------- STDERR READER THREAD (IMPORTANT FIX) ----------------
    def err_reader(proc):
        try:
            for line in proc.stderr:
                print("RAG ERROR:", line)
        except Exception:
            pass

    threading.Thread(
        target=err_reader,
        args=(process,),
        daemon=True
    ).start()

# ---------------- CHAT HISTORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- RENDER CHAT ----------------
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:
    process = st.session_state.process

    # ---- check if process is alive ----
    if process.poll() is not None:
        st.error("Backend process stopped. Restart Streamlit.")
        st.stop()

    # ---------------- SEND INPUT SAFELY ----------------
    try:
        process.stdin.write(user_input + "\n")
        process.stdin.flush()
    except Exception as e:
        st.error(f"Backend write failed: {e}")
        st.stop()

    # ---------------- READ OUTPUT (NON-BLOCKING SAFE LOOP) ----------------
    output_text = ""
    start_time = time.time()

    while time.time() - start_time < 5:  # timeout prevents freeze
        try:
            line = st.session_state.queue.get_nowait()
            output_text += line

            # stop condition based on your RAG output format
            if "Chatbot:" in line:
                output_text = line.split("Chatbot:")[-1].strip()
                break

        except queue.Empty:
            time.sleep(0.05)

    # fallback if nothing received
    if not output_text:
        output_text = "(No response received from backend)"

    # ---------------- SAVE CHAT ----------------
    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("assistant", output_text))

    st.rerun()