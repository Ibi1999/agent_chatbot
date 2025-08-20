# app.py — Agent-only (with sidebar info)
import time
import streamlit as st

from agent import agent_router  # your router (uses GitHub Models under the hood)

# === CONFIG ===
st.set_page_config(page_title="LLM Chatbot — Agent", page_icon="🤖")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Agent Settings")
    if "show_tools_used" not in st.session_state:
        st.session_state.show_tools_used = True
    st.session_state.show_tools_used = st.checkbox("🛠️ Show Tool Used", value=st.session_state.show_tools_used)

    # Info dropdown
    with st.expander("ℹ️ About Agent Tools", expanded=False):
        st.markdown("""
        The agent automatically decides how to answer your query using three tools:
        
        - **RAG (Retrieval-Augmented Generation)**:  
          Uses documents to answer questions about Ibrahim and his work or hobbies.
        
        - **Calculator**:  
          Handles straightforward mathematical queries such as arithmetic, multiplication, division, and percentages.
        
        - **General**:  
          A general-purpose LLM response for all other queries not suited for RAG or Calculator.
        """)

# === TITLE ===
st.title("🧠 Chatbot — Agent")

# === STATE INIT ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "show_tools_used" not in st.session_state:
    st.session_state.show_tools_used = True

# === INPUT UI ===
user_input = st.text_input(
    "Your Message",
    placeholder="Ask anything… (the agent will pick RAG / Calculator / General)",
    key="user_input",
    label_visibility="collapsed"
)
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    send_clicked = st.button("➡️ Send", use_container_width=True)
with btn_col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

# === CLEAR CHAT ===
if clear_clicked:
    st.session_state.messages = []
    st.session_state.pending_user_input = None
    st.rerun()

# === SUBMIT ===
if send_clicked and user_input:
    st.session_state.pending_user_input = user_input
    st.rerun()

# === HANDLE MESSAGE ===
if st.session_state.pending_user_input:
    prompt = st.session_state.pending_user_input
    st.session_state.pending_user_input = None
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        start_time = time.time()
        try:
            # Build chat history (user + assistant turns only)
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] in ("user", "assistant"):
                    chat_history.append({"role": msg["role"], "content": msg["content"]})

            # Agent route
            result = agent_router(prompt, chat_history=chat_history)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

            if result.get("sources"):
                st.session_state.messages.append({"role": "sources", "content": result["sources"]})
            st.session_state.messages.append({"role": "tools_used", "content": result["type"]})

            thinking_duration = time.time() - start_time
            st.session_state.messages.append({
                "role": "timer",
                "content": f"🕒 Answered in {thinking_duration:.2f} seconds"
            })

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})

    st.rerun()

# === DISPLAY CHAT ===
pairs = []
msgs = st.session_state.messages
i = 0
while i < len(msgs):
    if msgs[i]["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
        user_msg = msgs[i]
        assistant_msg = msgs[i + 1]
        i += 2

        # collect meta messages in ANY order
        source_msg = tool_msg = timer_msg = None
        while i < len(msgs) and msgs[i]["role"] in ("sources", "tools_used", "timer"):
            if msgs[i]["role"] == "sources" and source_msg is None:
                source_msg = msgs[i]
            elif msgs[i]["role"] == "tools_used" and tool_msg is None:
                tool_msg = msgs[i]
            elif msgs[i]["role"] == "timer" and timer_msg is None:
                timer_msg = msgs[i]
            i += 1

        pairs.append((user_msg, assistant_msg, source_msg, tool_msg, timer_msg))
    else:
        i += 1

# === SHOW CHAT ===
for idx, (user_msg, assistant_msg, source_msg, tool_msg, timer_msg) in enumerate(reversed(pairs)):
    with st.chat_message("user"):
        st.markdown(user_msg["content"])
    with st.chat_message("assistant"):
        st.markdown(assistant_msg["content"])

        if st.session_state.get("show_tools_used") and tool_msg:
            st.markdown(
                "<span style='color:green; font-weight:bold;'>Tool used: </span>"
                f"<span style='color:orange; font-weight:bold;'>{tool_msg['content']}</span>",
                unsafe_allow_html=True
            )
        if tool_msg and tool_msg["content"] == "RAG" and source_msg and source_msg["content"]:
            sources_str = ", ".join(f"<span style='color:orange;'>{src}</span>" for src in source_msg["content"])
            st.markdown(
                f"<span style='color:green; font-weight:bold;'>Sources: </span>{sources_str}",
                unsafe_allow_html=True
            )

        if timer_msg:
            st.markdown(f"<span style='color:red;'>{timer_msg['content']}</span>", unsafe_allow_html=True)

    if idx < len(pairs) - 1:
        st.markdown("---")
