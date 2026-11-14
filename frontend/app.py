import streamlit as st
import httpx

st.set_page_config(page_title="Website Controller", layout="centered")
st.title("🌐 Website Whitelist Controller")
st.caption("Interact with your Spring Boot MCP architecture via natural language.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g., Show allowed websites or Add google.com to my list"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            with httpx.Client() as client:
                res = client.post(
                    "http://localhost:8000/chat",
                    json={"messages": st.session_state.messages},
                    timeout=60.0,
                )
                res.raise_for_status()
                answer = res.json().get("response", "No response parsing available.")
        except Exception as e:
            answer = f"Error communicating with local proxy: {str(e)}"
            
        response_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})