import streamlit as st
from rag_chain import get_chat_response
from pathlib import Path
import base64

# ---------- Config ----------
LOGO_LOCAL = Path(r"C:\Users\gerar\Desktop\rdl_data\assets\RDL_LOGO_2024 J.png")
ASSISTANT_LOGO = Path(r"C:\Users\gerar\Desktop\rdl_data\assets\RDL__.png")  
width_px = 100  # header logo size

# ---------- page config ----------
st.set_page_config(
    page_title="RDL Assistant",
    page_icon=str(LOGO_LOCAL),
    layout="centered"
)

# ---------- prepare inline image as data URI ----------
def img_to_data_uri(path: Path):
    img_bytes = path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

header_img = img_to_data_uri(LOGO_LOCAL)
assistant_img = img_to_data_uri(ASSISTANT_LOGO)

# ---------- render inline title with alignment ----------
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:14px;">
        <img src="{header_img}" style="width:{width_px}px; height:auto;"/>
        <span style="font-size:48px; font-weight:700;">Assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Ask me anything ! I'm here to help you!")

# ---------- chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = None
    if msg["role"] == "assistant":
        avatar = assistant_img  # custom assistant logo
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------- user input ----------
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_chat_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar=assistant_img):
        st.markdown(response)
