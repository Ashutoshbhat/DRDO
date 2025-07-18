import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Phi-2 Chatbot", page_icon="🤖")

@st.cache_resource
def load_model():
    model_name = "microsoft/phi-2"  # ✅ Replace with your fine-tuned repo name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return tokenizer, model

st.title("🤖 Phi-2 Fine-tuned Chatbot")

with st.spinner("Loading Phi-2... Please wait, this may take time."):
    tokenizer, model = load_model()

def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    bot_reply = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))

for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")
