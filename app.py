import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="My Fine-tuned Chatbot", page_icon="🤖")

# -------------------------------
# ✅ Load Model & Tokenizer (cached to avoid reloading every time)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "Ashutosh1010/youmatterchatbot"  # Replace with your HF repo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return tokenizer, model

st.title("🤖 My Fine-tuned Chatbot")

with st.spinner("Loading model... Please wait, this may take a while."):
    tokenizer, model = load_model()

# -------------------------------
# ✅ Chat Function
# -------------------------------
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# -------------------------------
# ✅ Chat UI
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    bot_reply = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
