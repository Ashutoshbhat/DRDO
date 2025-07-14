# app.py
# app.py
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ──────────────────────────────────────────────────────────────
# Public HF repo ID (fallback to your repo if MODEL_ID isn’t set)
MODEL_ID = os.environ.get("MODEL_ID", "Ashutosh1010/phi2")

# Port (Render or Spaces will inject)
PORT = int(os.environ.get("PORT", 7860))
# ────────────────────────────────────────────────────────────────────────────────

# ✅ Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.eval()

# ✅ Chat function
def chat(message, history):
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{bot_msg}\n"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()

# ✅ Launch
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="🧠 YouMatter - Mental Health Chatbot",
        description="Talk to your fine-tuned Phi-2 chatbot 💬",
        theme="soft",
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        inbrowser=False,
    )
