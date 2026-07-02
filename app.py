import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load Facebook NLLB Model
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# -----------------------------
# Translation Function
# -----------------------------
def translate(text):

    if text.strip() == "":
        return ""

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
            max_length=128
        )

    hindi = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return hindi


# -----------------------------
# Custom CSS
# -----------------------------
css = """

.gradio-container{
background:#ece5dd;
}

.header{
background:#075E54;
color:white;
padding:18px;
border-radius:10px;
font-size:24px;
font-weight:bold;
text-align:center;
margin-bottom:10px;
}

.footer{
text-align:center;
padding:10px;
font-size:15px;
color:gray;
}

"""

# -----------------------------
# UI
# -----------------------------

with gr.Blocks(css=css,title="AI-Powered English to Hindi Language Translator") as demo:

    gr.HTML("""
    <div class="header">
        🇬🇧 ➜ 🇮🇳 AI-Powered English to Hindi Language Translator
    </div>
    """)

    chatbot = gr.Chatbot(
        height=500,
        label="Translator Chat",
        bubble_full_width=False
    )

    msg = gr.Textbox(
        placeholder="Type English sentence here...",
        show_label=False
    )

    with gr.Row():

        send = gr.Button("📤 Translate",variant="primary")

        clear = gr.Button("🗑 Clear")

    examples = gr.Examples(
        examples=[
            ["Hello"],
            ["Good Morning"],
            ["I love India"],
            ["How are you?"],
            ["Where are you going?"],
            ["Artificial Intelligence is amazing."],
            ["Today is a beautiful day."]
        ],
        inputs=msg
    )

    def respond(message, history):

        answer = translate(message)

        history.append((message, answer))

        return "", history

    send.click(
        respond,
        [msg, chatbot],
        [msg, chatbot]
    )

    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot]
    )

    clear.click(
        lambda: [],
        outputs=chatbot
    )

    gr.HTML("""
    <div class="footer">
    Powered by Facebook NLLB Transformer | Hugging Face | Gradio
    </div>
    """)

demo.launch()
