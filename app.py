import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------------------
# Load Facebook NLLB Model
# ----------------------------------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading AI Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model Loaded Successfully!")

# ----------------------------------------------------
# Translation Function
# ----------------------------------------------------

def translate(text):

    if text.strip() == "":
        return ""

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
            max_length=128
        )

    hindi = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return hindi


# ----------------------------------------------------
# Clear Function
# ----------------------------------------------------

def clear():
    return "", ""


# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

css = """
body{
    background:#ECE5DD;
}

.gradio-container{
    max-width:900px !important;
    margin:auto;
}

h1{
    text-align:center;
    color:#075E54;
    font-size:34px;
}

textarea{
    font-size:18px !important;
}

footer{
    visibility:hidden;
}

"""


# ----------------------------------------------------
# Interface
# ----------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
# 🇬🇧 ➜ 🇮🇳 AI-Powered English to Hindi Language Translator

### Translate English sentences into Hindi using Transformer
"""
    )

    with gr.Row():

        english = gr.Textbox(
            label="English Text",
            lines=8,
            placeholder="Enter English sentence..."
        )

        hindi = gr.Textbox(
            label="Hindi Translation",
            lines=8
        )

    with gr.Row():

        translate_btn = gr.Button(
            "Translate",
            variant="primary"
        )

        clear_btn = gr.Button("🗑 Clear")

    gr.Examples(
        examples=[
            ["Hello"],
            ["Good Morning"],
            ["How are you?"],
            ["Where are you going?"],
            ["I love India."],
            ["Artificial Intelligence is changing the world."],
            ["Today is a beautiful day."],
            ["Thank you very much."]
        ],
        inputs=english
    )

    translate_btn.click(
        translate,
        inputs=english,
        outputs=hindi
    )

    english.submit(
        translate,
        inputs=english,
        outputs=hindi
    )

    clear_btn.click(
        clear,
        outputs=[english, hindi]
    )

    gr.Markdown(
        """
---
###  Powered by Sugumarai
"""
    )

demo.launch()
