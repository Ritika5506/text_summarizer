import torch
import gradio as gr
from transformers import pipeline

# Download model automatically from Hugging Face
text_summary = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def summary(input):
    output = text_summary(input)
    return output[0]["summary_text"]

demo = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(
        label="Input text to summarize",
        lines=6
    ),
    outputs=gr.Textbox(
        label="Summarized text",
        lines=4
    ),
    title="Text Summarizer",
    description="This application summarizes the given text."
)

if __name__ == "__main__":
    demo.launch()