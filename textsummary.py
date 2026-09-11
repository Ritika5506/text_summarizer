import gradio as gr
from transformers import pipeline

# Download model automatically from Hugging Face
model_path = "sshleifer/tiny-t5"

text_summary = pipeline(
    "text2text-generation",
    model=model_path
)

def summary(input):
    output = text_summary("summarize: " + input)
    return output[0]['generated_text']

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