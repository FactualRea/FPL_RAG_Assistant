import gradio as gr
from answer import answer_question


def format_context(documents):
    sections = []
    for document in documents:
        source = document.metadata.get("source", "FPL API")
        snapshot = document.metadata.get("snapshot", "unknown")
        sections.append(f"Source: {source}\nSnapshot: {snapshot}\n\n{document.page_content}")
    return "\n\n".join(sections)


def chat(message, history):
    history = list(history or [])
    if not message or not message.strip():
        return "", history, ""

    try:
        response, documents = answer_question(message, history)
    except RuntimeError as error:
        response, documents = str(error), []

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", updated_history, format_context(documents)


def main():
    with gr.Blocks(title="FPL Data Assistant") as ui:
        gr.Markdown("# Fantasy Premier League Assistant")
        with gr.Row():
            with gr.Column(scale=3):
                conversation = gr.Chatbot(label="FPL questions", type="messages", height=520)
                prompt = gr.Textbox(
                    placeholder="Ask about players, teams, fixtures, or gameweeks...",
                    show_label=False,
                )
            with gr.Column(scale=2):
                sources = gr.Textbox(label="Retrieved FPL data", lines=24, interactive=False)
        prompt.submit(chat, inputs=[prompt, conversation], outputs=[prompt, conversation, sources])
    ui.launch(inbrowser=True, server_name="127.0.0.1")


if __name__ == "__main__":
    main()
