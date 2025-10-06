import gradio as gr

# Minimal centered UI — no flag button, no extra controls.
# Whatever you type returns a mock answer. Replace mock_answer() later.

def mock_answer(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "Please enter a question."
    return "This is a mock answer."

with gr.Blocks(
    title="Multi-hop QA — Minimal Demo",
    css="""
    #wrap {max-width: 880px; margin: 0 auto;}
    h1.title {text-align:center; font-size: 2.4rem; margin: 16px 0 10px;}
    #q textarea {min-height: 140px; font-size: 1.05rem;}
    .center {display:flex; justify-content:center;}
    """
) as demo:
    gr.Markdown("<h1 class='title'>Multi‑hop QA — Minimal Demo</h1>")
    with gr.Column(elem_id="wrap"):
        q = gr.Textbox(
            label="Ask a Wikipedia-style question",
            placeholder="e.g., Who discovered penicillin?",
            lines=3,
            elem_id="q",
        )
        with gr.Row(elem_classes=["center"]):
            submit = gr.Button("Submit", variant="primary")
        out = gr.Markdown(label="Answer")

        def on_submit(x: str):
            return mock_answer(x)

        submit.click(on_submit, q, out)
        q.submit(on_submit, q, out)  # press Enter to submit

if __name__ == "__main__":
    demo.launch(inbrowser = True)
