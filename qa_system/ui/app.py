# To run this app, from the project root (venv active; Ollama running via `ollama serve`):
# python -m qa_system.ui.app

from functools import lru_cache
from typing import Tuple
import gradio as gr
from qa_system.pipeline.qa_pipeline import build_pipeline

PROJECT_ABSTRACT = """
**FIRE-QA (Full Interaction Retrieval and Enhanced Question Answering)**

Question-answering (QA) remains one of the core challenges in Natural Language Processing (NLP). While most QA & Information Retrieval (IR) systems perform well on single-hop questions, they often struggle with multi-hop questions that require multiple reasoning steps across different documents.

**FIRE-QA** searches many documents, gathers the pieces of evidence you need, ranks the most relevant parts, and uses an AI model to produce an answer with a clear reasoning trail—all in a simple web app.
"""

@lru_cache(maxsize=2)
def get_pipeline(use_rewriter: bool):
    return build_pipeline(use_rewriter=use_rewriter)

def run_pipeline(question: str, use_rewriter: bool, show_steps: bool, show_sources: bool) -> Tuple[str, str, str]:
    q = (question or "").strip()
    if not q:
        return "Please enter a question.", "", ""
    try:
        pipe = get_pipeline(use_rewriter)
        result = pipe.answer_question(q)
    except Exception as e:
        return f"Error: {e}", "", ""

    answer = (result.get("answer") or "").strip() or "INSUFFICIENT EVIDENCE"

    steps_md = ""
    if show_steps:
        steps = result.get("reasoning_steps", []) or []
        flat = []
        for s in steps:
            flat.extend(s if isinstance(s, (list, tuple)) else [s])
        if flat:
            steps_md = "\n".join(f"{i+1}. {x}" for i, x in enumerate(flat))

    sources_md = ""
    if show_sources:
        parts = []
        for i, d in enumerate(result.get("contexts", []) or [], 1):
            txt = d.get("text") or d.get("chunk") or d.get("content") or ""
            score = d.get("reranker_score", d.get("retriever_score", ""))
            if len(txt) > 1500:
                txt = txt[:1500] + " …"
            parts.append(f"**Doc {i}** (score: {score})\n\n{txt}")
        sources_md = "\n\n---\n\n".join(parts)

    return answer, steps_md, sources_md


with gr.Blocks(
    title="FIRE-QA — Multi-hop Question Answering",
    css="""
    #wrap {max-width: 980px; margin: 0 auto;}
    h1.title {text-align:center; font-size: 2.2rem; margin: 16px 0 12px;}
    #q textarea {min-height: 140px; font-size: 1.05rem;}
    .meta {font-size: 0.92rem; opacity: 0.9;}
    """
) as demo:
    gr.Markdown("<h1 class='title'>FIRE-QA — Multi-hop Question Answering</h1>")
    with gr.Row():
        with gr.Accordion("About FIRE-QA", open=False):
            gr.Markdown(PROJECT_ABSTRACT)

    with gr.Column(elem_id="wrap"):
        gr.Markdown(
            "Ask multi-hop questions that require combining evidence across documents. "
            "Toggle options to view reasoning and sources.", elem_id=None, visible=True
        )

        q = gr.Textbox(
            label="Ask a question",
            placeholder="e.g., Are the Laleli Mosque and Esma Sultan Mansion in the same neighborhood?",
            lines=4,
            elem_id="q",
        )
        with gr.Row():
            use_rewriter = gr.Checkbox(value=True, label="Use Query Rewriter")
            show_steps = gr.Checkbox(value=True, label="Show Reasoning Steps")
            show_sources = gr.Checkbox(value=True, label="Show Top Contexts")

        btn = gr.Button("Submit", variant="primary")

        out_answer = gr.Markdown(label="Answer")
        out_steps = gr.Markdown(label="Reasoning")
        out_sources = gr.Markdown(label="Contexts")

        btn.click(run_pipeline, [q, use_rewriter, show_steps, show_sources], [out_answer, out_steps, out_sources])
        q.submit(run_pipeline, [q, use_rewriter, show_steps, show_sources], [out_answer, out_steps, out_sources])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
