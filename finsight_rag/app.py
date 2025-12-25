import gradio as gr
import finsight_rag.agent.agent as agent

app = agent.app

def call_agent(message: str, history):
    output = app.invoke({"query": message})

    answer = output.get("answer", "")
    route_mode = output.get("route_mode")

    # Build debug markdown
    debug_md = []
    if route_mode:
        debug_md.append(f"**route_mode:** `{route_mode}`")

    if route_mode == "multihop_rag":
        hop = output.get("hop")
        notes = output.get("notes", [])
        subquestions = output.get("subquestions", [])

        if hop is not None:
            debug_md.append(f"**hop:** {hop}")

        if subquestions:
            sq = "\n".join([f"- {s}" for s in subquestions])
            debug_md.append("**subquestions:**\n" + sq)

        if notes:
            nt = "\n".join([f"- {s}" for s in notes])
            debug_md.append("**notes:**\n" + nt)

    debug_text = "\n\n".join(debug_md) if debug_md else "—"

    # Update chat
    history = history + [[message, answer]]
    return "", history, debug_text

with gr.Blocks(title="Finsight RAG Chat") as demo:
    gr.Markdown("# Finsight RAG Chat")

    with gr.Row():
        chatbot = gr.Chatbot(height=520)

        with gr.Column(scale=1):
            gr.Markdown("### Debug")
            with gr.Accordion("Debug info", open=False):
                debug_box = gr.Markdown("—")

    msg = gr.Textbox(
        label="Message",
        placeholder="Ask a question…",
        autofocus=True,
    )
    send = gr.Button("Send", variant="primary")
    clear = gr.Button("Clear")

    send.click(
        call_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, debug_box],
    )
    msg.submit(
        call_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, debug_box],
    )

    clear.click(lambda: ([], "—"), None, [chatbot, debug_box])

if __name__ == "__main__":
    demo.launch(share=False)
