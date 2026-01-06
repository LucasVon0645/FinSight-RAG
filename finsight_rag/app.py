import os
import shutil
import gradio as gr

import finsight_rag.agent.agent as agent
from finsight_rag.utils import get_local_pdfs_dir, list_local_pdfs, get_pdf_path

app = agent.app
vector_store_wrapper = agent.vector_store_wrapper

PDF_DIR = get_local_pdfs_dir()

def call_agent(message: str, history):
    """
    Call the RAG agent with the user's message and chat history.
    """
    history = history or []

    output = app.invoke({"query": message})
    answer = output.get("answer", "")
    route_mode = output.get("route_mode")

    # Build debug markdown (unchanged)
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
            debug_md.append("**subquestions:**\n" + "\n".join([f"- {s}" for s in subquestions]))
        if notes:
            debug_md.append("**notes:**\n" + "\n".join([f"{s}" for s in notes]))
    debug_text = "\n\n".join(debug_md) if debug_md else "—"

    # NEW: messages format
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history, debug_text

def refresh_pdf_list(dropdown_value=None):
    """
    Refresh the list of local PDFs for the dropdown.
    """
    local_pdfs_list = list_local_pdfs()
    new_pdf_dropdown = gr.Dropdown(
        label="Available PDFs (local folder)",
        choices=local_pdfs_list,
        value=dropdown_value,
    )
    return new_pdf_dropdown

def ingest_pdf(uploaded_pdf_path: str):
    """
    uploaded_pdf_path is a temporary filepath provided by Gradio when type="filepath".
    We copy it to PDF_DIR, load it, add metadata, and upsert into vector store.
    """
    if not uploaded_pdf_path:
        return (
            gr.Markdown(value="❌ No file received."),
            refresh_pdf_list(),
        )

    filename = os.path.basename(uploaded_pdf_path)
    dest_path = os.path.join(PDF_DIR, filename)

    # Save locally
    shutil.copy2(uploaded_pdf_path, dest_path)

    # Add to vector store
    status = vector_store_wrapper.add_document_from_filepath(dest_path)

    status = "✅ PDF saved to local folder!\n\n" + status
    
    return (
        gr.Markdown(value=status),
        refresh_pdf_list(dropdown_value=filename),
    )


with gr.Blocks(title="Finsight RAG Chat") as demo:
    gr.Markdown("# Finsight RAG Chat")

    with gr.Row():
        chatbot = gr.Chatbot(height=520)

        with gr.Column(scale=1):
            gr.Markdown("### Debug")
            with gr.Accordion("Debug info", open=False):
                debug_box = gr.Markdown("—")
            
            with gr.Accordion("PDF Management", open=False):
                gr.Markdown("### PDF Manager")

                pdf_upload = gr.File(
                    label="Upload a PDF (it will be saved locally + indexed)",
                    file_types=[".pdf"],
                    type="filepath",
                )
                ingest_btn = gr.Button("Add PDF to Vector DB", variant="primary")
                ingest_status = gr.Markdown("—")
            with gr.Accordion("Local PDFs", open=False):   
                gr.Markdown("### Local PDFs")
                pdf_dropdown = gr.Dropdown(
                    label="Available PDFs (local folder)",
                    choices=list_local_pdfs(),
                    value=None,
                )
                refresh_btn = gr.Button("Refresh list")

                # This will let the user download/open the selected PDF from the UI
                pdf_file_out = gr.File(label="Selected PDF (download/view)")

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
    
    # PDF ingest wiring
    ingest_btn.click(
        ingest_pdf,
        inputs=[pdf_upload],
        outputs=[ingest_status, pdf_dropdown],
    )

    # Refresh list
    refresh_btn.click(
        refresh_pdf_list,
        inputs=None,
        outputs=[pdf_dropdown],
    )

    # Selecting a PDF shows it in a File component (download/view)
    pdf_dropdown.change(
        get_pdf_path,
        inputs=[pdf_dropdown],
        outputs=[pdf_file_out],
    )

if __name__ == "__main__":
    demo.launch(share=False)
