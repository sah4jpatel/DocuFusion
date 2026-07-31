import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


@pytest.fixture(scope="session")
def simple_pdf(tmp_path_factory):
    """Clean born-digital prose: should route FAST."""
    path = tmp_path_factory.mktemp("pdfs") / "simple.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    body = ("The quarterly business review covers revenue trends across regions. "
            "Customer retention remained strong and operational costs declined. ") * 30
    doc.build([Paragraph("Quarterly Business Review", styles["Title"]),
               Spacer(1, 12), Paragraph(body, styles["Normal"])])
    return path


@pytest.fixture(scope="session")
def math_pdf(tmp_path_factory):
    """Equation-dense page: should route VLM (math density trigger)."""
    path = tmp_path_factory.mktemp("pdfs") / "math.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    y = 720
    for i in range(30):
        c.drawString(72, y, f"f_{i}(x) = ∑ a_i * x^i + ∫ g(t) dt ≈ √(b^2 - 4*a*c) / (2*a) ± ε")
        y -= 20
    c.save()
    return path


@pytest.fixture(scope="session")
def scan_pdf(tmp_path_factory):
    """Full-page image with no text layer: should route VLM (sparse text + image coverage)."""
    import io

    from PIL import Image
    from reportlab.lib.utils import ImageReader

    path = tmp_path_factory.mktemp("pdfs") / "scan.pdf"
    img = Image.new("RGB", (1700, 2200), (240, 238, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(ImageReader(buf), 0, 0, width=8.5 * inch, height=11 * inch)
    c.save()
    return path


@pytest.fixture(scope="session")
def mixed_pdf(tmp_path_factory):
    """Page 1 clean prose, page 2 math-dense — exercises per-page routing."""
    path = tmp_path_factory.mktemp("pdfs") / "mixed.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    prose = ("Plain narrative text describing the project status in complete sentences. ") * 40
    math = " ".join("∑ x_i^2 = ∫ f(t) dt ≈ √(a^2+b^2) ± δ" for _ in range(60))
    doc.build([Paragraph(prose, styles["Normal"]), PageBreak(),
               Paragraph(math, styles["Normal"])])
    return path


def olmocr_reply(body: str = "# Recovered Page\n\nThe equation is \\(E = mc^2\\).",
                 primary_language: str = "en",
                 is_rotation_valid: bool = True,
                 rotation_correction: int = 0,
                 is_table: bool = False,
                 is_diagram: bool = False) -> str:
    """Build a reply in olmOCR-2's real response format.

    The model emits YAML front matter ahead of the Markdown body. Fixtures must
    reproduce that, otherwise the suite silently certifies a client that leaks
    ``---\\nprimary_language: ...`` into every escalated page — which is exactly
    the bug an earlier clean-Markdown mock hid.
    """
    return (
        "---\n"
        f"primary_language: {primary_language}\n"
        f"is_rotation_valid: {str(is_rotation_valid)}\n"
        f"rotation_correction: {rotation_correction}\n"
        f"is_table: {str(is_table)}\n"
        f"is_diagram: {str(is_diagram)}\n"
        "---\n"
        f"{body}"
    )


class _MockVLLMHandler(BaseHTTPRequestHandler):
    """Emulates vLLM's OpenAI-compatible /v1/chat/completions endpoint."""

    def log_message(self, *_):  # silence
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)

        # Real-vLLM behavior: reject OpenAI's structured-outputs contract.
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") not in (None, "text", "json_object"):
            self._send(400, {"error": {"message":
                "1 validation error for ChatCompletionRequest\nresponse_format.type\n"
                "  Input should be 'text' or 'json_object'"}})
            return

        text = "".join(
            part.get("text", "")
            for msg in body.get("messages", [])
            for part in (msg["content"] if isinstance(msg["content"], list) else [])
            if isinstance(part, dict) and part.get("type") == "text"
        )

        if "JSON Schema" in text or (isinstance(rf, dict) and rf.get("type") == "json_object"):
            # Marker path: fenced JSON, to exercise tolerant extraction.
            content = "```json\n" + json.dumps(self.server.json_reply) + "\n```"
            finish_reason = "stop"
        else:
            # olmOCR path: a scripted queue lets a test drive the retry ladder.
            if self.server.reply_queue:
                content, finish_reason = self.server.reply_queue.pop(0)
            else:
                content, finish_reason = self.server.markdown_reply, "stop"

        self._send(200, {
            "id": "cmpl-mock", "object": "chat.completion", "created": 0,
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def mock_vllm():
    server = HTTPServer(("127.0.0.1", 0), _MockVLLMHandler)
    server.requests = []
    server.json_reply = {"corrected_markdown": "| a | b |\n|---|---|\n| 1 | 2 |"}
    server.markdown_reply = olmocr_reply()
    # (content, finish_reason) pairs consumed one per request, ahead of markdown_reply.
    server.reply_queue = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
