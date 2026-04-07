# app/main.py
import os
import io
import json
import logging
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel

# PDF extraction
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# Vertex AI / Gemini
try:
    from google import genai
    from google.genai.types import HttpOptions
    USE_VERTEX = True
except ImportError:
    USE_VERTEX = False

# ─── CONFIG ───
BUCKET_NAME = os.environ.get("BUCKET_NAME")
KNOWLEDGE_PREFIX = "knowledge/"  # prefix in bucket for chatbot knowledge docs

app = FastAPI(title="Mission Log Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

logger = logging.getLogger(__name__)

# ─── SYSTEM PROMPT ───
BASE_SYSTEM_PROMPT = """You are an AI hiring agent representing a cloud and platform engineer candidate. Your role is to communicate with hiring managers, answer questions about the candidate’s qualifications, experience, and approach, and position the candidate effectively for relevant opportunities.

HOW TO RESPOND:

Speak in third person, referring to the candidate as “Carter” or “the candidate.”
Represent the candidate confidently and accurately, as a knowledgeable advocate.
If a topic is not explicitly covered in the knowledge base, explain how the candidate would approach it using transferable experience and the ability to learn quickly.
Provide concrete, technical answers. Reference specific services, patterns, and real-world decision-making where applicable.
Keep responses focused, practical, and high-signal — prioritize clarity over fluff.
Tone: professional, technically fluent, and advisory. Avoid sounding salesy; aim to be credible and informative.

KNOWLEDGE BASE (loaded from admin-uploaded documents):
"""


# ─── HELPERS ───

def upload_to_gcs(data: bytes, filename: str) -> str:
    blob = bucket.blob(filename)
    blob.upload_from_string(data)
    return f"gs://{BUCKET_NAME}/{filename}"


def extract_text(content: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf") and HAS_PDF:
        try:
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return content.decode("utf-8", errors="ignore")
    else:
        return content.decode("utf-8", errors="ignore")


def summarise_text(text: str) -> str:
    if USE_VERTEX:
        try:
            client = genai.Client(
                http_options=HttpOptions(api_version="v1"),
                vertexai=True
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Summarize the following text in 2-3 sentences:\n\n{text}"
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Vertex AI call failed: {e}")
            sentences = text.split(".")
            return ".".join(sentences[:2]).strip() + "."
    else:
        sentences = text.split(".")
        return ".".join(sentences[:2]).strip() + "."


def load_knowledge() -> str:
    """Load all knowledge documents from Cloud Storage."""
    try:
        blobs = list(bucket.list_blobs(prefix=KNOWLEDGE_PREFIX))
        knowledge_parts = []
        for blob in blobs:
            if blob.name == KNOWLEDGE_PREFIX:
                continue  # skip the "directory" itself
            if blob.name.endswith(".meta.json"):
                continue  # skip metadata files
            try:
                content = blob.download_as_text()
                # Try to load corresponding metadata
                meta_blob = bucket.blob(blob.name + ".meta.json")
                title = blob.name.replace(KNOWLEDGE_PREFIX, "")
                if meta_blob.exists():
                    meta = json.loads(meta_blob.download_as_text())
                    title = meta.get("title", title)
                knowledge_parts.append(f"--- {title} ---\n{content}")
            except Exception as e:
                logger.error(f"Failed to load knowledge blob {blob.name}: {e}")
        return "\n\n".join(knowledge_parts)
    except Exception as e:
        logger.error(f"Failed to load knowledge: {e}")
        return ""


def chat_with_gemini(messages: list, knowledge: str) -> str:
    if not USE_VERTEX:
        return "The AI service is not available. Please check that Vertex AI is configured."

    system_prompt = BASE_SYSTEM_PROMPT + (knowledge if knowledge else "(No knowledge documents uploaded yet. Answer based on general cloud engineering expertise.)")

    try:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=True
        )

        # Build the conversation for Gemini
        # Gemini expects alternating user/model roles
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        from google.genai.types import GenerateContentConfig

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_contents,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=1024,
                temperature=0.7,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Chat Gemini call failed: {e}")
        return f"I encountered an error processing your question. Please try again. (Error: {str(e)})"


# ─── PAGE ROUTES ───

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active": "home"})

@app.get("/architecture", response_class=HTMLResponse)
async def architecture(request: Request):
    return templates.TemplateResponse(request, "architecture.html", {"active": "architecture"})

@app.get("/analyzer", response_class=HTMLResponse)
async def analyzer(request: Request):
    return templates.TemplateResponse(request, "analyzer.html", {"active": "analyzer"})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"active": "chat"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html", {"active": "admin"})


# ─── API ROUTES ───

@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        gcs_uri = upload_to_gcs(content, file.filename)
        text = extract_text(content, file.filename)
        if not text:
            summary = "Could not extract text from this file."
        else:
            summary = summarise_text(text)
        results.append({
            "filename": file.filename,
            "gcs_uri": gcs_uri,
            "summary": summary
        })
    return JSONResponse(content={"results": results})


class ChatRequest(BaseModel):
    messages: list


@app.post("/chat/message")
async def chat_message(req: ChatRequest):
    knowledge = load_knowledge()
    reply = chat_with_gemini(req.messages, knowledge)
    return JSONResponse(content={"reply": reply})


# ─── ADMIN API ROUTES ───

class KnowledgeInput(BaseModel):
    title: str
    content: str


@app.get("/admin/knowledge")
async def list_knowledge():
    try:
        blobs = list(bucket.list_blobs(prefix=KNOWLEDGE_PREFIX))
        items = []
        for blob in blobs:
            if blob.name == KNOWLEDGE_PREFIX:
                continue
            if blob.name.endswith(".meta.json"):
                continue
            filename = blob.name.replace(KNOWLEDGE_PREFIX, "")
            title = filename
            # Load metadata if exists
            meta_blob = bucket.blob(blob.name + ".meta.json")
            if meta_blob.exists():
                try:
                    meta = json.loads(meta_blob.download_as_text())
                    title = meta.get("title", title)
                except:
                    pass
            items.append({
                "filename": filename,
                "title": title,
                "size": blob.size or 0,
            })
        return JSONResponse(content={"items": items})
    except Exception as e:
        logger.error(f"Failed to list knowledge: {e}")
        return JSONResponse(content={"items": []})


@app.post("/admin/knowledge")
async def save_knowledge(req: KnowledgeInput):
    try:
        # Create a filename from the title
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in req.title)
        safe_name = safe_name.strip().replace(' ', '_').lower()
        if not safe_name:
            safe_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{safe_name}.txt"
        blob_name = f"{KNOWLEDGE_PREFIX}{filename}"

        # Save content
        blob = bucket.blob(blob_name)
        blob.upload_from_string(req.content, content_type="text/plain")

        # Save metadata
        meta_blob = bucket.blob(f"{blob_name}.meta.json")
        meta = {"title": req.title, "created": datetime.now().isoformat()}
        meta_blob.upload_from_string(json.dumps(meta), content_type="application/json")

        return JSONResponse(content={"status": "ok", "filename": filename})
    except Exception as e:
        logger.error(f"Failed to save knowledge: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/admin/knowledge/{filename}")
async def get_knowledge(filename: str):
    try:
        blob_name = f"{KNOWLEDGE_PREFIX}{filename}"
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()

        title = filename
        meta_blob = bucket.blob(f"{blob_name}.meta.json")
        if meta_blob.exists():
            meta = json.loads(meta_blob.download_as_text())
            title = meta.get("title", title)

        return JSONResponse(content={"title": title, "content": content, "filename": filename})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)


@app.delete("/admin/knowledge/{filename}")
async def delete_knowledge(filename: str):
    try:
        blob_name = f"{KNOWLEDGE_PREFIX}{filename}"
        blob = bucket.blob(blob_name)
        blob.delete()

        # Also delete metadata
        meta_blob = bucket.blob(f"{blob_name}.meta.json")
        if meta_blob.exists():
            meta_blob.delete()

        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        logger.error(f"Failed to delete knowledge: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── DEBUG ───

@app.get("/debug")
async def debug():
    return {
        "USE_VERTEX": USE_VERTEX,
        "HAS_PDF": HAS_PDF,
        "BUCKET_NAME": BUCKET_NAME,
    }
