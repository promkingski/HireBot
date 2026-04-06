import os
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage

# Optional: import Vertex AI client
try:
    from vertexai.preview.language_models import TextGenerationModel
    USE_VERTEX = True
except ImportError:
    USE_VERTEX = False

BUCKET_NAME = os.environ.get("BUCKET_NAME")

app = FastAPI(title="Mission Log Analyzer")

# Initialize Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def upload_to_gcs(data: bytes, filename: str) -> str:
    """
    Uploads file data to Cloud Storage and returns the public URI.
    """
    blob = bucket.blob(filename)
    blob.upload_from_string(data)
    return f"gs://{BUCKET_NAME}/{filename}"


def summarise_text(text: str) -> str:
    """
    Summarises text using Vertex AI if available; otherwise a simple local
    method.
    """
    if USE_VERTEX:
        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(text, max_output_tokens=128)
        return response.text.strip()
    else:
        # Simple fallback: return the first 2 sentences as a pseudo-summary
        sentences = text.split(".")
        return ".".join(sentences[:2]).strip() + "."


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <body>
        <h1>Mission Log Analyzer</h1>
        <form action="/analyze" enctype="multipart/form-data" method="post">
          <input type="file" name="files" multiple>
          <input type="submit" value="Analyze">
        </form>
      </body>
    </html>
    """


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        content = await file.read()
        gcs_uri = upload_to_gcs(content, file.filename)
        text = content.decode("utf-8", errors="ignore")
        summary = summarise_text(text)

        results.append(
            {
                "filename": file.filename,
                "gcs_uri": gcs_uri,
                "summary": summary,
            }
        )

    return JSONResponse(content={"results": results})