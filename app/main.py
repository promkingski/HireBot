import os
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage

# Optional: import Vertex AI client
try:
    from google import genai
    from google.genai.types import HttpOptions
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
    if USE_VERTEX:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), 
                      location="us-central1")
        model = GenerativeModel("gemini-1.5-flash-001")
        response = model.generate_content(
            f"Summarize the following text in 2-3 sentences:\n\n{text}"
        )
        return response.text.strip()
    else:
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
