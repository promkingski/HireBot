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
            import logging
            logging.error(f"Vertex AI call failed: {e}")
            # Fall back to simple summary
            sentences = text.split(".")
            return ".".join(sentences[:2]).strip() + "."
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

@app.get("/debug")
async def debug():
    return {"USE_VERTEX": USE_VERTEX}

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
