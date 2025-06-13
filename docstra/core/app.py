# File: ./docstra/core/app.py

from pathlib import Path
import subprocess
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/docs", StaticFiles(directory="docs"), name="docs")
templates = Jinja2Templates(directory="templates")


@app.post("/generate-documentation")
async def generate_docs(
    request: Request,
    files: List[UploadFile] = File(...),
    output_format: str = Form("html"),
):
    """Generate documentation for uploaded files."""
    # Create temp directory for uploads
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    # Process each uploaded file
    file_paths = []
    for file in files:
        # Handle case where filename might be None
        filename = file.filename or "uploaded_file"
        file_path = temp_dir / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(str(file_path))

    # Generate documentation
    doc_dir = Path("./docs/generated")
    doc_dir.mkdir(exist_ok=True, parents=True)

    # Run the generation (this could use the function from CLI)
    for path in file_paths:
        subprocess.run(
            [
                "python",
                "-m",
                "docstra",
                "generate",
                path,
                "--output",
                str(doc_dir),
                "--format",
                output_format,
            ]
        )

    # Return documentation URL
    doc_url = request.url_for("docs", path="/generated/index.html")
    return {"status": "success", "documentation_url": doc_url}


@app.get("/documentation", response_class=HTMLResponse)
async def documentation_ui(request: Request):
    """Documentation generation UI."""
    return templates.TemplateResponse("documentation.html", {"request": request})
