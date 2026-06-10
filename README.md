# Docstra

Docstra is an LLM-powered tool for generating and querying codebase documentation. It ships as a CLI and also includes a small FastAPI app for browser-based workflows.

## Install

### Use the published CLI

```bash
uv tool install docstra
docstra --help
```

For one-off runs, you can also use `uvx docstra --help`.

### Work on the repo locally

```bash
git clone https://github.com/jorgenosberg/docstra.git
cd docstra
uv sync
uv run docstra --help
```

The repo is pinned to Python 3.12 in `.python-version`. `uv sync` will create `.venv` and install the right interpreter if needed.

## Configure

Docstra stores project-specific state in a `.docstra/` directory at the root of the codebase you are indexing. That directory contains generated embeddings, indexes, and the local `.env` file used for provider credentials.

Run `docstra init` in the target repository to create or update that configuration:

```bash
uv run docstra init
```

Remember to keep `.docstra/` out of version control.

## Common commands

```bash
uv run docstra init
uv run docstra ingest
uv run docstra query "How does authentication work?"
uv run docstra chat
```

## FastAPI app

Start the bundled FastAPI app from the repo with:

```bash
uv run uvicorn docstra.core.app:app --reload
```

The app will be available at `http://127.0.0.1:8000`.

## Developer workflow

Use `uv` for the local environment and lockfile:

```bash
uv sync --locked --all-groups
uv lock --check
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync ty check
uv run --locked --no-sync pip-audit
uv run --locked --no-sync pytest
```

Install the Git hooks once per clone:

```bash
uv run pre-commit install
```

`uv` already defaults to the `first-index` resolution strategy. This repo pins that behavior explicitly in `pyproject.toml` so any future custom index setup still prefers the first matching index and avoids dependency-confusion fallback.

## Documentation generator

The generator can build MkDocs output for another repository:

```bash
uv run docstra generate ./your-project --output ./docs --format mkdocs
cd ./docs
mkdocs serve
```
