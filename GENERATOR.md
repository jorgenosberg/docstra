# Docstra Documentation Generator

Docstra can generate documentation sites for existing repositories by combining code analysis, LLM summaries, and MkDocs output.

## Local setup

```bash
git clone https://github.com/jorgenosberg/docstra.git
cd docstra
uv sync --locked --all-groups
```

If you only want the published CLI, use `uv tool install docstra` instead.

## Prepare the docs environment

```bash
uv run python -m docstra.core.documentation.setup --init --output ./docs
```

## Generate documentation

```bash
uv run docstra generate ./your-project
uv run docstra generate ./your-project --output ./docs --format mkdocs
uv run docstra generate ./your-project --serve
uv run docstra generate ./your-project --exclude "tests/*" --exclude "*.pyc"
```

## Serve generated docs

```bash
cd ./docs
mkdocs serve
```

Or let Docstra serve the generated output directly:

```bash
uv run docstra generate ./your-project --serve --port 8080
```

## Configuration

Docstra can read a project config file such as:

```yaml
model:
  provider: anthropic
  model_name: claude-3-opus-20240229
  temperature: 0.7

processing:
  exclude_patterns:
    - .git
    - __pycache__
    - node_modules
    - venv
```

You can also configure it through the CLI:

```bash
uv run docstra config --model anthropic
uv run docstra config --show
```

## Troubleshooting

If the documentation helpers are missing packages in a fresh clone, resync the repo environment:

```bash
uv sync --locked --all-groups
```
