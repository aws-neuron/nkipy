# Building Documentation

This guide explains how to build the NKIPy documentation locally.

## Prerequisites

Install the documentation dependencies:

```bash
uv sync --group docs
```

This installs Sphinx and related packages (sphinx-book-theme, myst-parser, myst-nb, sphinx-autodoc-typehints).

## Building HTML Documentation

From the repository root:

```bash
uv run make -C docs html
```

Or from the `docs/` directory:

```bash
cd docs
uv run make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to view it.

## What Happens During Build

The `make html` command:

1. **Generates API docs** - Runs `generate_api_docs.py` to auto-generate API reference pages from source code
2. **Builds HTML** - Runs Sphinx to convert Markdown/RST files to HTML

## Other Build Targets

```bash
# Clean build artifacts
uv run make -C docs clean

# Build PDF (requires LaTeX)
uv run make -C docs latexpdf

# Check for broken links
uv run make -C docs linkcheck
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── generate_api_docs.py # API doc generator script
├── index.md             # Main landing page
├── installation.md      # Installation guide
├── quickstart.md        # Getting started guide
├── api/                 # API reference (auto-generated)
├── dev_guide/           # Developer guides
├── tutorials/           # Jupyter notebook tutorials
└── user_guide/          # User guides
```

## Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to the table of contents in `index.md` or the relevant section index
3. Rebuild with `uv run make -C docs html`

## Troubleshooting

### Import errors during build

If Sphinx can't import NKIPy modules, ensure you've synced the workspace:

```bash
uv sync
uv sync --group docs
```

### Stale API docs

If API docs are outdated, manually regenerate them:

```bash
uv run python docs/generate_api_docs.py
