"""Tiny helpers for building the numbered notebooks from ``_build_*.py`` scripts.

Each build script assembles a list of cells with :func:`md` / :func:`code`, then
calls :func:`write_notebook`. Notebooks are written with empty outputs; execute
them with ``jupyter nbconvert --execute`` (see the project README / CLAUDE.md).
"""
import json
from pathlib import Path


def md(source, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(path)
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")
    return path
