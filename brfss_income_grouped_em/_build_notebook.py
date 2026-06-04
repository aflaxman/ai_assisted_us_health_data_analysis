"""Helper to build a notebook from a list of (kind, source) cells."""
import json


def cell(kind, source, cell_id=None):
    src = source if isinstance(source, list) else [source]
    base = {
        'cell_type': kind,
        'metadata': {},
        'source': src,
    }
    if cell_id is not None:
        base['id'] = cell_id
    if kind == 'code':
        base['execution_count'] = None
        base['outputs'] = []
    return base


def build(cells, path):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.12',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)


def md(text, cid=None):
    return cell('markdown', text, cid)


def code(src, cid=None):
    return cell('code', src, cid)
