"""Final checks that apply to both submitted files. Run by hand under any Python with numpy and pandas:
    python research/both_checks.py
Prints the official numbers so two environments can be compared line by line, plus static checks on the source."""
import ast
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import harness_public as H  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


print('python', sys.version.split()[0], 'pandas', pd.__version__, 'numpy', np.__version__)
s = H.run_task1(os.path.join(ROOT, 'task1.py'))
print('T1 %.10f %.10f %.10f %d' % (s['cagr'], s['max_dd'], s['sharpe'], s['trades']))
s = H.run_task2(os.path.join(ROOT, 'task2.py'))
print('T2 %.10f %.10f %.10f %d' % (s['cagr'], s['max_dd'], s['sharpe'], s['trades']))

for name in ['task1', 'task2']:
    src = open(os.path.join(ROOT, name + '.py'), encoding='utf-8').read()
    ast.parse(src, feature_version=(3, 9))
    tree = ast.parse(src)
    imports = sorted({a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})
    calls = sorted({n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})
    code = '\n'.join(l for l in src.splitlines())
    body = ast.get_docstring(tree) or ''
    no_doc = code.replace(body, '')
    print('%s: parses as Python 3.9, imports %s, n PARAMS %d' % (name, imports, len(load(name).PARAMS)))
    print('   file or network access:', [c for c in calls if c in ('open', 'exec', 'eval', '__import__')] or 'none',
          '| read_csv etc:', bool(re.search(r'read_csv|read_parquet|to_csv|urlopen|requests', src)),
          '| random:', bool(re.search(r'\brandom\b', src)),
          '| 4-digit year or date literal in code:', re.findall(r"20\d\d[-/]\d\d|Timestamp\(", no_doc) or 'none')
