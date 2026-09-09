"""Guards against functions that are implemented, tested, and reachable by
nothing.

The whole-branch review found apply_daily_minimum unreachable from any
production code path. Green tests over unreachable code is the failure mode
per-task review structurally cannot catch. fingerprint_input is already
reachable; its assertion here is a regression guard against it silently
becoming dead again.
"""
import ast
import os

HARNESS = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


def _calls_in_package():
    names = set()
    for root, _, files in os.walk(HARNESS):
        if "tests" in root or "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, f)).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name):
                        names.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        names.add(fn.attr)
    return names


def test_apply_daily_minimum_is_reachable_from_production_code():
    assert "apply_daily_minimum" in _calls_in_package(), (
        "apply_daily_minimum is implemented and tested but called by nothing; "
        "wire it into run() behind an ExecConfig flag, or delete it")


def test_fingerprint_input_is_reachable():
    assert "fingerprint_input" in _calls_in_package()
