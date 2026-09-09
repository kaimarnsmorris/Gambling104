"""Pins that the template is importable and uses the public surface.

The template teaches the pattern; if it shows the old sys.path dance, every
new investigation copies it.
"""
import os
import re

TEMPLATE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        os.pardir, "investigations", "_template", "run.py")


def test_the_template_exists():
    assert os.path.exists(os.path.abspath(TEMPLATE))


def test_the_template_does_not_manipulate_sys_path():
    src = open(os.path.abspath(TEMPLATE)).read()
    assert "sys.path" not in src, (
        "the template must import harness as an installed package")


def test_the_template_uses_the_public_entry_point():
    src = open(os.path.abspath(TEMPLATE)).read()
    assert re.search(r"from harness import .*backtest", src)
