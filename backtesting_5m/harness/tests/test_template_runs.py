"""Pins that the template is importable, uses the public surface, and that
its `require=` clause is satisfiable by what it actually supplies.

The template teaches the pattern; if it shows the old sys.path dance, every
new investigation copies it. And `Sample(require=(...))` drops episodes
silently rather than raising -- a template whose `require=` names a stream or
panel it never supplies quietly selects zero markets, produces an empty plot,
and exits 0. That shipped once (an all-False `has_spot` against a
`require=("spot", "chainlink")` sample, with no `spot_path` given to
`load_episodes`), so this file pins the two lists staying consistent.

Running the real template needs real data (a spot panel, a Chainlink oracle
capture) that a test environment does not have, so these tests do not import
or execute the template. They check the part that does not need data: every
name the template requires, it also supplies data for, by construction of the
calls in its own source. That would catch a `require=` naming a stream/panel
the template forgets to wire up (this bug, or its `require=("chainlink",)`
sibling), but it would NOT catch the source of data being wrong, empty, or
outside the requested day range -- only that the template asks for what it
also feeds.
"""
import ast
import os

TEMPLATE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        os.pardir, "investigations", "_template", "run.py")


def _source():
    return open(os.path.abspath(TEMPLATE)).read()


def _tree():
    return ast.parse(_source())


def _str_tuple_elts(node):
    """String constants inside a Tuple/List AST node, else ()."""
    if not isinstance(node, (ast.Tuple, ast.List)):
        return ()
    return tuple(e.value for e in node.elts if isinstance(e, ast.Constant)
                and isinstance(e.value, str))


def _call_kwargs(tree, func_name):
    """kwarg name -> value node, for every call to a function/name literally
    spelled `func_name` anywhere in the module (there may be more than one,
    e.g. `load_episodes` called once but `backtest` called through `one()`)."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name == func_name:
            calls.append({kw.arg: kw.value for kw in node.keywords if kw.arg})
    return calls


def _required_names():
    """Every name passed to `Sample(require=(...))` in the template."""
    names = set()
    for kwargs in _call_kwargs(_tree(), "Sample"):
        if "require" in kwargs:
            names.update(_str_tuple_elts(kwargs["require"]))
    return names


def _supplied_stream_names():
    """Names the template feeds through `streams=(...)`, to either
    `load_episodes` or `backtest` -- both must carry a stream for
    `select_episodes` to find it on the episode."""
    names = set()
    for func in ("load_episodes", "backtest"):
        for kwargs in _call_kwargs(_tree(), func):
            if "streams" in kwargs:
                names.update(_str_tuple_elts(kwargs["streams"]))
    return names


def _load_episodes_has_spot_path():
    for kwargs in _call_kwargs(_tree(), "load_episodes"):
        if "spot_path" in kwargs:
            return True
    return False


def test_the_template_exists():
    assert os.path.exists(os.path.abspath(TEMPLATE))


def test_the_template_does_not_manipulate_sys_path():
    assert "sys.path" not in _source(), (
        "the template must import harness as an installed package")


def test_the_template_uses_the_public_entry_point():
    import re
    assert re.search(r"from harness import .*backtest", _source())


def test_every_required_stream_name_is_supplied_to_load_episodes():
    """Every non-"spot" name in `require=` must also appear in a `streams=`
    tuple somewhere in the template, or `select_episodes` drops every
    episode as missing that stream -- silently, per B2."""
    supplied = _supplied_stream_names()
    for name in _required_names() - {"spot"}:
        assert name in supplied, (
            f"template requires {name!r} but never passes it in streams=")


def test_requiring_spot_is_paired_with_a_spot_path():
    """`require=("spot", ...)` needs `ep.has_spot` set, which needs
    `load_episodes(spot_path=...)`. Without it every episode's `has_spot` is
    all-False and `require=("spot",...)` drops the whole sample -- exactly
    the bug this test exists to catch."""
    if "spot" in _required_names():
        assert _load_episodes_has_spot_path(), (
            "template requires 'spot' but load_episodes() is never given "
            "spot_path=")


def test_the_template_checks_for_zero_selected_markets():
    """The template must inspect summary['sample']['n_selected'] (or fail
    another way) rather than silently plotting an empty sample -- `run()`
    itself stays quiet on an empty sample by design, so the loudness has to
    live here."""
    src = _source()
    assert "n_selected" in src


def _load_episodes_kwarg_names():
    names = set()
    for kwargs in _call_kwargs(_tree(), "load_episodes"):
        names.update(kwargs)
    return names


def test_using_the_basis_learning_model_is_paired_with_an_rtds_path():
    """Naming the `chainlink` STREAM is not the same as loading the oracle.

    `models/normal_qq/fair.py` learns the BTC/USDT-to-BTC/USD basis against
    the oracle, and it reads the built-in array `ep.chainlink`, which only
    exists when `load_episodes` is given `rtds_path=`. Passing
    `streams=("chainlink",)` populates `ep.streams["chainlink"]` instead --
    a different attribute, which that block never touches.

    The two look interchangeable and are not. With only the stream, every
    `s` is NaN: the run selects every market, quotes on none, fills nothing,
    reports a $0.00 headline and exits 0. That shipped, and neither the
    `require=` check above nor the zero-market check caught it, because the
    sample was full and only the fills were empty.
    """
    if "normal_qq" not in _source():
        return
    assert "rtds_path" in _load_episodes_kwarg_names(), (
        "template uses the basis-learning model but never passes rtds_path= "
        "to load_episodes; ep.chainlink is then absent and every s is NaN")


def test_the_template_checks_for_zero_fills():
    """Zero markets and zero fills are different silent failures.

    A model that quotes nothing keeps every market and fills none, so the
    n_selected check passes and the plot is a flat line at zero. The
    template has to reject that too."""
    assert "n_fills" in _source(), (
        "template must fail on a run that selected markets but filled "
        "nothing -- a flat-zero plot is not a result")
