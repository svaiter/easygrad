#!/usr/bin/env python

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from easygrad import numeric, symbolic  # noqa: E402
from easygrad.scalar_reverse import Value, topo_sort  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ASSETS.parent.parent)}")


# ---------------------------------------------------------------------------
# §7.2 finite-difference error U-curve
# ---------------------------------------------------------------------------
def fig_finite_difference() -> None:
    f = lambda x: np.exp(0.5 * x)
    fp = lambda x: 0.5 * np.exp(0.5 * x)
    x0 = 0.7
    hs = np.logspace(-1, -15, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for scheme, slope in (
        ("forward", r"$\sqrt{\varepsilon}$"),
        ("central", r"$\varepsilon^{1/3}$"),
    ):
        err = numeric.error_curve(f, fp, x0, hs, scheme=scheme)
        ax.loglog(hs, err, label=f"{scheme} (opt $h\\sim${slope})")
    ax.set_xlabel("step size $h$")
    ax.set_ylabel("absolute error")
    ax.set_title("Finite-difference error: truncation vs. cancellation")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fd_error_curve.svg")


# ---------------------------------------------------------------------------
# §7.4 complex step overlaid on the U-curve (flat, no wall)
# ---------------------------------------------------------------------------
def fig_complex_step() -> None:
    f = lambda x: np.exp(0.5 * x)
    fp = lambda x: 0.5 * np.exp(0.5 * x)
    x0 = 0.7
    hs_fd = np.logspace(-1, -15, 200)
    hs_cs = np.logspace(-1, -30, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(
        hs_fd,
        numeric.error_curve(f, fp, x0, hs_fd, "central"),
        label="central difference",
    )
    ax.loglog(
        hs_cs,
        np.maximum(numeric.complex_step_error_curve(f, fp, x0, hs_cs), 1e-18),
        label="complex step",
    )
    ax.set_xlabel("step size $h$")
    ax.set_ylabel("absolute error")
    ax.set_title("Complex step has no roundoff wall")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "complex_step.svg")


# ---------------------------------------------------------------------------
# §7.3 symbolic expression swell
# ---------------------------------------------------------------------------
def fig_symbolic_swell() -> None:
    data = symbolic.swell_demo(max_iter=6)
    orders = [o for o, _ in data]
    sizes = [s for _, s in data]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(orders, sizes, "o-")
    ax.set_xlabel("logistic-map iterations $n$")
    ax.set_ylabel("derivative node count (log scale)")
    ax.set_title("Expression swell: derivative of the iterated logistic map")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "symbolic_swell.svg")


# ---------------------------------------------------------------------------
# §7.5 annotated dual-number trace (primal + tangent propagate together)
# ---------------------------------------------------------------------------
def fig_dual_trace() -> None:
    # f(x) = sin(x) * x at x = 1.1, seed tangent 1
    from easygrad import dual
    from easygrad.dual import Dual

    x = Dual(1.1, 1.0)
    s = dual.sin(x)
    out = s * x
    rows = [
        ("x", x.primal, x.tangent),
        ("sin(x)", s.primal, s.tangent),
        ("sin(x)*x", out.primal, out.tangent),
    ]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.axis("off")
    ax.set_title("Dual-number trace of  f(x) = sin(x)·x  at x = 1.1")
    cell = [[f"{p:+.4f}", f"{t:+.4f}"] for _, p, t in rows]
    table = ax.table(
        cellText=cell,
        rowLabels=[r[0] for r in rows],
        colLabels=["primal (value)", "tangent (derivative)"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.8)
    _save(fig, "dual_trace.svg")


# ---------------------------------------------------------------------------
# graphviz DAGs (need the `dot` binary; degrade gracefully if missing)
# ---------------------------------------------------------------------------
def _render_dag(root, topo, label, name: str) -> bool:
    """Render a computational-graph DAG to ``docs/assets/<name>.svg``.

    ``topo`` is the topological order of ``root``; ``label(node)`` returns the
    box text. Returns False (and cleans up the orphan dot source) when the
    system ``dot`` binary is unavailable.
    """
    try:
        import graphviz
    except Exception:  # pragma: no cover
        return False

    g = graphviz.Digraph(graph_attr={"rankdir": "LR"})
    ids: dict[int, str] = {}

    def node_id(v) -> str:
        if id(v) not in ids:
            ids[id(v)] = f"n{len(ids)}"
        return ids[id(v)]

    for v in topo:
        nid = node_id(v)
        g.node(nid, label(v), shape="box")
        for child in v._prev:
            g.edge(node_id(child), nid)
    try:
        g.render(ASSETS / name, format="svg", cleanup=True)
        print(f"wrote docs/assets/{name}.svg")
        return True
    except Exception as exc:  # graphviz.ExecutableNotFound and friends
        # render() writes the dot source before invoking `dot`; drop the orphan
        (ASSETS / name).unlink(missing_ok=True)
        print(f"skip {name}.svg: {exc}")
        return False


def fig_scalar_dag() -> bool:
    # f(a,b) = a*b + exp(a)  at a=2, b=3, with grads after backward()
    a, b = Value(2.0), Value(3.0)
    a._op, b._op = "a", "b"
    out = a * b + a.exp()
    out.backward()
    return _render_dag(
        out,
        topo_sort(out),
        lambda v: f"{v._op or 'leaf'}\ndata={v.data:.3f}\ngrad={v.grad:.3f}",
        "scalar_dag",
    )


def fig_vector_dag() -> bool:
    # §7.8: DAG of a 1-layer MLP loss over numpy arrays, with shapes + grads
    import numpy as _np
    from easygrad import reverse as _rev
    from easygrad.reverse import Node, _topo

    rng = _np.random.default_rng(0)
    X = Node(rng.standard_normal((4, 3)))
    y = Node(rng.standard_normal((4, 2)))
    W = Node(_np.zeros((3, 2)))
    b = Node(_np.zeros((2,)))
    X._op, y._op, W._op, b._op = "X", "y", "W", "b"
    pred = _rev.tanh(X @ W + b)
    diff = pred - y
    loss = (diff * diff).mean()
    loss.backward()

    def label(v) -> str:
        shape = "×".join(map(str, v.value.shape)) or "scalar"
        gnorm = float(_np.linalg.norm(v.grad))
        return f"{v._op or 'op'}\nshape={shape}\n‖grad‖={gnorm:.3f}"

    return _render_dag(loss, _topo(loss), label, "vector_dag")


def fig_eager_vs_traced() -> None:
    """Side-by-side: eager tape ops vs. the traced IR for the same function."""

    def prog(x):
        return texp(x * 2.0) + x

    jaxpr = make_jaxpr(prog, 1)()

    # eager side: list the ops as they would execute
    eager_lines = [
        "x            # leaf tensor",
        "t0 = x * 2.0 # executes now, records grad_fn",
        "t1 = exp(t0) # executes now, records grad_fn",
        "y  = t1 + x  # executes now, records grad_fn",
        "y.backward() # walk tape in reverse",
    ]
    traced_lines = str(jaxpr).splitlines()

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, title, lines in (
        (axl, "PyTorch: eager define-by-run tape", eager_lines),
        (axr, "JAX: staged IR (jaxpr) you transform", traced_lines),
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        ax.text(
            0.0,
            0.95,
            "\n".join(lines),
            family="monospace",
            fontsize=9,
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
    _save(fig, "eager_vs_traced.svg")


def fig_mlp_training() -> None:
    """Capstone: train a 2-layer ReLU MLP to fit sin with plain gradient descent.

    Left: the loss curve. Right: the learned function vs. the target -- both
    produced by the same reverse-mode engine the rest of the site builds, driven
    by the ``gradient_descent`` helper the student writes in
    ``easygrad.train`` (see docs/train-mlp.md).
    """
    from easygrad import reverse
    from easygrad.reverse import Node
    from easygrad.train import gradient_descent

    rng = np.random.default_rng(0)
    X = np.linspace(-1.0, 1.0, 64).reshape(64, 1)
    y = np.sin(np.pi * X)
    H = 16

    def loss(W1, b1, W2, b2):
        h = reverse.relu(Node(X) @ W1 + b1)
        return ((h @ W2 + b2 - Node(y)) ** 2).mean()

    def predict(params):
        W1, b1, W2, b2 = params
        return np.maximum(X @ W1 + b1, 0.0) @ W2 + b2

    params = [
        rng.standard_normal((1, H)),
        rng.standard_normal(H),
        rng.standard_normal((H, 1)) / np.sqrt(H),
        np.zeros(1),
    ]

    # plain gradient descent suffices on this gentle target (see docs/train-mlp.md)
    steps = 1500
    params, history = gradient_descent(loss, params, lr=0.1, steps=steps)

    # thin the loss curve for plotting
    steps_h = list(range(0, steps, 15))
    loss_h = [history[i] for i in steps_h]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 3.6))
    axl.semilogy(steps_h, loss_h)
    axl.set_xlabel("gradient-descent step")
    axl.set_ylabel("mean squared error (log)")
    axl.set_title("Training loss")
    axl.grid(True, which="both", alpha=0.3)

    axr.plot(X.ravel(), y.ravel(), "o", ms=3, label="target $\\sin 2\\pi x$")
    axr.plot(X.ravel(), predict(params).ravel(), "-", label="MLP fit")
    axr.set_xlabel("$x$")
    axr.set_title(f"Learned function (final MSE {history[-1]:.2e})")
    axr.legend()
    _save(fig, "mlp_training.svg")


def main() -> None:
    fig_finite_difference()
    fig_complex_step()
    fig_symbolic_swell()
    fig_dual_trace()
    fig_eager_vs_traced()
    fig_mlp_training()
    has_dot = fig_scalar_dag()
    fig_vector_dag()
    if not has_dot:
        print(
            "note: graphviz `dot` binary not found -- DAG figures skipped. "
            "Install system graphviz to regenerate them."
        )


if __name__ == "__main__":
    main()
