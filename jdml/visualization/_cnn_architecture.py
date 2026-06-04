from types import SimpleNamespace


def _normalize_model_config(config_or_model):
    """Return a CNNModel-style config dictionary from a dict, namespace, or model."""
    if isinstance(config_or_model, dict):
        return dict(config_or_model)
    if isinstance(config_or_model, SimpleNamespace):
        return vars(config_or_model).copy()
    if hasattr(config_or_model, "get_config"):
        return config_or_model.get_config()
    return vars(config_or_model).copy()


def _cnn_architecture_steps(config):
    """Build display-ready layer metadata from a CNNModel config dictionary."""
    input_channels, h, w = config["input_dim"]
    num_classes = config["num_classes"]
    conv_layers = config["conv_layers"]
    fc_layers = config.get("fc_layers") or []
    pool_config = config.get("pool_config", ("max", 2, 2))
    use_gap = config.get("use_gap", False)
    dropout = config.get("dropout", 0.0)

    steps = [
        {
            "kind": "input",
            "title": "Input",
            "shape": f"{input_channels} x {h} x {w}",
            "details": "",
            "size": input_channels * h * w,
        }
    ]

    in_channels = input_channels
    for i, (out_channels, kernel_size, stride, padding, use_pool) in enumerate(conv_layers, start=1):
        h = (h + 2 * padding - kernel_size) // stride + 1
        w = (w + 2 * padding - kernel_size) // stride + 1
        steps.append(
            {
                "kind": "conv",
                "title": f"Conv {i}",
                "shape": f"{out_channels} x {h} x {w}",
                "details": f"{in_channels}->{out_channels}, k={kernel_size}, s={stride}, p={padding}",
                "size": out_channels * h * w,
            }
        )

        if use_pool:
            h = h // pool_config[2]
            w = w // pool_config[2]
            steps.append(
                {
                    "kind": "pool",
                    "title": f"{pool_config[0].title()} Pool {i}",
                    "shape": f"{out_channels} x {h} x {w}",
                    "details": f"k={pool_config[1]}, s={pool_config[2]}",
                    "size": out_channels * h * w,
                }
            )
        in_channels = out_channels

    if use_gap:
        steps.append(
            {
                "kind": "gap",
                "title": "Global Avg Pool",
                "shape": f"{in_channels} x 1 x 1",
                "details": f"from {in_channels} x {h} x {w}",
                "size": in_channels,
            }
        )
        flat_features = in_channels
    else:
        flat_features = in_channels * h * w
        steps.append(
            {
                "kind": "flatten",
                "title": "Flatten",
                "shape": f"{flat_features}",
                "details": f"from {in_channels} x {h} x {w}",
                "size": flat_features,
            }
        )

    in_features = flat_features
    for i, hidden_size in enumerate(fc_layers, start=1):
        details = f"{in_features}->{hidden_size}"
        if dropout and dropout > 0:
            details += f", dropout={dropout}"
        steps.append(
            {
                "kind": "fc",
                "title": f"FC {i}",
                "shape": f"{hidden_size}",
                "details": details,
                "size": hidden_size,
            }
        )
        in_features = hidden_size

    steps.append(
        {
            "kind": "classifier",
            "title": "Classifier",
            "shape": f"{num_classes}",
            "details": f"{in_features}->{num_classes}",
            "size": num_classes,
        }
    )
    return steps


def plot_cnn_architecture(config_or_model, figsize=None, title="CNN Architecture",
                          save_path=None, show=True, ax=None):
    """
    Plot a CNNModel architecture from a config dictionary or model instance.

    Args:
        config_or_model: CNNModel, dict, or SimpleNamespace with CNNModel config keys.
        figsize: Optional matplotlib figure size. Defaults to a width based on layer count.
        title: Figure title.
        save_path: Optional path to save the figure.
        show: Whether to call ``plt.show()``.
        ax: Optional matplotlib axis to draw into.

    Returns:
        (fig, ax): matplotlib figure and axis.

    Example:
        config = {
            "input_dim": (3, 32, 32),
            "num_classes": 10,
            "conv_layers": [(32, 3, 1, 1, True), (64, 3, 1, 1, True)],
            "fc_layers": [128],
            "pool_config": ("max", 2, 2),
            "use_gap": False,
            "dropout": 0.5,
        }
        fig, ax = plot_cnn_architecture(config)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    config = _normalize_model_config(config_or_model)
    steps = _cnn_architecture_steps(config)
    if figsize is None:
        figsize = (max(10, 1.65 * len(steps)), 4.8)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = {
        "input": "#d8ecff",
        "conv": "#dff3df",
        "pool": "#fff1c7",
        "gap": "#eee3ff",
        "flatten": "#f2f2f2",
        "fc": "#ffe1d6",
        "classifier": "#ffd6dd",
    }
    edge_color = "#3a3a3a"
    max_size = max(step["size"] for step in steps)

    x_gap = 1.55
    box_width = 1.12
    for i, step in enumerate(steps):
        x = i * x_gap
        height = 0.85 + 1.25 * (step["size"] / max_size) ** 0.35
        y = -height / 2

        patch = FancyBboxPatch(
            (x, y),
            box_width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=edge_color,
            facecolor=colors.get(step["kind"], "#f2f2f2"),
        )
        ax.add_patch(patch)

        ax.text(x + box_width / 2, 0.24, step["title"], ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + box_width / 2, -0.08, step["shape"], ha="center", va="center",
                fontsize=8)
        if step["details"]:
            ax.text(x + box_width / 2, -0.42, step["details"], ha="center", va="center",
                    fontsize=7, color="#444444")

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + x_gap - 0.16, 0),
                xytext=(x + box_width + 0.16, 0),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": edge_color},
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.25, (len(steps) - 1) * x_gap + box_width + 0.25)
    ax.set_ylim(-1.65, 1.65)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax
