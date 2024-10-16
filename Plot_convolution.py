import matplotlib.pyplot as plt


def format_parameters(inp_parameters, num_columns=4):
    max_len = max(len(param) for param in inp_parameters) + 2  # +2 for padding
    if max_len > 30:
        num_columns = 2

    parameters = inp_parameters + [
        ""
        for _ in range(
            num_columns - ((len(inp_parameters) % num_columns) or num_columns)
        )
    ]

    rows = (len(parameters) + num_columns - 1) // num_columns  # ceiling division
    table_str = ""

    for r in range(rows):
        row_params = parameters[r::rows]  # Get every nth element starting from r
        row_str = " | ".join(param.ljust(max_len) for param in row_params)
        table_str += f" {row_str} \n"

    border_len = len(table_str.split("\n")[0]) - 1
    table_str = table_str

    return table_str


def simple_plot(
    data,
    labels,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
    num_columns=4,
):
    fig, axd = plt.subplot_mosaic(
        (
            """
            AAAA
            AAAA
            XBDZ
            XCEZ
            IIII

            """
            if len(parameters)
            else """
            AAAA
            AAAA
            XBDZ
            XCEZ
            """
        ),
        layout="constrained",
        figsize=(
            4 * scaling_factor,
            (3 + (0.5 if len(parameters) else 0)) * scaling_factor,
        ),
    )

    fig.suptitle(
        title or "Plot",
        fontweight="bold",
        fontsize=(label_font_size + 4) * scaling_factor,
    )

    axd["Z"].axis("off")
    axd["X"].axis("off")
    if len(parameters):
        axd["I"].axis("off")

        params_text = format_parameters(parameters, num_columns)
        axd["I"].text(
            0.5,
            0.5,
            params_text,
            fontsize=(label_font_size - 4) * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axd["I"].transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )
    for i, c in enumerate("ABCDE"):
        axd[c].imshow(data[i], cmap="gray")
        axd[c].set_title(labels[i], fontsize=(label_font_size) * scaling_factor)
        axd[c].axis("off")  # Remove axis

    fig.tight_layout()

    # Show the plot
    fig.show()


def simple_plot3d(
    data,
    labels,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
    num_columns=4,
):
    fig, axd = plt.subplot_mosaic(
        (
            """
            AAAA
            AAAA
            XBDZ
            XCEZ
            IIII
            """
            if len(parameters)
            else """
            AAAA
            AAAA
            XBDZ
            XCEZ

            """
        ),
        layout="constrained",
        figsize=(
            4 * scaling_factor,
            (3 + (0.5 if len(parameters) else 0)) * scaling_factor,
        ),
    )

    fig.suptitle(
        title or "Plot",
        fontweight="bold",
        fontsize=(label_font_size + 4) * scaling_factor,
    )

    axd["Z"].axis("off")
    axd["X"].axis("off")
    if len(parameters):
        axd["I"].axis("off")

        params_text = format_parameters(parameters, num_columns)
        axd["I"].text(
            0.5,
            0.5,
            params_text,
            fontsize=(label_font_size - 4) * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axd["I"].transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    for i, c in enumerate("ABCDE"):
        axd[c].imshow(data[i])
        axd[c].set_title(labels[i], fontsize=(label_font_size) * scaling_factor)
        axd[c].axis("off")  # Remove axis

    fig.tight_layout()

    fig.show()


def batch_plot(
    data, title="", scaling_factor=3, label_font_size=7, parameters=[], num_columns=4
):
    fig, axd = plt.subplot_mosaic(
        (
            """
        AAAAAA
        AAAAAA
        BDFHJL
        BDFHJL
        CEGIKM
        CEGIKM
        OOOOOO
        """
            if len(parameters)
            else """
        AAAAAA
        AAAAAA
        BDFHJL
        BDFHJL
        CEGIKM
        CEGIKM
        """
        ),
        layout="constrained",
        figsize=(
            6 * scaling_factor,
            (2 + (0.5 if len(parameters) else 0)) * scaling_factor,
        ),
    )

    fig.suptitle(
        title or "Plot",
        fontweight="bold",
        fontsize=(label_font_size + 4) * scaling_factor,
    )

    if len(parameters):
        axd["O"].axis("off")

        params_text = format_parameters(parameters, num_columns)
        axd["O"].text(
            0.5,
            0.5,
            params_text,
            fontsize=(label_font_size - 4) * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axd["O"].transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    for key in data:
        axd[key].imshow(data[key]["data"], cmap="gray")
        axd[key].set_title(
            data[key]["title"], fontsize=(label_font_size) * scaling_factor
        )
        axd[key].axis("off")
    fig.tight_layout()

    fig.show()
