from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from TetriumColor.Visualization.PlotStyle import apply_style as shared_apply_style
except Exception:
    shared_apply_style = None


def _scale_plot_fonts(factor: float = 3.0) -> None:
    """Multiply default text sizes (axis, legend, title, ticks) after PlotStyle."""
    keys = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "legend.title_fontsize",
        "figure.titlesize",
    )
    for key in keys:
        if key not in plt.rcParams:
            continue
        v = plt.rcParams[key]
        if isinstance(v, (int, float)):
            plt.rcParams[key] = v * factor


def r_squared(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def main() -> None:
    parser = ArgumentParser(description="Combine gamma_check CSVs and plot linearity.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing gamma_check_<n>.csv files.",
    )
    args = parser.parse_args()

    if shared_apply_style is not None:
        shared_apply_style()
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    _scale_plot_fonts(2.0)

    base_dir = args.input_dir.resolve()
    csv_paths = sorted(
        path
        for path in base_dir.glob("gamma_check_*.csv")
        if path.stem.split("_")[-1].isdigit()
    )

    if not csv_paths:
        raise FileNotFoundError(f"No gamma_check_<n>.csv files found in {base_dir}")

    led_series = {}
    fit_results = []
    led_label_map = {0: "B", 1: "G", 2: "O", 3: "R"}
    led_color_map = {
        0: "#2166ac",
        1: "#1b7837",
        2: "#f08c00",
        3: "#b2182b",
    }

    for csv_path in csv_paths:
        led = int(csv_path.stem.split("_")[-1])
        df = pd.read_csv(csv_path, usecols=["Control", "Power"])
        led_label = led_label_map.get(led, f"LED_{led}")
        led_series[led] = df.set_index("Control")["Power"].rename(led_label)

        endpoint_power = float(df.loc[df["Control"] == 255, "Power"].iloc[0])
        a = endpoint_power / 255.0
        pred = df["Control"] * a
        fit_results.append((led, a, r_squared(df["Power"], pred), df, pred))

    combined_df = pd.concat(
        [led_series[led] for led in sorted(led_series.keys())], axis=1
    ).reset_index()
    combined_out = base_dir / "gamma_check_all_leds.csv"
    combined_df.to_csv(combined_out, index=False)

    fig, ax = plt.subplots(figsize=(4.0, 4.6))
    # Keep a square canvas, but use a non-square plotting region with minimal top padding.
    ax.set_position([0.16, 0.30, 0.78, 0.66])
    for led, a, r2, df, pred in sorted(fit_results, key=lambda x: x[0]):
        led_label = led_label_map.get(led, f"LED_{led}")
        color = led_color_map.get(led, None)
        ax.scatter(
            df["Control"],
            df["Power"],
            s=4,
            alpha=0.22,
            color=color,
        )
        ax.plot(
            df["Control"],
            pred,
            linewidth=0.6,
            color=color,
            label=rf"{led_label}: y={a:.3f}x, $R^2$={r2:.1f}",
        )

    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xlim(0, 255)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Control")
    ax.set_ylabel(r"Power ($\mu$W)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.23),
        ncol=2,
        fontsize=12,
        frameon=False,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )

    plot_out = base_dir / "gamma_check_all_leds_fit.png"
    fig.savefig(plot_out, dpi=600)
    plt.close(fig)

    print(f"Saved combined CSV: {combined_out}")
    print(f"Saved plot: {plot_out}")
    for led, a, r2, _, _ in sorted(fit_results, key=lambda x: x[0]):
        print(f"LED {led}: a={a:.8f}, R^2={r2:.8f}")


if __name__ == "__main__":
    main()
