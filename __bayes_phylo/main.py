from __future__ import annotations

from dataclasses import asdict

from flask import Flask, render_template, request

from bayes_phylo import SamplerConfig, parse_sequences, run_mcmc
from bayes_phylo.visualization import render_newick_svg

app = Flask(__name__)

DEFAULT_SEQUENCES = """
>Human
ACGTACGTACGT
>Chimp
ACGTACGTACGA
>Gorilla
ACGTACGTTCGA
>Orangutan
ACGTTCGTTCGA
"""

DEFAULT_CONFIG = SamplerConfig()


@app.route("/", methods=["GET", "POST"])
def index():
    sequences_text = request.form.get("sequences", DEFAULT_SEQUENCES)
    config_values = _read_form_with_defaults(request.form)

    result = None
    map_tree_svg = None
    topology_visual_rows: list[dict[str, object]] = []
    error = None

    if request.method == "POST":
        try:
            data = parse_sequences(sequences_text)
            config = SamplerConfig(
                iterations=int(config_values["iterations"]),
                burn_in=int(config_values["burn_in"]),
                thinning=int(config_values["thinning"]),
                seed=_optional_int(config_values["seed"]),
                init_branch_mean=float(config_values["init_branch_mean"]),
                branch_prior_rate=float(config_values["branch_prior_rate"]),
                mu_prior_rate=float(config_values["mu_prior_rate"]),
                topology_move_prob=float(config_values["topology_move_prob"]),
                branch_move_prob=float(config_values["branch_move_prob"]),
                branch_log_step=float(config_values["branch_log_step"]),
                mu_log_step=float(config_values["mu_log_step"]),
            )
            result = run_mcmc(data, config)
            try:
                map_tree_svg = render_newick_svg(result.map_newick, width=860)
            except Exception:
                map_tree_svg = None

            for row in result.topologies:
                row_svg = None
                row_svg_error = None
                try:
                    row_svg = render_newick_svg(row.representative_newick, width=780)
                except Exception as exc:
                    row_svg_error = str(exc)
                topology_visual_rows.append(
                    {"row": row, "svg": row_svg, "svg_error": row_svg_error}
                )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        sequences=sequences_text,
        config=config_values,
        result=result,
        map_tree_svg=map_tree_svg,
        topology_visual_rows=topology_visual_rows,
        error=error,
    )


def _read_form_with_defaults(form) -> dict[str, str]:
    defaults = asdict(DEFAULT_CONFIG)
    out: dict[str, str] = {}
    for key, value in defaults.items():
        out[key] = form.get(key, "" if value is None else str(value))
    return out


def _optional_int(raw: str) -> int | None:
    stripped = raw.strip()
    if not stripped:
        return None
    return int(stripped)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False)
