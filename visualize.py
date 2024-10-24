import colorsys
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from configs.parser import parse_args

if __name__ == "__main__":
    args = parse_args()
    config_path = args.cfg

    def generate_colors(n):
        HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
        return [
            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b in RGB_tuples
        ]

    # Load the data from the JSON file
    with open(config_path, "r") as f:
        data = json.load(f)

    # Define metrics to plot
    metrics = [
        ("R2 Score", "r2_score"),
        ("MSE", "mse"),
        ("R2 Score Yield Strength", "r2_score_Yield strength"),
        ("R2 Score Ultimate Tensile Strength", "r2_score_Ultimate tensile strength"),
        ("MSE Yield Strength", "mse_Yield strength"),
        ("MSE Ultimate Tensile Strength", "mse_Ultimate tensile strength"),
        ("MSE Quality", "mse_Quality"),
        ("R2 Score Quality", "r2_score_Quality"),
    ]

    # Create a 3x2 subplot figure
    fig = make_subplots(rows=4, cols=2, subplot_titles=[m[0] for m in metrics])

    # Generate colors for each unique model
    unique_models = list(set(model["model"] for model in data))
    colors = {
        model: color
        for model, color in zip(unique_models, generate_colors(len(unique_models)))
    }

    # Create scatter plots for each metric
    for i, (metric_name, metric_key) in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        for j, model in enumerate(data):
            if f"{metric_key}_train" not in model:
                continue
            
            model_name = model["model"]
            color = colors[model_name]

            hover_text = f"""
    Model: {model_name}
    Model args: {json.dumps(model.get('model_args', {}), indent=8)}
    Test size: {model.get('test_size', 'N/A')}
    Method: {model.get('method', 'N/A')}
    Method args: {json.dumps(model.get('method_args', {}), indent=8)}
    Data args: {json.dumps(model.get('data_args', {}), indent=8)}
            """.strip().replace("\n", "<br>")

            print(j, model[f"{metric_key}_train"], metric_name, metric_key)

            # Train score
            fig.add_trace(
                go.Scatter(
                    x=[j],
                    y=[model[f"{metric_key}_train"]],
                    mode="markers",
                    name=f"{model_name} (Train)",
                    marker=dict(color=color, size=12, symbol="circle", opacity=0.7),
                    showlegend=i == 0,  # Only show in legend for the first subplot
                    hovertext=[
                        f"{hover_text}<br>Score (Train): {model[f'{metric_key}_train']:.4f}"
                    ],
                    hoverinfo="text",
                ),
                row=row,
                col=col,
            )

            # Test score
            fig.add_trace(
                go.Scatter(
                    x=[j],
                    y=[model[f"{metric_key}_test"]],
                    mode="markers",
                    name=f"{model_name} (Test)",
                    marker=dict(color=color, size=12, symbol="circle"),
                    showlegend=i == 0,  # Only show in legend for the first subplot
                    hovertext=[
                        f"{hover_text}<br>Score (Test): {model[f'{metric_key}_test']:.4f}"
                    ],
                    hoverinfo="text",
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1200,
        title_text="Model Performance Comparison",
        legend_title_text="Models",
    )

    # Update axes
    for i in range(1, 7):
        fig.update_xaxes(
            showticklabels=False, row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )  # Hide x-axis labels
        fig.update_yaxes(
            rangemode="tozero", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )  # Start y-axis from 0

    # Show the plot
    fig.show()
