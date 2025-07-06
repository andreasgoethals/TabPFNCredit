import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Utility: identify TabPFN models
def is_tabpfn(model):
    """Detect if a model is TabPFN or variant."""
    return "tabpfn" in model.lower()


def plot_metric_bar_avg_by_model(df, metric, title=""):
    group_cols = ["model"]
    if "tuning" in df.columns:
        group_cols.append("tuning")
    if "imbalance_setting" in df.columns:
        group_cols.append("imbalance_setting")

    mean_df = df[df[metric] != 0].copy()
    mean_df = mean_df.groupby(group_cols)[metric].mean().reset_index()

    # Assign colors: highlight TabPFN models
    model_names = mean_df["model"].tolist()
    colors = ["#FF8000" if is_tabpfn(m) else "#2980b9" for m in model_names]  # Orange for TabPFN, blue otherwise

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(model_names, mean_df[metric], color=colors)
    # Optional: bold labels for TabPFN
    for label in ax.get_xticklabels():
        if is_tabpfn(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#FF8000")

    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Model (mean over folds)")
    # Add legend
    import matplotlib.patches as mpatches
    tabpfn_patch = mpatches.Patch(color="#FF8000", label="TabPFN Family")
    other_patch = mpatches.Patch(color="#2980b9", label="Other Models")
    ax.legend(handles=[tabpfn_patch, other_patch])
    st.pyplot(fig)
    plt.close()


def plot_training_time_bar(df, min_time=60, title=None):
    """
    Bar plot for average training time per model, filtered by minimum time.
    Highlights TabPFN models in orange and bold.
    """
    df_plot = df.groupby("model")["training_time"].mean().reset_index()
    df_plot = df_plot[df_plot["training_time"] >= min_time]
    if df_plot.empty:
        st.info(f"No models with avg training_time >= {min_time} seconds.")
        return

    # Assign colors
    colors = ["#FF8000" if is_tabpfn(m) else "#2980b9" for m in df_plot["model"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_plot["model"], df_plot["training_time"], color=colors)
    ax.set_ylabel("training_time (sec)")
    ax.set_title(title or f"Training Time by Model (only ≥ {min_time}s)")

    # Bold TabPFN model x-labels
    for label in ax.get_xticklabels():
        if is_tabpfn(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#FF8000")

    # Legend
    import matplotlib.patches as mpatches
    tabpfn_patch = mpatches.Patch(color="#FF8000", label="TabPFN Family")
    other_patch = mpatches.Patch(color="#2980b9", label="Other Models")
    ax.legend(handles=[tabpfn_patch, other_patch])

    st.pyplot(fig)
    plt.close()

# ------------------------------------2nd part with meta results single dataset ----------------------------------------


def plot_model_boxplot(all_dfs, metric):
    """
    Boxplot of a metric by model (and optionally by tuning).
    Highlights TabPFN models in orange and bold.
    """
    for label, df in all_dfs.items():
        df = df[df[metric] != 0].copy()
        plt.figure(figsize=(10, 5))
        hue_col = "tuning" if "tuning" in df.columns and df["tuning"].nunique() > 1 else None
        # Only use custom palette if NOT using hue
        if hue_col is None:
            palette = {m: "#FF8000" if is_tabpfn(m) else "#2980b9" for m in df["model"].unique()}
            ax = sns.boxplot(data=df, x="model", y=metric, palette=palette)
        else:
            ax = sns.boxplot(data=df, x="model", y=metric, hue=hue_col)
        plt.title(f"Boxplot of {metric} by Model")
        plt.tight_layout()
        # Bold/Orange x-tick labels for TabPFN
        for ticklabel in ax.get_xticklabels():
            if is_tabpfn(ticklabel.get_text()):
                ticklabel.set_fontweight("bold")
                ticklabel.set_color("#FF8000")
        st.pyplot(plt)
        plt.close()


def plot_model_metric_heatmap(df, metrics):
    """
    Heatmap: Model × Metric table, averaged over splits.
    TabPFN model rows and annotation in orange and bold.
    """
    group_cols = ["model"]
    if "tuning" in df.columns:
        group_cols.append("tuning")
    mean_df = df.replace(0, np.nan).groupby(group_cols)[metrics].mean().reset_index()

    # For heatmap: index=model, columns=metric
    if "tuning" in mean_df.columns and mean_df["tuning"].nunique() > 1:
        for tuning_val in mean_df["tuning"].unique():
            tdf = mean_df[mean_df["tuning"] == tuning_val].set_index("model")[metrics]
            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(tdf, annot=True, cmap="viridis", fmt=".3f", annot_kws={"color": "black"})
            plt.title(f"Model × Metric Heatmap (tuning={tuning_val})")

            # Bold/orange y-labels (models) for TabPFN
            for ytick, model_name in zip(ax.get_yticklabels(), tdf.index):
                if is_tabpfn(model_name):
                    ytick.set_color("#FF8000")
                    ytick.set_fontweight("bold")
            st.pyplot(plt)
            plt.close()
    else:
        tdf = mean_df.set_index("model")[metrics]
        plt.figure(figsize=(max(12, len(metrics)*2), 6))
        ax = sns.heatmap(tdf, annot=True, cmap="viridis", fmt=".3f", annot_kws={"color": "black"})
        plt.title("Model × Metric Heatmap")
        for ytick, model_name in zip(ax.get_yticklabels(), tdf.index):
            if is_tabpfn(model_name):
                ytick.set_color("#FF8000")
                ytick.set_fontweight("bold")
        st.pyplot(plt)
        plt.close()


def plot_training_time_vs_metric(df, metric, min_time=60, min_metric=None):
    """
    Scatterplot of training time vs. selected metric, colored by model (optionally by tuning).
    Optionally filters by min_metric (e.g., accuracy).
    """
    plot_df = df[df['training_time'] >= min_time]
    # FILTER: Ignore zero metric values
    plot_df = plot_df[plot_df[metric] != 0]
    if min_metric is not None:
        plot_df = plot_df[plot_df[metric] >= min_metric]
    if plot_df.empty:
        st.info(f"No data for models with training_time >= {min_time} seconds and {metric} >= {min_metric}.")
        return
    plt.figure(figsize=(8, 5))
    hue_col = "model"
    style_col = "tuning" if "tuning" in df.columns and df["tuning"].nunique() > 1 else None
    ax = sns.scatterplot(data=plot_df, x='training_time', y=metric, hue=hue_col, style=style_col)
    plt.title(f"Training Time vs {metric}")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# ------------------------------------3rd part with meta results all datasets----------------------------------------

def plot_combined_metric_bar(df, metric, title=None):
    # Group by model, mean over everything (all datasets, splits, etc.)
    mean_df = df.replace(0, np.nan).groupby("model")[metric].mean().reset_index()
    palette = {m: "#FF8000" if is_tabpfn(m) else "#2980b9" for m in mean_df["model"]}
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(mean_df["model"], mean_df[metric], color=[palette[m] for m in mean_df["model"]])
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Model (All Datasets)")
    # Bold x-ticks for TabPFN
    for label in ax.get_xticklabels():
        if is_tabpfn(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#FF8000")
    st.pyplot(fig)
    plt.close()

def plot_combined_boxplot(df, metric):
    df = df[df[metric] != 0].copy()
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df, x="model", y=metric, color="#b6d0e2")
    plt.title(f"Boxplot of {metric} by Model (All Datasets)")
    plt.tight_layout()
    # Highlight TabPFN models
    for label in ax.get_xticklabels():
        if is_tabpfn(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#FF8000")
    st.pyplot(plt)
    plt.close()


def plot_combined_metric_heatmap(df, metrics):
    mean_df = df.replace(0, np.nan).groupby("model")[metrics].mean().reset_index()
    plt.figure(figsize=(12, 6))
    hm = sns.heatmap(mean_df.set_index("model"), annot=True, cmap="viridis", fmt=".3f")
    plt.title("Model × Metric Heatmap (All Datasets)")
    # Highlight TabPFN rows
    for ytick in hm.get_yticklabels():
        if is_tabpfn(ytick.get_text()):
            ytick.set_fontweight("bold")
            ytick.set_color("#FF8000")
    st.pyplot(plt)
    plt.close()

# def plot_combined_radar(df, metrics):
#     # Average over all runs for each model
#     mean_df = df.replace(0, np.nan).groupby("model")[metrics].mean()
#     categories = metrics
#     N = len(categories)
#     angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#     angles += angles[:1]
#     fig = plt.figure(figsize=(7, 7))
#     ax = plt.subplot(111, polar=True)
#
#     for model, row in mean_df.iterrows():
#         values = row.tolist()
#         values += values[:1]
#         if is_tabpfn(model):
#             ax.plot(angles, values, label=model, color="#FF8000", linewidth=2.5)
#             ax.fill(angles, values, color="#FF8000", alpha=0.3)
#         else:
#             ax.plot(angles, values, label=model, alpha=0.6)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories)
#     plt.title("Radar Plot: Model Comparison")
#     plt.legend(bbox_to_anchor=(1.1, 1.05))
#     st.pyplot(fig)
#     plt.close()

def plot_combined_radar_interactive(df, metrics):
    mean_df = df.replace(0, np.nan).groupby("model")[metrics].mean()
    categories = metrics + [metrics[0]]  # wrap around for radar

    fig = go.Figure()
    for model, row in mean_df.iterrows():
        values = row.tolist() + [row.tolist()[0]]  # close the loop
        color = "#FF8000" if is_tabpfn(model) else None
        line_width = 4 if is_tabpfn(model) else 2
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            name=model,
            line=dict(color=color, width=line_width),
            fill='toself',
            opacity=0.6 if not is_tabpfn(model) else 0.8
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        legend_title="Model (click to filter)",
        title="Radar Plot: Model Comparison",
        showlegend=True,
        width=700,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)



# The pairwise comparison table answers: “How many times does model A beat model B?”
# for a given metric (e.g., accuracy), across all datasets/folds/configurations in the results.
def plot_pairwise_comparison_table(df, metric):
    # Only use rows where metric > 0
    df = df[df[metric] != 0].copy()
    models = df["model"].unique()
    # Use these keys to join on: you can add more if needed
    join_keys = ["dataset", "split"]
    if "tuning" in df.columns:
        join_keys.append("tuning")
    if "imbalance_setting" in df.columns:
        join_keys.append("imbalance_setting")

    comp = pd.DataFrame(index=models, columns=models)
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                comp.loc[m1, m2] = "-"
            else:
                d1 = df[df["model"] == m1][join_keys + [metric]].rename(columns={metric: "score1"})
                d2 = df[df["model"] == m2][join_keys + [metric]].rename(columns={metric: "score2"})
                merged = pd.merge(d1, d2, on=join_keys, how="inner")
                # Only compare where both have valid values
                wins = (merged["score1"] > merged["score2"]).sum()
                comp.loc[m1, m2] = wins
    st.dataframe(comp)



def plot_relative_rank(df, metric):
    # For each group (dataset/split/imbalance/tuning), rank models by metric
    df = df[df[metric] != 0].copy()
    rank_df = df.groupby(['dataset', 'split']).apply(
        lambda x: x.assign(rank=x[metric].rank(ascending=False, method='min'))
    )
    mean_ranks = rank_df.groupby("model")["rank"].mean().reset_index()
    palette = {m: "#FF8000" if is_tabpfn(m) else "#2980b9" for m in mean_ranks["model"]}
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(mean_ranks["model"], mean_ranks["rank"], color=[palette[m] for m in mean_ranks["model"]])
    ax.set_ylabel("Avg. Rank (lower=better)")
    ax.set_title(f"Relative Rank of {metric} (All Datasets)")
    for label in ax.get_xticklabels():
        if is_tabpfn(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#FF8000")
    st.pyplot(fig)
    plt.close()

