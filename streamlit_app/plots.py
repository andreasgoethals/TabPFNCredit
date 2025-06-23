import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

def plot_metric_by_model(df, metric, title=None):
    """Barplot of a metric by model, across all splits/runs."""
    if metric not in df.columns or "model" not in df.columns:
        st.warning(f"Cannot plot {metric}: not found in dataframe.")
        return
    group = df.groupby("model")[metric].mean().sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=group.values, y=group.index, ax=ax)
    ax.set_title(title or f"{metric} by Model")
    ax.set_xlabel(metric)
    st.pyplot(fig)

def plot_grouped_bar_by_model(all_dfs, metric, dataset_labels=None):
    """
    Create a grouped barplot comparing metric (e.g. accuracy, training_time)
    for each model across datasets (average across folds).
    """
    if not dataset_labels:
        dataset_labels = list(all_dfs.keys())

    # Build a DataFrame: columns = dataset, rows = models, values = mean metric
    all_models = sorted(set(m for df in all_dfs.values() for m in df['model']))
    bar_data = pd.DataFrame(index=all_models)

    for label, df in all_dfs.items():
        means = df.groupby('model')[metric].mean()
        bar_data[label] = means

    bar_data = bar_data.fillna(0)  # In case some models are missing in a dataset

    fig, ax = plt.subplots(figsize=(min(1.5*len(all_models), 16), 6))
    colors = sns.color_palette("tab10", n_colors=len(bar_data.columns))

    bar_width = 0.8 / len(bar_data.columns)
    indices = np.arange(len(bar_data.index))

    for i, col in enumerate(bar_data.columns):
        ax.bar(indices + i*bar_width, bar_data[col], width=bar_width,
               label=col.split('/')[-2], color=colors[i])  # use just the dataset folder

    ax.set_xticks(indices + bar_width*(len(bar_data.columns)-1)/2)
    ax.set_xticklabels(bar_data.index, rotation=30)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Model (across datasets)")
    ax.legend(title="Dataset", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


def plot_model_boxplot(all_dfs, metric):
    # Concatenate all dfs, add dataset info
    plot_df = pd.concat(
        [df.assign(dataset=label.split('/')[-2]) for label, df in all_dfs.items()],
        ignore_index=True
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x='model', y=metric, hue='dataset', ax=ax)
    ax.set_title(f"Distribution of {metric} by Model (across folds and datasets)")
    st.pyplot(fig)


def plot_model_metric_heatmap(df, metrics, title="Performance Heatmap"):
    pivot = df.groupby("model")[metrics].mean()
    fig, ax = plt.subplots(figsize=(1+1.5*len(metrics), 0.5+0.5*len(pivot)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def plot_training_time_vs_metric(df, metric):
    agg = df.groupby("model").agg({metric: "mean", "training_time": "mean"}).reset_index()
    fig, ax = plt.subplots()
    sns.scatterplot(data=agg, x="training_time", y=metric, hue="model", s=100, palette="tab10", ax=ax)
    ax.set_title(f"{metric} vs Training Time by Model")
    ax.legend(bbox_to_anchor=(1, 1))
    st.pyplot(fig)
