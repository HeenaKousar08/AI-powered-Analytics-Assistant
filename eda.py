import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def get_basic_stats(df):
    """Calculates high-level statistics for the metrics dashboard."""
    return {
        "shape": df.shape,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": len(df.select_dtypes(include=['number']).columns),
        "categorical_cols": len(df.select_dtypes(include=['object']).columns)
    }

def auto_clean_data(df):
    """Intelligent data sanitation using Mean/Mode imputation."""
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)
    
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            # Fill numeric with Mean
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        else:
            # Fill categorical with Mode
            if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")
    return df_clean

def generate_visual(df, chart_type, x_axis, y_axis=None, hue=None):
    """Multi-Engine Plotting Hub optimized for Light Professional UI."""
    
    # Performance sampling for large datasets
    plot_df = df.sample(2000) if len(df) > 5000 else df
    
    # Professional Styling for Matplotlib/Seaborn
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#F8FAFC' # Matches your app BG

    try:
        # --- INTERACTIVE PLOTLY SECTION ---
        if chart_type == "Sunburst":
            path = [x_axis] if hue is None else [x_axis, hue]
            return px.sunburst(plot_df, path=path, values=y_axis if y_axis else None, template="plotly_white")

        elif chart_type == "TreeMap":
            return px.treemap(plot_df, path=[x_axis], values=y_axis if y_axis else None, template="plotly_white")

        elif chart_type == "Pie Chart":
            return px.pie(plot_df, names=x_axis, values=y_axis if y_axis else None, hole=0.3, template="plotly_white")

        elif chart_type == "Interactive Scatter":
            return px.scatter(plot_df, x=x_axis, y=y_axis, color=hue, template="plotly_white")

        # --- STATIC SEABORN SECTION ---
        else:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "Histogram":
                sns.histplot(data=plot_df, x=x_axis, kde=True, hue=hue, ax=ax, palette="viridis")
            
            elif chart_type == "Box Plot":
                sns.boxplot(data=plot_df, x=x_axis, y=y_axis, hue=hue, ax=ax)
            
            elif chart_type == "Violin Plot":
                sns.violinplot(data=plot_df, x=x_axis, y=y_axis, hue=hue, split=True, ax=ax)
            
            elif chart_type == "Density (KDE)":
                sns.kdeplot(data=plot_df, x=x_axis, hue=hue, fill=True, ax=ax)
            
            elif chart_type == "Count Plot":
                sns.countplot(data=plot_df, x=x_axis, hue=hue, ax=ax, palette="magma")

            elif chart_type == "Area Chart":
                if y_axis:
                    plot_df.set_index(x_axis)[y_axis].plot.area(ax=ax, alpha=0.5, color='#4F46E5')

            elif chart_type == "Heatmap (Correlation)":
                num_df = plot_df.select_dtypes(include=['number'])
                if not num_df.empty:
                    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                else: return None

            elif chart_type == "Hexbin":
                if y_axis and plot_df[x_axis].dtype != 'object' and plot_df[y_axis].dtype != 'object':
                    hb = ax.hexbin(plot_df[x_axis], plot_df[y_axis], gridsize=20, cmap='YlGnBu')
                    plt.colorbar(hb, ax=ax, label='Density Count')
                else: return None

            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

    except Exception as e:
        print(f"Visualization Logic Error: {e}")
        return None