import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, recall_score
from functions.data_preparation import load_data, load_model, compute_precision
import os
from functions.strategies_to_plot import (
    aggregate_strategies_for_plot,
    buy_model_signals, constant_investment_model,
    buy_all_sectors, top_sector_with_model, sp500_benchmark,
    average_sector_strategy, top_n_periodic_strategy)
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------
# App Streamlit
# ------------------------------------

# Determine the folder where this script is located
current_folder = os.path.dirname(os.path.abspath(__file__))
# Determine the parent folder of this script’s directory
parent_folder = os.path.abspath(os.path.join(current_folder, os.pardir))

def main():
    st.set_page_config(layout="wide", page_title="Validación y SHAP - LGBM Model")
    st.title("Dashboard de Validación Semanal y SHAP para Modelo LGBM")

    # Carga de datos y modelo
    (valid, valid_calibrated, fi, params, valid_df,
     top_sector_analysis, total_gains, predicted_proba,
     returns_spy, target, correlation_data) = load_data(base_path=parent_folder)
    model = load_model(base_path=parent_folder)

    # Sidebar: selección de sectores y opciones
    st.sidebar.header("Configuración")
    sectors = valid.columns.tolist()
    sel_sectors = st.sidebar.multiselect("Seleccionar sectores", sectors, default=sectors)
    show_shap = st.sidebar.checkbox("Mostrar interpretación SHAP", value=True)
    sample_frac = st.sidebar.slider("Fracción de muestra para SHAP", 0.01, 1.0, 0.5)


    # Resumen general de predicciones
    total_predictions = (valid != 0).sum().sum()
    true_positives = (valid == 1).sum().sum()
    false_positives = (valid == -1).sum().sum()
    precision = true_positives / (true_positives + false_positives)

    # Calcula métricas adicionales usando predicción agregada por sector:
    # Si alguna semana predice 1 para un sector, consideramos ese sector como positivo
    pred = valid.eq(1).any(axis=0).astype(int)

    # Número de semanas en el dataframe valid (asumiendo que cada fila corresponde a una semana)
    num_weeks = valid.shape[0]

    st.subheader("Resumen General de Predicciones")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Predicciones", total_predictions)
    col2.metric("Verdaderos Positivos", true_positives)
    col3.metric("Falsos Positivos", false_positives)
    col4.metric("Precisión", f"{precision:.2%}")
    col5.metric("Semanas", num_weeks)

    # Resumen por sector
    st.subheader("Resumen por Sector")
    sector_summary = pd.DataFrame({
        'TP': (valid[sel_sectors] == 1).sum(),
        'FP': (valid[sel_sectors] == -1).sum(),
        'TP_calibrated': (valid_calibrated[sel_sectors] == 1).sum(),
        'FP_calibrated': (valid_calibrated[sel_sectors] == -1).sum()
    })
    sector_summary['Precision'] = (sector_summary['TP'] / (sector_summary['TP'] + sector_summary['FP'])).round(3)
    sector_summary['Precision_calibrated'] = (sector_summary['TP_calibrated'] / (sector_summary['TP_calibrated'] + sector_summary['FP_calibrated'])).round(3)

    # Compute Total_returns per sector by multiplying the absolute values of valid with target and summing over rows
    total_returns = valid.abs().mul(target).sum(axis=0).round(3)
    sector_summary['Total_returns'] = total_returns

    # Create style function to highlight better precision
    def highlight_better_precision(row):
        if row['Precision'] > row['Precision_calibrated']:
            return ['background-color: green' if col == 'Precision' else '' for col in row.index]
        elif row['Precision'] < row['Precision_calibrated']:
            return ['background-color: green' if col == 'Precision_calibrated' else '' for col in row.index]
        else:
            return ['' for col in row.index]

    st.dataframe(
        sector_summary.style
        .format({"Precision": "{:.3f}", "Precision_calibrated": "{:.3f}", "Total_returns": "{:.3f}"})
        .apply(highlight_better_precision, axis=1)
    )

    # Tabla de validación con selector
    st.subheader("Matriz de Predicciones")
    table_choice = st.radio("Seleccionar datos:", ["Original", "Calibrado"], horizontal=True)
    display_df = valid[sel_sectors].astype(int) if table_choice == "Original" else valid_calibrated[sel_sectors].astype(int)

    # Create diff dataframe to identify changes
    diff_mask = valid[sel_sectors] != valid_calibrated[sel_sectors]

    # Style function to highlight differences
    def highlight_diff(val, i, c):
        if diff_mask.loc[i, c]:
            return 'background-color: yellow'
        return ''

    # Apply styling
    styled_df = display_df.style.apply(lambda x: [highlight_diff(v, x.name, x.index[i]) for i, v in enumerate(x)], axis=1)
    st.dataframe(styled_df, height=200)  # Height for ~5 rows

    # Precisión semanal
    weekly = compute_precision(valid[sel_sectors])
    # Compute rolling 12‐week mean of the weekly precision, skipping nulls
    weekly['Precision_Rolling12'] = (
        weekly['Precision']
        .rolling(window=12, min_periods=1)
        .mean()
    )
    st.subheader("Precisión Semanal")


    # Recompute rolling 24‑week mean
    weekly['Precision_Rolling24'] = (
        weekly['Precision']
        .rolling(window=24, min_periods=1)
        .mean()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly.index,
        y=weekly['Precision'],
        mode='lines',
        name='Precision 1w',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=weekly.index,
        y=weekly['Precision_Rolling24'],
        mode='lines',
        name='Precision_Rolling 6mo',
        line=dict(color='red', width=4)
    ))
    fig.update_layout(
        xaxis_title='Semana',
        yaxis_title='Precisión',
        legend_title='Métrica'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("*Tabla de métricas por semana*")
    st.dataframe(weekly, height=200)

    # Sección 1.5: Análisis de Ganancias
    st.subheader("Análisis de Ganancias Acumuladas")

    # # Create cumulative returns with rounded values
    # cum_returns = pd.DataFrame({
    #     'Buy model signals': total_gains.cumsum().round(3),
    #     'Model with constant investment': valid.abs().mul(target).dropna().apply(lambda x: x.sum()/np.count_nonzero(x) if np.count_nonzero(x) > 0 else 0, axis = 1).fillna(0).cumsum().round(3),
    #     'Buy all sectors': potential_gains.cumsum().round(3),
    #     'Top 1 sector with model': top_sector_analysis['actual_return'].cumsum().round(3),
    #     'S&P 500': returns_spy.loc[total_gains.index].cumsum().round(3)
    # })

# Aggregate into a DataFrame for plotting
    cum_returns = aggregate_strategies_for_plot(
        strategy_results = [
            buy_model_signals(total_gains=total_gains),
            # constant_investment_model(valid=valid, target=target),
            # top_sector_with_model(top_sector_analysis=top_sector_analysis),
            sp500_benchmark(returns_spy=returns_spy, reference_index=valid.index),
            average_sector_strategy(target=target),
            top_n_periodic_strategy(predicted_proba, target, 1),
            top_n_periodic_strategy(predicted_proba, target, 3),
            top_n_periodic_strategy(predicted_proba, target, 1, valid=valid.abs()),
            top_n_periodic_strategy(predicted_proba, target, 3, valid=valid.abs())
        ]
            ,
        strategy_names = [
            "Buy Model Signals",
            # "Constant Investment Model",
            # "Top Sector with Model",
            "S&P 500 Benchmark",
            "Average Sector Strategy",
            "Top 1 Periodic Strategy",
            "Top 3 Periodic Strategy",
            "Top 1 Periodic Strategy (Valid)",
            "Top 3 Periodic Strategy (Valid)"
        ])

    # Plot with line chart for all strategies


    # Use a radio button to let the user select which line to highlight
    selected_line = st.radio("Resaltar línea:", list(cum_returns.columns), horizontal=True)

    # Build an interactive Plotly figure
    fig = go.Figure()
    for col in cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            mode='lines',
            name=col,
            line=dict(width=4 if col == selected_line else 2)
        ))

    fig.update_layout(
        title="Comparación de Retornos Acumulados",
        xaxis_title="Fecha",
        yaxis_title="Retorno Acumulado",
        legend_title="Estrategia"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("*Comparación de retornos acumulados entre diferentes estrategias*")

    # Sección 2: Feature importances
    st.header("2. Importancia de Variables")
    # Convert importances to percentages and sort
    fi_pct = fi.assign(
        importance=lambda df: df['importance'] / df['importance'].sum() * 100
    )
    fi_sorted = fi_pct.sort_values('importance', ascending=False)
    top20 = fi_sorted.head(20)


    fig_importance = px.bar(
        top20,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Importancias de Variables",
        labels={"importance": "Importancia (%)", "feature": "Variable"},
        text=top20['importance'].round(2).astype(str) + '%'
    )
    # show value labels to the right of each bar
    fig_importance.update_traces(textposition="outside")
    fig_importance.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=120, r=40, t=50, b=20)
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("**Hiperparámetros del modelo:**")
    st.dataframe(params, height=200)  # Shows 10 rows with scrollbar

    # Sección 3: Interpretación SHAP
    if show_shap:
        st.header("3. Interpretación con SHAP")
        try:
            # Get raw features from lgb.Dataset
            X = valid_df.data

            # Sample data for performance using pandas
            sample_X = pd.DataFrame(X).sample(frac=sample_frac, random_state=42)
            sample_X = sample_X.values


            # Use LightGBM's built-in SHAP method
            shap_values = model.predict(sample_X, pred_contrib=True)
            # Reshape if needed (LightGBM adds an extra column for bias)
            shap_values = shap_values[:, :-1]

            # For binary classification, shap_values is a list with 2 elements
            # We take index 1 for the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]


            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            # Adjust the size of the current figure to a smaller dimension
            shap.plots.beeswarm(shap.Explanation(
                values=shap_values,
                data=sample_X,
                feature_names=X.columns.tolist()
            ), show=False, max_display=15, plot_size = (15, 6))
            # plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

            # SHAP Scatter Plot Explanation using modern Streamlit practices
            st.subheader("SHAP Scatter Plot Explanation")
            feature_to_plot = st.selectbox("Select feature for scatter plot:", X.columns.tolist())

            # Create a SHAP Explanation object for all features
            expl = shap.Explanation(
                values=shap_values,
                data=sample_X,
                feature_names=X.columns.tolist()
            )

            # Plot scatter plot for the selected feature with 20% left/right padding
            plt.figure(figsize=(8, 6))
            shap.plots.scatter(
                expl[:, feature_to_plot],
                color=expl.data[:, expl.feature_names.index(feature_to_plot)],
                show=False,
                dot_size=10,
            )
            # Add 20% padding on left and right
            plt.subplots_adjust(top=0.8, bottom=0.2)
            st.pyplot(plt.gcf())
            plt.close()

        except Exception as e:
            st.error(f"Error generando SHAP: {e}")

    st.markdown("---")
    st.write("Powered by Streamlit. Modelo LGBM con interpretabilidad SHAP.")

    # Correlation Heatmap
    st.subheader("Mapa de Calor de Correlaciones")
    corr_data = correlation_data.round(2)
    plt.figure(figsize=(18, 12))
    ax = sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Heatmap de Correlaciones")
    st.pyplot(plt.gcf())
    plt.close()

if __name__ == '__main__':
    main()
