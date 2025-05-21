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
from sklearn.calibration import calibration_curve
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from functions.strategies_to_plot import (
    aggregate_strategies_for_plot,
    aggregate_lag_returns_for_plot,
    sp500_benchmark,
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
    (valid, valid_calibrated, valid_meta, fi, params, valid_df, 
            predicted_proba, predictions, calibrator_predictions, meta_predictions,
            target, correlation_data, raw_returns, raw_spy,
            calibrator_predicted_proba, meta_predicted_proba, true_labels) = load_data(base_path=parent_folder)
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
        'FP_calibrated': (valid_calibrated[sel_sectors] == -1).sum(),
        'TP_meta': (valid_meta[sel_sectors] == 1).sum(),
        'FP_meta': (valid_meta[sel_sectors] == -1).sum()
    })
    
    # Calculate precision metrics
    sector_summary['Precision'] = (sector_summary['TP'] / (sector_summary['TP'] + sector_summary['FP'])).round(3)
    sector_summary['Precision_calibrated'] = (sector_summary['TP_calibrated'] / (sector_summary['TP_calibrated'] + sector_summary['FP_calibrated'])).round(3)
    sector_summary['Precision_meta'] = (sector_summary['TP_meta'] / (sector_summary['TP_meta'] + sector_summary['FP_meta'])).round(3)
    
    # Compute returns for each model
    # First calculate returns for each prediction-target pair
    returns = predictions.mul(raw_returns.loc[valid.index])
    returns_calibrated = calibrator_predictions.mul(raw_returns.loc[valid.index])
    returns_meta = meta_predictions.mul(raw_returns.loc[valid.index])

    # Calculate cumulative returns per sector using the formula: (1 + returns).cumprod() - 1
    cum_returns = (1 + returns).cumprod() - 1
    cum_returns_calibrated = (1 + returns_calibrated).cumprod() - 1
    cum_returns_meta = (1 + returns_meta).cumprod() - 1

    # Get total returns per sector (last value of cumulative returns)
    total_returns = cum_returns.iloc[-1]
    total_returns_calibrated = cum_returns_calibrated.iloc[-1]
    total_returns_meta = cum_returns_meta.iloc[-1]

    # Add returns to summary
    sector_summary['Total_returns'] = total_returns.round(3)
    sector_summary['Total_returns_calibrated'] = total_returns_calibrated.round(3)
    sector_summary['Total_returns_meta'] = total_returns_meta.round(3)
    
    # Reorder columns for better readability
    sector_summary = sector_summary[['TP', 'FP', 'TP_calibrated', 'FP_calibrated', 'TP_meta', 'FP_meta',
                                     'Precision', 'Precision_calibrated','Precision_meta',
                                     'Total_returns', 'Total_returns_calibrated','Total_returns_meta'
                                      ]]
    # Calculate averages for each column
    averages = sector_summary.mean()

    # Create a new row for averages and concatenate it to the dataframe
    average_row = pd.DataFrame(averages).T
    average_row.index = ['Average']
    sector_summary = pd.concat([sector_summary, average_row])
    
    # Create style function to highlight best model across all three
    def highlight_better_precision(row):
        result = ['' for _ in row.index]
        
        # Check if this is the average row and highlight it in gray with 20% alpha
        if row.name == 'Average':
            return ['background-color: rgba(128, 128, 128, 0.2)' for _ in row.index]
            
        # Compare precision across all three models
        precision_values = [row['Precision'], row['Precision_calibrated'], row['Precision_meta']]
        max_precision = max(precision_values)
        if row['Precision'] == max_precision:
            result[row.index.get_loc('Precision')] = 'background-color: green; color: white'
        if row['Precision_calibrated'] == max_precision:
            result[row.index.get_loc('Precision_calibrated')] = 'background-color: green; color: white'
        if row['Precision_meta'] == max_precision:
            result[row.index.get_loc('Precision_meta')] = 'background-color: green; color: white'
        
        # Compare returns across all three models
        returns_values = [row['Total_returns'], row['Total_returns_calibrated'], row['Total_returns_meta']]
        max_returns = max(returns_values)
        if row['Total_returns'] == max_returns:
            result[row.index.get_loc('Total_returns')] = 'background-color: green; color: white'
        if row['Total_returns_calibrated'] == max_returns:
            result[row.index.get_loc('Total_returns_calibrated')] = 'background-color: green; color: white'
        if row['Total_returns_meta'] == max_returns:
            result[row.index.get_loc('Total_returns_meta')] = 'background-color: green; color: white'
        
        return result
    
    # Cast TP and FP columns to int type
    tp_fp_columns = [col for col in sector_summary.columns if 'TP' in col or 'FP' in col]
    sector_summary[tp_fp_columns] = sector_summary[tp_fp_columns].astype(int)

    st.dataframe(
        sector_summary.style
        .format({"Precision": "{:.3f}", "Precision_calibrated": "{:.3f}", "Total_returns": "{:.3f}", "Total_returns_calibrated": "{:.3f}"})
        .apply(highlight_better_precision, axis=1),
        height=450  # Set height to accommodate approximately 11 rows
    )

    # Tabla de validación con selector
    st.subheader("Matriz de Predicciones")
    table_choice = st.radio("Seleccionar datos:", ["Original", "Calibrado", "Meta"], horizontal=True)
    if table_choice == "Original":
        display_df = valid[sel_sectors].astype(int)
    elif table_choice == "Calibrado":
        display_df = valid_calibrated[sel_sectors].astype(int)
    else:  # Meta
        display_df = valid_meta[sel_sectors].astype(int)

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

    # Selector for precision calculation
    precision_model_choice = st.radio(
        "Usar decisiones del modelo para cálculo de precisión:",
        ["Original", "Calibrado", "Meta"],
        horizontal=True,
        key="precision_model_choice"  # Unique key
    )
    
    # Choose the appropriate prediction matrix based on the selection
    if precision_model_choice == "Original":
        valid_for_precision = valid[sel_sectors]
        model_label = "Original"
    elif precision_model_choice == "Calibrado":
        valid_for_precision = valid_calibrated[sel_sectors]
        model_label = "Calibrado"
    else: # Meta
        valid_for_precision = valid_meta[sel_sectors]
        model_label = "Meta"
    
    # Precisión semanal
    weekly = compute_precision(valid_for_precision)
    # Compute rolling 12‐week mean of the weekly precision, skipping nulls
    weekly['Precision_Rolling12'] = (
        weekly['Precision']
        .rolling(window=12, min_periods=1)
        .mean()
    )
    st.subheader(f"Precisión Semanal ({model_label})")

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
        legend_title='Métrica',
        title=f"Precisión Semanal - Modelo {model_label}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"*Tabla de métricas por semana (Modelo {model_label})*")
    st.dataframe(weekly, height=200)

    # Sección 1.5: Análisis de Ganancias
    # Sidebar selector for gain analysis
    model_choice = st.sidebar.radio(
        "Usar probabilidades/decisiones del modelo para análisis de ganancias:",
        ["Original", "Calibrado", "Meta"],
        key="gain_analysis_model_choice" # Unique key
    )
    # Add a checkbox to filter by validated signals
    filter_by_validated = st.sidebar.checkbox("Validated signals only", value=True)
    st.subheader("Análisis de Ganancias Acumuladas")

    # Select the appropriate probability matrix and validation matrix based on model choice
    if model_choice == "Original":
        proba_to_use = predicted_proba  # Use original model probabilities
        valid_to_use = valid           # Use original model decisions for filtering
        model_suffix = "(Original)"
    elif model_choice == "Calibrado":
        try:
            proba_to_use = calibrator_predicted_proba # Use calibrated probabilities
        except NameError:
            st.error("Error: 'calibrator_predicted_proba' no está cargado. Asegúrate de que load_data lo devuelve.")
            st.stop() # Stop execution if calibrated probabilities are missing
        valid_to_use = valid_calibrated # Use calibrated model decisions for filtering
        model_suffix = "(Calibrado)"
    else: # Meta
        try:
            proba_to_use = meta_predicted_proba # Use meta model probabilities
        except NameError:
            st.error("Error: 'meta_predicted_proba' no está cargado. Asegúrate de que load_data lo devuelve.")
            st.stop() # Stop execution if meta probabilities are missing
        valid_to_use = valid_meta # Use meta model decisions for filtering
        model_suffix = "(Meta)"

    # If filter_by_validated is checked, set valid_to_use to None
    if not filter_by_validated:
        valid_to_use = None

    # Set up the data for raw returns analysis
    sector_returns_data = raw_returns.loc[valid.index]
    benchmark_returns_data = raw_spy.loc[valid.index]

    # Aggregate into a DataFrame for plotting
    cum_returns = aggregate_lag_returns_for_plot(
        strategy_results = [
            average_sector_strategy(target_df=sector_returns_data), # Use raw sector returns
            sp500_benchmark(returns_spy=benchmark_returns_data, reference_index=valid.index), # Use raw benchmark returns
            top_n_periodic_strategy(proba_to_use, sector_returns_data, 1, valid=valid_to_use),
            top_n_periodic_strategy(proba_to_use, sector_returns_data, 3, valid=valid_to_use),
        ],
        strategy_names = [
            "Average Sector Strategy (Raw)",
            "S&P 500 Benchmark (Raw)",
            "Top 1 Periodic Strategy",
            "Top 3 Periodic Strategy",
        ]
    )

    # Use a radio button to let the user select which line to highlight
    selected_line = st.radio(
        "Resaltar línea:",
        list(cum_returns.columns),
        horizontal=True,
        key="highlight_line_choice_gain_analysis"
    )

    # Build an interactive Plotly figure
    fig_returns = go.Figure()
    for col in cum_returns.columns:
        # Check if "Filtered" is in the strategy name
        is_filtered = "Filtered" in col
        
        fig_returns.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            mode='lines',
            name=col,
            line=dict(
                width=4 if col == selected_line else 2,
                dash='dash' if is_filtered else None  # Make filtered strategies dashed
            )
        ))

    fig_returns.update_layout(
        title=f"Comparación de Retornos Acumulados (Raw, Usando Modelo {model_choice})",
        xaxis_title="Fecha",
        yaxis_title="Retorno Acumulado (Raw)",
        legend_title="Estrategia"
    )

    st.plotly_chart(fig_returns, use_container_width=True)

    st.write(f"*Comparación de retornos acumulados (Raw). Estrategias basadas en probabilidades usan el modelo {model_choice}.*")

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
                dot_size=5,
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

    # Sección 4: Calibration Plots
    st.header("4. Calibration Plots")
    
    try:
        
        st.subheader("Comparación de calibración entre modelos Original y Calibrado")

        true_labels_stacked = true_labels.stack().sort_index()
        predicted_proba_stacked = predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
        calibrator_predicted_proba_stacked = calibrator_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
        # Add this line - defines meta_predicted_proba_stacked
        meta_predicted_proba_stacked = meta_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
        
        # Create calibration curves
        prob_true_orig, prob_pred_orig = calibration_curve(true_labels_stacked.values, 
                                                           predicted_proba_stacked.values, n_bins=10)
        prob_true_calib, prob_pred_calib = calibration_curve(true_labels_stacked.values, 
                                                             calibrator_predicted_proba_stacked.values, n_bins=10)
        
        # Add meta model calibration curve
        prob_true_meta, prob_pred_meta = calibration_curve(true_labels_stacked.values, 
                                                           meta_predicted_proba_stacked.values, n_bins=10)
        
        # Create perfect calibration line
        perfect_calibration = [0, 1]
        
        # Plot calibration curves
        fig_calibration = go.Figure()
        
        # Add perfect calibration line
        fig_calibration.add_trace(go.Scatter(
            x=perfect_calibration,
            y=perfect_calibration,
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='black', width=1, dash='dash')
        ))
        
        # Add original model calibration curve
        fig_calibration.add_trace(go.Scatter(
            x=prob_pred_orig,
            y=prob_true_orig,
            mode='lines+markers',
            name='Modelo Original',
            line=dict(color='blue', width=2)
        ))
        
        # Add calibrated model calibration curve
        fig_calibration.add_trace(go.Scatter(
            x=prob_pred_calib,
            y=prob_true_calib,
            mode='lines+markers',
            name='Modelo Calibrado',
            line=dict(color='red', width=2)
        ))

        # Add meta model calibration curve:
        fig_calibration.add_trace(go.Scatter(
            x=prob_pred_meta,
            y=prob_true_meta,
            mode='lines+markers',
            name='Modelo Meta',
            line=dict(color='purple', width=2)
        ))
        
        fig_calibration.update_layout(
            xaxis_title='Predicted Probability',
            yaxis_title='Fraction of Positives',
            legend_title='Model',
            title='Calibration Curves Comparison',
            width=800,
            height=500
        )
        
        st.plotly_chart(fig_calibration, use_container_width=True)
        
        st.markdown("""
        **Interpretación:**
        - La línea punteada diagonal representa la calibración perfecta
        - Puntos por encima de la diagonal indican subestimación (underconfidence)
        - Puntos por debajo de la diagonal indican sobreestimación (overconfidence)
        - Un modelo bien calibrado debería seguir la diagonal de cerca
        """)
    except Exception as e:
        st.error(f"Error generando gráficos de calibración: {e}")

    avg_strategy = average_sector_strategy(target_df=target), # Use selected sector returns
    top_prediction_validated = top_n_periodic_strategy(predicted_proba, target, 1, valid=predictions)
    top_prediction_validated_cali = top_n_periodic_strategy(calibrator_predicted_proba, target, 1, valid=calibrator_predictions)

    # Sección 6: Análisis de correlación entre estrategias
    st.header("6. Correlación entre Estrategias")
    
    # Fix avg_strategy (remove tuple structure)
    avg_strategy = avg_strategy[0]  # Extract from tuple
    
    # Create DataFrame with all strategies
    strategy_comparison = pd.DataFrame({
        'Average Strategy': avg_strategy,
        'Top Prediction (Original)': top_prediction_validated,
        'Top Prediction (Calibrated)': top_prediction_validated_cali,
        'Top Prediction (Meta)': top_n_periodic_strategy(meta_predicted_proba, target, 1, valid=meta_predictions)
    })
    
    # Add date range slider
    date_range = pd.to_datetime(strategy_comparison.index)
    min_date = date_range.min().date()
    max_date = date_range.max().date()
    
    start_date, end_date = st.select_slider(
        "Seleccionar rango de fechas:",
        options=sorted(date_range.date),
        value=(min_date, max_date)
    )
    
    # Filter data based on date range
    filtered_data = strategy_comparison.loc[pd.Timestamp(start_date).tz_localize('UTC'):pd.Timestamp(end_date).tz_localize('UTC')]
    
    # Create scatter plot tabs for different comparisons
    # Create columns for side-by-side comparisons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original vs Average")
        fig = px.scatter(
            filtered_data, 
            x='Average Strategy', 
            y='Top Prediction (Original)',
            trendline='ols',
            hover_data=[filtered_data.index.strftime('%Y-%m-%d')]
        )
        # Add 1:1 dashed line
        min_val = min(filtered_data['Average Strategy'].min(), filtered_data['Top Prediction (Original)'].min())
        max_val = max(filtered_data['Average Strategy'].max(), filtered_data['Top Prediction (Original)'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', line=dict(dash='dash', color='gray'),
                                name='1:1 Line'))
        fig.update_layout(
            title="Estrategia Promedio vs Original",
            xaxis_title="Retornos Promedio",
            yaxis_title="Retornos Original",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = filtered_data['Average Strategy'].corr(filtered_data['Top Prediction (Original)'])
        st.metric("Correlación", f"{corr:.4f}")
    
    with col2:
        st.subheader("Calibrated vs Average")
        fig = px.scatter(
            filtered_data, 
            x='Average Strategy', 
            y='Top Prediction (Calibrated)',
            trendline='ols',
            hover_data=[filtered_data.index.strftime('%Y-%m-%d')]
        )
        # Add 1:1 dashed line
        min_val = min(filtered_data['Average Strategy'].min(), filtered_data['Top Prediction (Calibrated)'].min())
        max_val = max(filtered_data['Average Strategy'].max(), filtered_data['Top Prediction (Calibrated)'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', line=dict(dash='dash', color='gray'),
                                name='1:1 Line'))
        fig.update_layout(
            title="Estrategia Promedio vs Calibrado",
            xaxis_title="Retornos Promedio",
            yaxis_title="Retornos Calibrado",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = filtered_data['Average Strategy'].corr(filtered_data['Top Prediction (Calibrated)'])
        st.metric("Correlación", f"{corr:.4f}")
    
    with col3:
        st.subheader("Original vs Calibrado")
        fig = px.scatter(
            filtered_data, 
            x='Top Prediction (Original)', 
            y='Top Prediction (Calibrated)',
            trendline='ols',
            hover_data=[filtered_data.index.strftime('%Y-%m-%d')]
        )
        # Add 1:1 dashed line
        min_val = min(filtered_data['Top Prediction (Original)'].min(), filtered_data['Top Prediction (Calibrated)'].min())
        max_val = max(filtered_data['Top Prediction (Original)'].max(), filtered_data['Top Prediction (Calibrated)'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', line=dict(dash='dash', color='gray'),
                                name='1:1 Line'))
        fig.update_layout(
            title="Original vs Calibrado",
            xaxis_title="Retornos Original",
            yaxis_title="Retornos Calibrado",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = filtered_data['Top Prediction (Original)'].corr(filtered_data['Top Prediction (Calibrated)'])
        st.metric("Correlación", f"{corr:.4f}")

    # Section 7: KDE Plots of Probability Distributions by Sector
    st.header("7. Distribución de Probabilidades por Sector")

    # Get available date range
    date_range = pd.to_datetime(predicted_proba.index)
    min_date = date_range.min().date()
    max_date = date_range.max().date()

    # Create date range slider
    kde_start_date, kde_end_date = st.select_slider(
        "Seleccionar rango de fechas para análisis de distribución:",
        options=sorted(date_range.date),
        value=(min_date, max_date),
        key="kde_date_slider"  # Unique key to avoid conflicts
    )

    # Filter data based on date range
    filtered_predicted_proba = predicted_proba.loc[pd.Timestamp(kde_start_date).tz_localize('UTC'):pd.Timestamp(kde_end_date).tz_localize('UTC')]
    filtered_calibrator_predicted_proba = calibrator_predicted_proba.loc[pd.Timestamp(kde_start_date).tz_localize('UTC'):pd.Timestamp(kde_end_date).tz_localize('UTC')]
    # Add meta model probabilities filtering
    filtered_meta_predicted_proba = meta_predicted_proba.loc[pd.Timestamp(kde_start_date).tz_localize('UTC'):pd.Timestamp(kde_end_date).tz_localize('UTC')]

    # Create 3x4 subplot grid

    # Create a grid of 3 rows and 4 columns (total 12 plots for 11 sectors)
    fig = make_subplots(rows=3, cols=4, subplot_titles=filtered_predicted_proba.columns)

    # Get all sectors
    all_sectors = filtered_predicted_proba.columns

    # Set up colors
    original_color = 'blue'
    calibrated_color = 'red'
    meta_color = 'purple'  # Add a color for the meta model

    # For each sector, create a KDE plot
    for i, sector in enumerate(all_sectors):
        # Calculate row and column position (1-indexed for plotly)
        row = i // 4 + 1
        col = i % 4 + 1
        
        # Extract data for the current sector
        original_probs = filtered_predicted_proba[sector].values.ravel()
        calibrated_probs = filtered_calibrator_predicted_proba[sector].values.ravel()
        meta_probs = filtered_meta_predicted_proba[sector].values.ravel()  # Add meta probabilities
        
        # Remove NaN values if any
        original_probs = original_probs[~np.isnan(original_probs)]
        calibrated_probs = calibrated_probs[~np.isnan(calibrated_probs)]
        meta_probs = meta_probs[~np.isnan(meta_probs)]  # Handle NaN for meta probabilities
        
        # Minimum number of points needed for KDE
        min_points = 5
        
        # Calculate KDE for original probabilities if enough data
        if len(original_probs) >= min_points:
            try:
                kde_original = gaussian_kde(original_probs)
                x_original = np.linspace(0, 1, 1000)
                y_original = kde_original(x_original)
                
                # Add original KDE trace
                fig.add_trace(
                    go.Scatter(
                        x=x_original,
                        y=y_original,
                        fill='tozeroy',
                        fillcolor=f'rgba(0, 0, 255, 0.3)',
                        line=dict(color=original_color),
                        name='Original' if i==0 else None,
                        showlegend=i==0
                    ),
                    row=row, col=col
                )
            except Exception as e:
                st.warning(f"No se pudo calcular KDE para {sector} (original): {e}")
        
        # Calculate KDE for calibrated probabilities if enough data
        if len(calibrated_probs) >= min_points:
            try:
                kde_calibrated = gaussian_kde(calibrated_probs)
                x_calibrated = np.linspace(0, 1, 1000)
                y_calibrated = kde_calibrated(x_calibrated)
                
                # Add calibrated KDE trace
                fig.add_trace(
                    go.Scatter(
                        x=x_calibrated,
                        y=y_calibrated,
                        fill='tozeroy',
                        fillcolor=f'rgba(255, 0, 0, 0.3)',
                        line=dict(color=calibrated_color),
                        name='Calibrado' if i==0 else None,
                        showlegend=i==0
                    ),
                    row=row, col=col
                )
            except Exception as e:
                st.warning(f"No se pudo calcular KDE para {sector} (calibrado): {e}")
        
        # Calculate KDE for meta probabilities if enough data
        if len(meta_probs) >= min_points:
            try:
                kde_meta = gaussian_kde(meta_probs)
                x_meta = np.linspace(0, 1, 1000)
                y_meta = kde_meta(x_meta)
                
                # Add meta KDE trace
                fig.add_trace(
                    go.Scatter(
                        x=x_meta,
                        y=y_meta,
                        fill='tozeroy',
                        fillcolor=f'rgba(128, 0, 128, 0.3)',  # Purple with transparency
                        line=dict(color=meta_color),
                        name='Meta' if i==0 else None,
                        showlegend=i==0
                    ),
                    row=row, col=col
                )
            except Exception as e:
                st.warning(f"No se pudo calcular KDE para {sector} (meta): {e}")
        
        # Update axes for each subplot
        fig.update_xaxes(title_text="Probabilidad", range=[0, 1], row=row, col=col)
        fig.update_yaxes(title_text="Densidad", row=row, col=col)

    # Update the layout
    fig.update_layout(
        title_text="Distribución de Probabilidades por Sector",
        height=800,
        width=1000,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretación:**
    - Los gráficos muestran la distribución de probabilidades predichas para cada sector
    - Azul: Modelo Original
    - Rojo: Modelo Calibrado
    - Un modelo bien calibrado debería mostrar una distribución más cercana a la distribución real de las probabilidades
    - Las áreas superpuestas indican donde ambos modelos asignan probabilidades similares
    """)

    # Section 5: Precision-Recall and ROC Curves
    st.header("5. Precision-Recall and ROC Curves")

    # Prepare data - stack the true labels and predictions
    true_labels_stacked = true_labels.stack().sort_index()
    predicted_proba_stacked = predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
    calibrator_predicted_proba_stacked = calibrator_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
    meta_predicted_proba_stacked = meta_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]

    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precision-Recall Curve")
        
        # Calculate precision-recall points for original model
        precision_orig, recall_orig, _ = precision_recall_curve(true_labels_stacked, predicted_proba_stacked)
        pr_auc_orig = auc(recall_orig, precision_orig)
        
        # Calculate precision-recall points for calibrated model
        precision_calib, recall_calib, _ = precision_recall_curve(true_labels_stacked, calibrator_predicted_proba_stacked)
        pr_auc_calib = auc(recall_calib, precision_calib)
        
        # Calculate precision-recall points for meta model
        precision_meta, recall_meta, _ = precision_recall_curve(true_labels_stacked, meta_predicted_proba_stacked)
        pr_auc_meta = auc(recall_meta, precision_meta)
        
        # Create Precision-Recall curve plot
        fig_pr = go.Figure()
        
        # Add original model PR curve
        fig_pr.add_trace(go.Scatter(
            x=recall_orig, 
            y=precision_orig,
            mode='lines',
            name=f'Original (AUC={pr_auc_orig:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add calibrated model PR curve
        fig_pr.add_trace(go.Scatter(
            x=recall_calib, 
            y=precision_calib,
            mode='lines',
            name=f'Calibrado (AUC={pr_auc_calib:.3f})',
            line=dict(color='red', width=2)
        ))
        
        # Add meta model PR curve
        fig_pr.add_trace(go.Scatter(
            x=recall_meta, 
            y=precision_meta,
            mode='lines',
            name=f'Meta (AUC={pr_auc_meta:.3f})',
            line=dict(color='purple', width=2)
        ))
        
        # Update layout
        fig_pr.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            legend_title='Modelo',
            title='Precision-Recall Curve Comparison'
        )
        
        st.plotly_chart(fig_pr, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        
        # Calculate ROC points for original model
        fpr_orig, tpr_orig, _ = roc_curve(true_labels_stacked, predicted_proba_stacked)
        roc_auc_orig = auc(fpr_orig, tpr_orig)
        
        # Calculate ROC points for calibrated model
        fpr_calib, tpr_calib, _ = roc_curve(true_labels_stacked, calibrator_predicted_proba_stacked)
        roc_auc_calib = auc(fpr_calib, tpr_calib)
        
        # Calculate ROC points for meta model
        fpr_meta, tpr_meta, _ = roc_curve(true_labels_stacked, meta_predicted_proba_stacked)
        roc_auc_meta = auc(fpr_meta, tpr_meta)
        
        # Create ROC curve plot
        fig_roc = go.Figure()
        
        # Add random guess line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='black', width=1, dash='dash')
        ))
        
        # Add original model ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr_orig, 
            y=tpr_orig,
            mode='lines',
            name=f'Original (AUC={roc_auc_orig:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add calibrated model ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr_calib, 
            y=tpr_calib,
            mode='lines',
            name=f'Calibrado (AUC={roc_auc_calib:.3f})',
            line=dict(color='red', width=2)
        ))
        
        # Add meta model ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr_meta, 
            y=tpr_meta,
            mode='lines',
            name=f'Meta (AUC={roc_auc_meta:.3f})',
            line=dict(color='purple', width=2)
        ))
        
        # Update layout
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend_title='Modelo',
            title='ROC Curve Comparison'
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("""
    **Interpretación:**
    - **Precision-Recall Curve**: Muestra el balance entre precisión y exhaustividad. Mayor área bajo la curva (AUC) indica mejor rendimiento.
    - **ROC Curve**: Muestra la relación entre la tasa de verdaderos positivos y falsos positivos. Un AUC cercano a 1 indica un mejor clasificador.
    """)

    # Section 8: Classification Metrics Table
    st.header("8. Métricas de Clasificación")

    # Prepare data for classification metrics
    true_labels_stacked = true_labels.stack().sort_index()
    predicted_proba_stacked = predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
    calibrator_predicted_proba_stacked = calibrator_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]
    meta_predicted_proba_stacked = meta_predicted_proba.stack().sort_index().loc[true_labels_stacked.index]

    # Create binary predictions at threshold 0.5
    predicted_binary = (predicted_proba_stacked >= 0.5).astype(int)
    calibrator_binary = (calibrator_predicted_proba_stacked >= 0.5).astype(int)
    meta_binary = (meta_predicted_proba_stacked >= 0.5).astype(int)

    # Calculate metrics for both models
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, average_precision_score, confusion_matrix
    )

    # Compute the metrics for original model
    accuracy_orig = accuracy_score(true_labels_stacked, predicted_binary)
    precision_orig = precision_score(true_labels_stacked, predicted_binary)
    recall_orig = recall_score(true_labels_stacked, predicted_binary)
    f1_orig = f1_score(true_labels_stacked, predicted_binary)
    roc_auc_orig = roc_auc_score(true_labels_stacked, predicted_proba_stacked)
    avg_precision_orig = average_precision_score(true_labels_stacked, predicted_proba_stacked)
    tn_orig, fp_orig, fn_orig, tp_orig = confusion_matrix(true_labels_stacked, predicted_binary).ravel()

    # Compute the metrics for calibrated model
    accuracy_calib = accuracy_score(true_labels_stacked, calibrator_binary)
    precision_calib = precision_score(true_labels_stacked, calibrator_binary)
    recall_calib = recall_score(true_labels_stacked, calibrator_binary)
    f1_calib = f1_score(true_labels_stacked, calibrator_binary)
    roc_auc_calib = roc_auc_score(true_labels_stacked, calibrator_predicted_proba_stacked)
    avg_precision_calib = average_precision_score(true_labels_stacked, calibrator_predicted_proba_stacked)
    tn_calib, fp_calib, fn_calib, tp_calib = confusion_matrix(true_labels_stacked, calibrator_binary).ravel()

    # Compute the metrics for meta model
    accuracy_meta = accuracy_score(true_labels_stacked, meta_binary)
    precision_meta = precision_score(true_labels_stacked, meta_binary)
    recall_meta = recall_score(true_labels_stacked, meta_binary)
    f1_meta = f1_score(true_labels_stacked, meta_binary)
    roc_auc_meta = roc_auc_score(true_labels_stacked, meta_predicted_proba_stacked)
    avg_precision_meta = average_precision_score(true_labels_stacked, meta_predicted_proba_stacked)
    tn_meta, fp_meta, fn_meta, tp_meta = confusion_matrix(true_labels_stacked, meta_binary).ravel()

    # Create dataframes for metrics
    metrics_df = pd.DataFrame({
        'Métrica': [
            'Accuracy', 'Precision', 'Recall', 'F1 Score', 
            'ROC AUC', 'Average Precision', 
            'True Positives', 'False Positives', 
            'True Negatives', 'False Negatives'
        ],
        'Modelo Original': [
            accuracy_orig, precision_orig, recall_orig, f1_orig, 
            roc_auc_orig, avg_precision_orig,
            tp_orig, fp_orig, tn_orig, fn_orig
        ],
        'Modelo Calibrado': [
            accuracy_calib, precision_calib, recall_calib, f1_calib, 
            roc_auc_calib, avg_precision_calib,
            tp_calib, fp_calib, tn_calib, fn_calib
        ],
        'Modelo Meta': [
            accuracy_meta, precision_meta, recall_meta, f1_meta, 
            roc_auc_meta, avg_precision_meta,
            tp_meta, fp_meta, tn_meta, fn_meta
        ]
    })

    # Function to highlight the better model for each metric
    def highlight_better_model(row):
        # Skip the first column (metric names)
        orig_val = row['Modelo Original']
        calib_val = row['Modelo Calibrado']
        meta_val = row['Modelo Meta']
        
        # For count metrics, don't highlight
        if row['Métrica'] in ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']:
            return [''] * len(row)
        
        # For all other metrics, higher is better
        result = [''] * len(row)  # Initialize with empty strings
        
        # Find the best model
        max_val = max(orig_val, calib_val, meta_val)
        
        # Highlight the best model(s)
        if orig_val == max_val:
            result[1] = 'background-color: green; color: white'
        if calib_val == max_val:
            result[2] = 'background-color: green; color: white'
        if meta_val == max_val:
            result[3] = 'background-color: green; color: white'
            
        return result

    # Apply styling to the metrics dataframe
    styled_metrics = metrics_df.style.apply(highlight_better_model, axis=1)

    # Format numbers for better readability
    styled_metrics = styled_metrics.format({
        'Modelo Original': lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x:,d}",
        'Modelo Calibrado': lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x:,d}",
        'Modelo Meta': lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x:,d}"
    })

    # Display the metrics table
    st.dataframe(styled_metrics, use_container_width=True)

    # Add explanatory text
    st.markdown("""
    **Interpretación de métricas:**
    - **Accuracy**: Proporción de predicciones correctas (tanto positivos como negativos).
    - **Precision**: De todos los casos que el modelo predijo como positivos, qué proporción era realmente positiva.
    - **Recall**: De todos los casos realmente positivos, qué proporción fue correctamente identificada por el modelo.
    - **F1 Score**: Media armónica entre precisión y recall, útil cuando las clases están desequilibradas.
    - **ROC AUC**: Área bajo la curva ROC, mide la habilidad del modelo para distinguir entre clases.
    - **Average Precision**: Resumen de la curva precision-recall, insensible al umbral de decisión.

    **Conteo de predicciones:**
    - **True Positives**: Casos correctamente predichos como positivos.
    - **False Positives**: Casos incorrectamente predichos como positivos (Error Tipo I).
    - **True Negatives**: Casos correctamente predichos como negativos.
    - **False Negatives**: Casos incorrectamente predichos como negativos (Error Tipo II).

    El modelo con mejor desempeño en cada métrica está resaltado en verde.
    """)

    # Section 9: Distribución de Retornos por Estrategia
    st.header("9. Distribución de Retornos por Estrategia")

    # Calculate strategy returns for S&P and Average Sector Strategy
    avg_sector_returns = average_sector_strategy(target_df=sector_returns_data).values
    sp500_returns = sp500_benchmark(returns_spy=benchmark_returns_data, reference_index=valid.index).values

    # Calculate Top 1 and Top 3 for each model (original, calibrated, meta)
    top1_original = top_n_periodic_strategy(predicted_proba, sector_returns_data, 1).values
    top1_calibrated = top_n_periodic_strategy(calibrator_predicted_proba, sector_returns_data, 1).values
    top1_meta = top_n_periodic_strategy(meta_predicted_proba, sector_returns_data, 1).values

    top3_original = top_n_periodic_strategy(predicted_proba, sector_returns_data, 3).values
    top3_calibrated = top_n_periodic_strategy(calibrator_predicted_proba, sector_returns_data, 3).values
    top3_meta = top_n_periodic_strategy(meta_predicted_proba, sector_returns_data, 3).values

    # Calculate global min and max for consistent x-axis limits
    all_returns = np.concatenate([avg_sector_returns, sp500_returns, 
                                 top1_original, top1_calibrated, top1_meta,
                                 top3_original, top3_calibrated, top3_meta])
    global_min = np.min(all_returns)
    global_max = np.max(all_returns)

    # Add some padding (15% on each side to accommodate legends)
    x_range_padding = (global_max - global_min) * 0.15
    x_min = global_min - x_range_padding
    x_max = global_max + x_range_padding

    # Create three subplots - one for each comparison
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=[
        "S&P 500 vs Average Sector Strategy",
        "Top 1 Strategy Comparison",
        "Top 3 Strategy Comparison"
    ])
    # Helper function to add a KDE trace to a subplot and return mean and std
    def add_kde_trace(returns, name, color, row, col, showlegend=True, dash=None):
        try:
            kde = gaussian_kde(returns)
            x_range = np.linspace(x_min, x_max, 1000)
            y_kde = kde(x_range)
            
            # Calculate mean and std
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Find maximum height of the KDE curve for proper scaling
            max_kde_height = np.max(y_kde)
            
            # Create customdata array with mean and std for hover info
            customdata = np.full((len(x_range), 2), [mean_return, std_return])
            
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2, dash=dash),
                    showlegend=showlegend,
                    customdata=customdata,
                    hovertemplate=f"{name}<br>Mean: %{{customdata[0]:.4f}}<br>Std: %{{customdata[1]:.4f}}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Add vertical line for mean that respects the scale of the KDE
            fig_dist.add_shape(
                type="line",
                x0=mean_return, x1=mean_return,
                y0=0, y1=max_kde_height * 0.9,
                xref=f"x{col}", yref=f"y{col}",
                line=dict(
                    color=color, 
                    width=2.5,
                    dash="dash" if dash is None else dash
                ),
                row=row, col=col
            )
            
            return mean_return, std_return
        except Exception as e:
            st.warning(f"No se pudo calcular KDE para {name}: {e}")
            return None, None

    # Plot 1: S&P 500 vs Average Sector Strategy
    mean_avg, std_avg = add_kde_trace(avg_sector_returns, "Average Sector", "blue", 1, 1)
    mean_sp, std_sp = add_kde_trace(sp500_returns, "S&P 500", "green", 1, 1)

    # Plot 2: Top 1 Strategy Comparison
    mean_top1_orig, std_top1_orig = add_kde_trace(top1_original, "Original Top 1", "blue", 1, 2)
    mean_top1_calib, std_top1_calib = add_kde_trace(top1_calibrated, "Calibrated Top 1", "red", 1, 2)
    mean_top1_meta, std_top1_meta = add_kde_trace(top1_meta, "Meta Top 1", "purple", 1, 2)
    # Add average strategy as dotted line for comparison
    mean_avg_plot2, std_avg_plot2 = add_kde_trace(avg_sector_returns, "Average Sector", "black", 1, 2, dash="dot")

    # Plot 3: Top 3 Strategy Comparison
    mean_top3_orig, std_top3_orig = add_kde_trace(top3_original, "Original Top 3", "blue", 1, 3)
    mean_top3_calib, std_top3_calib = add_kde_trace(top3_calibrated, "Calibrated Top 3", "red", 1, 3)
    mean_top3_meta, std_top3_meta = add_kde_trace(top3_meta, "Meta Top 3", "purple", 1, 3)
    # Add average strategy as dotted line for comparison
    mean_avg_plot3, std_avg_plot3 = add_kde_trace(avg_sector_returns, "Average Sector", "black", 1, 3, dash="dot")
    # Helper function to add a KDE trace to a subplot and return mean and std
    def add_kde_trace(returns, name, color, row, col, showlegend=True, dash=None):
        try:
            kde = gaussian_kde(returns)
            x_range = np.linspace(x_min, x_max, 1000)
            y_kde = kde(x_range)
            
            # Calculate mean and std
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Find maximum height of the KDE curve for proper scaling
            max_kde_height = np.max(y_kde)
            
            # Update name with mean value
            name_with_mean = f"{name} (Mean: {mean_return:.4f})"
            
            # Create KDE trace
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode='lines',
                    name=name_with_mean,
                    line=dict(color=color, width=2, dash=dash),
                    showlegend=showlegend
                ),
                row=row, col=col
            )
            
            # Add vertical line for mean that respects the scale of the KDE
            fig_dist.add_shape(
                type="line",
                x0=mean_return, x1=mean_return,
                y0=0, y1=max_kde_height * 0.9,  # Use 90% of max height for visibility
                xref=f"x{col}", yref=f"y{col}",  # Use data coordinates
                line=dict(
                    color=color, 
                    width=2.5,  # Slightly thicker
                    dash="dash" if dash is None else dash
                ),
                row=row, col=col
            )
            
            return mean_return, std_return
        except Exception as e:
            st.warning(f"No se pudo calcular KDE para {name}: {e}")
            return None, None

    # Plot 1: S&P 500 vs Average Sector Strategy
    mean_avg, std_avg = add_kde_trace(avg_sector_returns, "Average Sector", "blue", 1, 1)
    mean_sp, std_sp = add_kde_trace(sp500_returns, "S&P 500", "green", 1, 1)

    # Add annotations for plot 1 - find better positioning
    if mean_avg and mean_sp:
        # Find a good x position (between the means) for the annotation
        x_pos = (mean_avg + mean_sp) / 2
        # Place annotation higher with larger font size
        fig_dist.add_annotation(
            x=x_pos, y=0.85, xref="x1", yref="paper",
            text=f"Average: Mean={mean_avg:.4f}, Std={std_avg:.4f}<br>S&P 500: Mean={mean_sp:.4f}, Std={std_sp:.4f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=2,
            align="left", borderpad=6
        )

    # Plot 2: Top 1 Strategy Comparison
    mean_top1_orig, std_top1_orig = add_kde_trace(top1_original, "Original Top 1", "blue", 1, 2)
    mean_top1_calib, std_top1_calib = add_kde_trace(top1_calibrated, "Calibrated Top 1", "red", 1, 2)
    mean_top1_meta, std_top1_meta = add_kde_trace(top1_meta, "Meta Top 1", "purple", 1, 2)
    # Add average strategy as dotted line for comparison
    mean_avg_plot2, std_avg_plot2 = add_kde_trace(avg_sector_returns, "Average Sector", "black", 1, 2, dash="dot")

    # Add annotations for plot 2 - improved visibility and positioning
    if mean_top1_orig and mean_top1_calib and mean_top1_meta:
        # Find a good x position for annotation (weighted average of means)
        x_pos = (mean_top1_orig + mean_top1_calib + mean_top1_meta) / 3
        
        fig_dist.add_annotation(
            x=x_pos, y=0.85, xref="x2", yref="paper",
            text=f"<b>Model Comparison:</b><br>Original: Mean={mean_top1_orig:.4f}, Std={std_top1_orig:.4f}<br>Calibrated: Mean={mean_top1_calib:.4f}, Std={std_top1_calib:.4f}<br>Meta: Mean={mean_top1_meta:.4f}, Std={std_top1_meta:.4f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, 
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=2,
            align="left", borderpad=6
        )

    # Plot 3: Top 3 Strategy Comparison
    mean_top3_orig, std_top3_orig = add_kde_trace(top3_original, "Original Top 3", "blue", 1, 3)
    mean_top3_calib, std_top3_calib = add_kde_trace(top3_calibrated, "Calibrated Top 3", "red", 1, 3)
    mean_top3_meta, std_top3_meta = add_kde_trace(top3_meta, "Meta Top 3", "purple", 1, 3)
    # Add average strategy as dotted line for comparison
    mean_avg_plot3, std_avg_plot3 = add_kde_trace(avg_sector_returns, "Average Sector", "black", 1, 3, dash="dot")

    # Add annotations for plot 3
    if mean_top3_orig and mean_top3_calib and mean_top3_meta:
        fig_dist.add_annotation(
            x=0.5, y=0.9, xref="x3", yref="y3",
            text=f"Original: Mean={mean_top3_orig:.4f}, Std={std_top3_orig:.4f}<br>Calibrated: Mean={mean_top3_calib:.4f}, Std={std_top3_calib:.4f}<br>Meta: Mean={mean_top3_meta:.4f}, Std={std_top3_meta:.4f}",
            showarrow=False, font=dict(size=10),
            bgcolor="white", bordercolor="black", borderwidth=1
        )

    # Update axes for all subplots with consistent x range
    for i in range(1, 4):
        fig_dist.update_xaxes(title_text="Retorno", range=[x_min, x_max], row=1, col=i)
        fig_dist.update_yaxes(title_text="Densidad" if i == 1 else "", row=1, col=i)

    # Update layout
    fig_dist.update_layout(
        title_text="Distribución de Retornos por Estrategia",
        height=500,  # Increase height to accommodate legend below
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,  # Place legend below the plots
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=150)  # Add more bottom margin for legend
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("""
    **Interpretación:**
    - Estos gráficos muestran la distribución de los retornos para cada estrategia
    - El histograma (azul) muestra la frecuencia de los valores de retorno
    - La curva KDE (roja) muestra la estimación de densidad
    - La media y desviación estándar se muestran en cada gráfico
    - Todos los gráficos tienen la misma escala en el eje X para facilitar la comparación
    - Distribuciones más centradas en valores positivos indican mejores estrategias
    - Mayor dispersión (mayor desviación estándar) indica mayor volatilidad
    """)


if __name__ == '__main__':
    main()
