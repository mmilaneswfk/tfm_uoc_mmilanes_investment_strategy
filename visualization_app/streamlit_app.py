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
    (valid, valid_calibrated, fi, params, valid_df,
     predicted_proba, predictions, calibrator_predictions,
     target, correlation_data, raw_returns, raw_spy,
     calibrator_predicted_proba, true_labels) = load_data(base_path=parent_folder)
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
    total_returns = predictions.mul(target).sum(axis=0).round(3)
    sector_summary['Total_returns'] = total_returns.round(3)

    total_returns_calibrated = calibrator_predictions.mul(target).sum(axis=0).round(3)
    sector_summary['Total_returns_calibrated'] = total_returns_calibrated.round(3)

    # Create style function to highlight better precision and total returns
    def highlight_better_precision(row):
        result = ['' for col in row.index]
        # Highlight better precision
        if row['Precision'] > row['Precision_calibrated']:
            result[row.index.get_loc('Precision')] = 'background-color: green; color: white'
        elif row['Precision'] < row['Precision_calibrated']:
            result[row.index.get_loc('Precision_calibrated')] = 'background-color: green; color: white'
            
        # Highlight better total returns
        if row['Total_returns'] > row['Total_returns_calibrated']:
            result[row.index.get_loc('Total_returns')] = 'background-color: green; color: white'
        elif row['Total_returns'] < row['Total_returns_calibrated']:
            result[row.index.get_loc('Total_returns_calibrated')] = 'background-color: green; color: white'
            
        return result
            
        return result

    st.dataframe(
        sector_summary.style
        .format({"Precision": "{:.3f}", "Precision_calibrated": "{:.3f}", "Total_returns": "{:.3f}", "Total_returns_calibrated": "{:.3f}"})
        .apply(highlight_better_precision, axis=1),
        height=450  # Set height to accommodate approximately 11 rows
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

    # Selector for precision calculation
    precision_model_choice = st.radio(
        "Usar decisiones del modelo para cálculo de precisión:",
        ["Original", "Calibrado"],
        horizontal=True,
        key="precision_model_choice"  # Unique key
    )
    
    # Choose the appropriate prediction matrix based on the selection
    if precision_model_choice == "Original":
        valid_for_precision = valid[sel_sectors]
        model_label = "Original"
    else:  # Calibrado
        valid_for_precision = valid_calibrated[sel_sectors]
        model_label = "Calibrado"
    
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
    # Sidebar selectors for gain analysis
    model_choice = st.sidebar.radio(
        "Usar probabilidades/decisiones del modelo para análisis de ganancias:",
        ["Original", "Calibrado"],
        key="gain_analysis_model_choice" # Unique key
    )
    return_type_choice = st.sidebar.radio(
        "Tipo de retorno para análisis de ganancias:",
        ["Excess Returns", "Raw Returns"],
        key="return_type_choice" # Unique key
    )

    st.subheader("Análisis de Ganancias Acumuladas")

    # Select the appropriate probability matrix and validation matrix based on model choice
    if model_choice == "Original":
        proba_to_use = predicted_proba  # Use original model probabilities
        valid_to_use = valid            # Use original model decisions for filtering
        model_suffix = "(Original)"
    else: # Calibrated
        try:
            proba_to_use = calibrator_predicted_proba # Use calibrated probabilities
        except NameError:
            st.error("Error: 'calibrator_predicted_proba' no está cargado. Asegúrate de que load_data lo devuelve.")
            st.stop() # Stop execution if calibrated probabilities are missing
        valid_to_use = valid_calibrated # Use calibrated model decisions for filtering
        model_suffix = "(Calibrado)"

    # Select the appropriate return data based on return type choice
    if return_type_choice == "Raw Returns":
        sector_returns_data = raw_returns.loc[valid.index]
        # Use raw SPY returns for the benchmark when analyzing raw returns
        benchmark_returns_data = raw_spy.loc[valid.index]
        return_type_label = "Raw"
        function_to_aggregate = aggregate_lag_returns_for_plot
    else: # Excess Returns (Default/Original behavior)
        sector_returns_data = target.loc[valid.index] # Target usually represents excess returns
        # When analyzing excess returns (target), the benchmark is implicitly the S&P500,
        # so the benchmark's *excess* return is 0.
        benchmark_returns_data = pd.Series(0, index=valid.index) # Zero series for excess benchmark
        # Alternatively, if returns_spy represents the excess return of SPY over a risk-free rate, use that.
        # benchmark_returns_data = returns_spy # Uncomment if returns_spy is excess return over risk-free
        return_type_label = "Excess"
        function_to_aggregate = aggregate_strategies_for_plot


    # Aggregate into a DataFrame for plotting
    cum_returns = function_to_aggregate(
        strategy_results = [
            average_sector_strategy(target=sector_returns_data), # Use selected sector returns
            sp500_benchmark(returns_spy=benchmark_returns_data, reference_index=valid.index), # Use selected benchmark returns
            top_n_periodic_strategy(proba_to_use, sector_returns_data, 1),
            # top_n_periodic_strategy(proba_to_use, sector_returns_data, 1, valid=valid_to_use.abs()),
            top_n_periodic_strategy(proba_to_use, sector_returns_data, 3),
            # top_n_periodic_strategy(proba_to_use, sector_returns_data, 3, valid=valid_to_use.abs())
        ],
        strategy_names = [
            f"Average Sector Strategy ({return_type_label})",
            f"S&P 500 Benchmark ({return_type_label})",
            f"Top 1 Periodic Strategy",
            # f"Top 1 Periodic Strategy (Filtered)",
            f"Top 3 Periodic Strategy",
            # f"Top 3 Periodic Strategy (Filtered)"
        ]
    )

    # Plot with line chart for all strategies

    # Use a radio button to let the user select which line to highlight
    # Ensure unique key if this widget appears elsewhere
    selected_line = st.radio(
        "Resaltar línea:",
        list(cum_returns.columns),
        horizontal=True,
        key="highlight_line_choice_gain_analysis" # More specific key
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
        title=f"Comparación de Retornos Acumulados ({return_type_label}, Usando Modelo {model_choice})", # Updated title
        xaxis_title="Fecha",
        yaxis_title=f"Retorno Acumulado ({return_type_label})",
        legend_title="Estrategia"
    )

    st.plotly_chart(fig_returns, use_container_width=True)

    st.write(f"*Comparación de retornos acumulados ({return_type_label}). Estrategias basadas en probabilidades y/o filtradas usan el modelo {model_choice}.*") # Updated description

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
        predicted_proba_stacked = predicted_proba.stack().sort_index()
        calibrator_predicted_proba_stacked = calibrator_predicted_proba.stack().sort_index()
        
        # Create calibration curves
        prob_true_orig, prob_pred_orig = calibration_curve(true_labels_stacked.values, 
                                                           predicted_proba_stacked.loc[true_labels_stacked.index].values, n_bins=10)
        prob_true_calib, prob_pred_calib = calibration_curve(true_labels_stacked.values, 
                                                             calibrator_predicted_proba_stacked.loc[true_labels_stacked.index].values, n_bins=10)
        
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

    avg_strategy = average_sector_strategy(target=target), # Use selected sector returns
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
        'Top Prediction (Calibrated)': top_prediction_validated_cali
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
        st.subheader("Original vs Calibrated")
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

    # Create 3x4 subplot grid

    # Create a grid of 3 rows and 4 columns (total 12 plots for 11 sectors)
    fig = make_subplots(rows=3, cols=4, subplot_titles=filtered_predicted_proba.columns)

    # Get all sectors
    all_sectors = filtered_predicted_proba.columns

    # Set up colors
    original_color = 'blue'
    calibrated_color = 'green'

    # For each sector, create a KDE plot
    for i, sector in enumerate(all_sectors):
        # Calculate row and column position (1-indexed for plotly)
        row = i // 4 + 1
        col = i % 4 + 1
        
        # Extract data for the current sector
        original_probs = filtered_predicted_proba[sector].values.ravel()
        calibrated_probs = filtered_calibrator_predicted_proba[sector].values.ravel()
        
        # Remove NaN values if any
        original_probs = original_probs[~np.isnan(original_probs)]
        calibrated_probs = calibrated_probs[~np.isnan(calibrated_probs)]
        
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




if __name__ == '__main__':
    main()
