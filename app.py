import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.covariance import LedoitWolf
from pandas_datareader import data as pdr
from scipy.stats import norm, sem, t

# Configuración inicial
st.set_page_config(
    layout="wide",
    page_title="Optimizador de Carteras IBEX35",
    page_icon="📈"
)
np.random.seed(78)
random.seed(43)
tf.random.set_seed(78)

# Lista de tickers del IBEX35
tickers = [
    "ANA.MC", "ACX.MC", "AENA.MC", "ANE.MC", "AMS.MC", "BBVA.MC", "CABK.MC", "ELE.MC", "FER.MC",
    "FDR.MC", "GRF.MC", "IBE.MC", "IDR.MC", "ITX.MC", "MAP.MC", "MRL.MC", "MEL.MC", "MTS.MC", "NTGY.MC",
    "REP.MC", "SAN.MC", "TEF.MC"
]

# Función para descargar y procesar datos
def load_data():
    end_date = datetime(2025, 7, 7)
    start_date = end_date - timedelta(days=3*365)
    test_start_date = end_date - timedelta(days=365)
    
    data = yf.download(tickers, start=start_date, end=end_date, interval='1wk')['Close']
    
    train_data = data.loc[data.index < test_start_date]
    test_data = data.loc[data.index >= test_start_date]
    
    train_weekly_returns = np.log(train_data / train_data.shift(1)).dropna()
    test_weekly_returns = np.log(test_data / test_data.shift(1)).dropna()
    
    try:
        bond_data = pdr.get_data_fred('IRLTLT01ESM156N', start=start_date, end=end_date)
        if bond_data.empty or bond_data.isna().all().all():
            raise ValueError("Datos de bono vacíos o todos son NaN")
        risk_free_rate = bond_data.mean().item() / 100
        current_bond_yield = bond_data.iloc[-1].item() / 100
        if np.isnan(risk_free_rate) or np.isnan(current_bond_yield):
            raise ValueError("Datos de bono contienen valores NaN")
    except Exception as e:
        st.warning(f"Error al obtener la tasa libre de riesgo: {e}. Usando 3.23% por defecto.")
        risk_free_rate = 0.0323
        current_bond_yield = 0.0323
    
    return train_data, test_data, train_weekly_returns, test_weekly_returns, data, risk_free_rate, current_bond_yield

# Función para métricas de cartera
def portfolio_metrics(weights, returns_data, risk_free_rate_user):
    retornos_esperados = returns_data.mean() * 52
    lw = LedoitWolf()
    lw.fit(returns_data)
    matriz_covarianza = lw.covariance_ * 52
    
    portfolio_return = np.sum(retornos_esperados * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(matriz_covarianza, weights)))
    
    portfolio_weekly_returns = returns_data @ weights
    sharpe_ratio = (portfolio_return - risk_free_rate_user) / portfolio_volatility if portfolio_volatility > 1e-10 else -1e10
    
    # Calcular VaR y CVaR al 95%
    confidence_level = 0.95
    portfolio_mean = portfolio_weekly_returns.mean()
    portfolio_std = portfolio_weekly_returns.std()
    var_95 = norm.ppf(1 - confidence_level, portfolio_mean, portfolio_std) * np.sqrt(52)  # Anualizado
    cvar_95 = portfolio_mean - portfolio_std * norm.pdf(norm.ppf(confidence_level)) / (1 - confidence_level) * np.sqrt(52)  # Anualizado
    
    # Calcular Sortino Ratio
    downside_returns = portfolio_weekly_returns[portfolio_weekly_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else portfolio_volatility
    sortino_ratio = (portfolio_return - risk_free_rate_user) / downside_std if downside_std > 1e-10 else -1e10
    
    return portfolio_return, portfolio_volatility, sharpe_ratio, var_95, cvar_95, sortino_ratio

# --- Algoritmo NSGA-II ---
def nsga_ii_optimization(train_weekly_returns, risk_free_rate_user, tamano_poblacion=100, num_generaciones=50):
    retornos_esperados = train_weekly_returns.mean() * 52
    lw = LedoitWolf()
    lw.fit(train_weekly_returns)
    matriz_covarianza = lw.covariance_ * 52
    
    def calcular_objetivos(pesos):
        retorno = np.sum(retornos_esperados * pesos)
        volatilidad = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
        sharpe = (retorno - risk_free_rate_user) / volatilidad if volatilidad > 1e-10 else -1e10
        return volatilidad, sharpe

    def domina(obj1, obj2):
        riesgo1, sharpe1 = obj1
        riesgo2, sharpe2 = obj2
        return (riesgo1 <= riesgo2 and sharpe1 >= sharpe2) and (riesgo1 < riesgo2 or sharpe1 > sharpe2)

    def inicializar_poblacion(tamano):
        poblacion = np.zeros((tamano, len(tickers)))
        for i in range(tamano):
            num_activos = np.random.randint(5, len(tickers) + 1)
            activos = np.random.choice(len(tickers), num_activos, replace=False)
            pesos = np.random.dirichlet(np.ones(num_activos) * 5)
            poblacion[i, activos] = pesos
            poblacion[i] /= np.sum(poblacion[i])
        return poblacion

    def ordenamiento_no_dominado(poblacion, objetivos):
        frentes = [[]]
        dominados = {i: [] for i in range(len(poblacion))}
        conteo_dominancia = [0] * len(poblacion)
        
        for i in range(len(poblacion)):
            for j in range(len(poblacion)):
                if i != j:
                    if domina(objetivos[i], objetivos[j]):
                        dominados[i].append(j)
                    elif domina(objetivos[j], objetivos[i]):
                        conteo_dominancia[i] += 1
            if conteo_dominancia[i] == 0:
                frentes[0].append(i)
        
        i = 0
        while frentes[i]:
            siguiente_frente = []
            for idx in frentes[i]:
                for dom_idx in dominados[idx]:
                    conteo_dominancia[dom_idx] -= 1
                    if conteo_dominancia[dom_idx] == 0:
                        siguiente_frente.append(dom_idx)
            i += 1
            frentes.append(siguiente_frente)
        return frentes[:-1]

    def calcular_crowding_distance(objetivos, frente):
        if len(frente) == 0:
            return []
        distancias = [0] * len(objetivos)
        for m in range(2):
            sorted_indices = np.argsort([objetivos[i][m] for i in frente])
            distancias[frente[sorted_indices[0]]] = float('inf')
            distancias[frente[sorted_indices[-1]]] = float('inf')
            if len(frente) > 2:
                max_val = objetivos[frente[sorted_indices[-1]]][m]
                min_val = objetivos[frente[sorted_indices[0]]][m]
                if max_val != min_val:
                    for i in range(1, len(frente) - 1):
                        distancias[frente[sorted_indices[i]]] += (
                            objetivos[frente[sorted_indices[i+1]]][m] - 
                            objetivos[frente[sorted_indices[i-1]]][m]
                        ) / (max_val - min_val)
        return distancias

    def seleccionar_padres(poblacion, objetivos, tamano_torneo=2):
        frentes = ordenamiento_no_dominado(poblacion, objetivos)
        distancias = []
        for frente in frentes:
            distancias.extend([(i, d) for i, d in zip(frente, calcular_crowding_distance(objetivos, frente))])
        
        def torneo():
            candidatos = np.random.choice(len(poblacion), tamano_torneo, replace=False)
            mejor = None
            mejor_rango = float('inf')
            mejor_distancia = -float('inf')
            for idx in candidatos:
                rango = next(i for i, frente in enumerate(frentes) if idx in frente)
                distancia = next(d for i, d in distancias if i == idx)
                if rango < mejor_rango or (rango == mejor_rango and distancia > mejor_distancia):
                    mejor = idx
                    mejor_rango = rango
                    mejor_distancia = distancia
            return poblacion[mejor]
        
        return torneo(), torneo()

    def cruzar(padre1, padre2):
        alpha = np.random.uniform(0.3, 0.7)
        hijo1 = alpha * padre1 + (1 - alpha) * padre2
        hijo2 = (1 - alpha) * padre1 + alpha * padre2
        hijo1 = np.clip(hijo1, 0, 1)
        hijo2 = np.clip(hijo2, 0, 1)
        hijo1 /= np.sum(hijo1) if np.sum(hijo1) > 1e-10 else 1
        hijo2 /= np.sum(hijo2) if np.sum(hijo2) > 1e-10 else 1
        return hijo1, hijo2

    def mutar(pesos, tasa_mutacion=0.2):
        if np.random.rand() < tasa_mutacion:
            mutacion = np.random.normal(0, 0.02, len(pesos))
            pesos = np.clip(pesos + mutacion, 0, 1)
            pesos /= np.sum(pesos) if np.sum(pesos) > 1e-10 else 1
        return pesos

    poblacion = inicializar_poblacion(tamano_poblacion)
    
    for _ in tqdm(range(num_generaciones), desc="Optimizando con NSGA-II"):
        objetivos = [calcular_objetivos(ind) for ind in poblacion]
        
        poblacion_hija = []
        while len(poblacion_hija) < tamano_poblacion:
            padre1, padre2 = seleccionar_padres(poblacion, objetivos)
            hijo1, hijo2 = cruzar(padre1, padre2)
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)
            poblacion_hija.extend([hijo1, hijo2])
        
        poblacion_combinada = np.vstack((poblacion, poblacion_hija[:tamano_poblacion]))
        objetivos_combinados = [calcular_objetivos(ind) for ind in poblacion_combinada]
        
        frentes = ordenamiento_no_dominado(poblacion_combinada, objetivos_combinados)
        nueva_poblacion = []
        i = 0
        
        while len(nueva_poblacion) < tamano_poblacion and i < len(frentes):
            frente = frentes[i]
            if len(nueva_poblacion) + len(frente) <= tamano_poblacion:
                nueva_poblacion.extend([poblacion_combinada[idx] for idx in frente])
            else:
                distancias = calcular_crowding_distance(objetivos_combinados, frente)
                sorted_frente = [x for _, x in sorted(zip(distancias, frente), reverse=True)]
                nueva_poblacion.extend([poblacion_combinada[idx] for idx in sorted_frente[:tamano_poblacion - len(nueva_poblacion)]])
            i += 1
        
        poblacion = np.array(nueva_poblacion)
    
    objetivos = [calcular_objetivos(ind) for ind in poblacion]
    frentes = ordenamiento_no_dominado(poblacion, objetivos)
    frente_pareto = frentes[0]
    
    sharpes = [objetivos[i][1] for i in frente_pareto]
    volatilidades = [objetivos[i][0] for i in frente_pareto]
    
    idx_max_sharpe = frente_pareto[np.argmax(sharpes)]
    idx_min_vol = frente_pareto[np.argmin(volatilidades)]
    
    return poblacion[idx_max_sharpe], poblacion[idx_min_vol]

# --- Red Neuronal Recurrente ---
class TqdmCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.tqdm_bar = None

    def on_train_begin(self, logs=None):
        self.tqdm_bar = tqdm(total=self.total_epochs, desc="Entrenando RNN", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.tqdm_bar.update(1)
        self.tqdm_bar.set_postfix({
            'loss': f"{logs.get('loss', 0):.4f}",
            'mae': f"{logs.get('mae', 0):.4f}",
            'val_loss': f"{logs.get('val_loss', 0):.4f}",
            'val_mae': f"{logs.get('val_mae', 0):.4f}"
        })

    def on_train_end(self, logs=None):
        self.tqdm_bar.close()

def cnn_optimization(train_weekly_returns, risk_free_rate_user):
    def prepare_cnn_data(returns, lookback=10):
        X, y = [], []
        for i in range(len(returns) - lookback):
            X.append(returns.iloc[i:i+lookback].values)
            y.append(returns.iloc[i+lookback].values)
        return np.array(X), np.array(y)

    def build_cnn_model(input_shape, n_assets):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(n_assets, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mape', metrics=['mae'])
        return model

    def train_cnn_model(X, y, epochs=50, batch_size=32):
        model = build_cnn_model((X.shape[1], X.shape[2]), y.shape[1])
        model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0,
                  callbacks=[TqdmCallback(total_epochs=epochs)])
        return model

    def predict_returns(model, returns_data):
        X, _ = prepare_cnn_data(returns_data)
        if len(X) == 0:
            return np.zeros(len(tickers))
        predictions = model.predict(X[-1:], verbose=0)
        return predictions[0]

    def objective_max_sharpe(weights):
        ret, vol, sharpe, _, _, _ = portfolio_metrics(weights, train_weekly_returns, risk_free_rate_user)
        return -sharpe

    def objective_min_volatility(weights):
        ret, vol, sharpe, _, _, _ = portfolio_metrics(weights, train_weekly_returns, risk_free_rate_user)
        return vol

    def optimize_portfolio(objective, max_attempts=5):
        best_result = None
        best_value = float('inf') if 'min' in objective.__name__ else float('-inf')
        
        for _ in tqdm(range(max_attempts), desc=f"Optimizando pesos {'Max Sharpe' if 'sharpe' in objective.__name__ else 'Min Volatilidad'} con RNN"):
            initial_weights = np.random.random(len(tickers))
            initial_weights /= np.sum(initial_weights)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=[(0, 1)] * len(tickers),
                constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                value = result.fun if 'min' in objective.__name__ else -result.fun
                if 'min' in objective.__name__ and value < best_value:
                    best_value = value
                    best_result = result
                elif 'max' in objective.__name__ and value > best_value:
                    best_value = value
                    best_result = result
        
        return best_result.x if best_result else np.ones(len(tickers)) / len(tickers)

    X, y = prepare_cnn_data(train_weekly_returns)
    if len(X) > 0:
        cnn_model = train_cnn_model(X, y)
        
        weights_sharpe = optimize_portfolio(objective_max_sharpe)
        weights_volatility = optimize_portfolio(objective_min_volatility)
        
        return weights_sharpe, weights_volatility
    else:
        uniform_weights = np.ones(len(tickers)) / len(tickers)
        return uniform_weights, uniform_weights

# Función para optimización Markowitz
def markowitz_optimization(train_weekly_returns, risk_free_rate_user):
    retornos_esperados = train_weekly_returns.mean() * 52
    lw = LedoitWolf()
    lw.fit(train_weekly_returns)
    matriz_covarianza = lw.covariance_ * 52
    
    num_assets = len(tickers)
    num_portfolios = 10000
    port_returns, port_volatility, sharpe_ratio, stock_weights = [], [], [], []

    for _ in tqdm(range(num_portfolios), desc="Optimizando con Markowitz"):
        num_selected_assets = np.random.randint(1, num_assets + 1)
        selected_assets = np.random.choice(tickers, num_selected_assets, replace=False)
        weights = np.random.random(num_selected_assets)
        weights /= np.sum(weights)
        full_weights = np.zeros(num_assets)
        for i, stock in enumerate(selected_assets):
            full_weights[tickers.index(stock)] = weights[i]
        
        ret, vol, sharpe, _, _, _ = portfolio_metrics(full_weights, train_weekly_returns, risk_free_rate_user)
        
        port_returns.append(ret)
        port_volatility.append(vol)
        sharpe_ratio.append(sharpe)
        stock_weights.append(full_weights)

    df = pd.DataFrame({
        'Returns': port_returns, 'Volatility': port_volatility, 'Sharpe Ratio': sharpe_ratio
    })
    for i, stock in enumerate(tickers):
        df[stock + ' Weight'] = [weights[i] for weights in stock_weights]

    sharpe_portfolio = df.loc[df['Sharpe Ratio'].idxmax()]
    min_variance_port = df.loc[df['Volatility'].idxmin()]

    return sharpe_portfolio[[f'{ticker} Weight' for ticker in tickers]].values, min_variance_port[[f'{ticker} Weight' for ticker in tickers]].values

# Función para crear treemap
def create_treemap_plot(weights, tickers):
    # Filter weights > 1% and sort
    composicion = {ticker: peso for ticker, peso in zip(tickers, weights) if peso > 0.01}
    composicion_ordenada = dict(sorted(composicion.items(), key=lambda item: item[1], reverse=True))
    
    labels = list(composicion_ordenada.keys())
    values = list(composicion_ordenada.values())
    
    # Create labels with weights as percentages
    formatted_labels = [f"{ticker}<br>{peso*100:.2f}%" for ticker, peso in zip(labels, values)]
    
    # Escala de colores personalizada: tonos de azul del 30% al 100% de opacidad
    custom_colorscale = [
        [0.0, 'rgba(0, 0, 225, 0.2)'],  # Azul oscuro al 30% de opacidad
        [1.0, 'rgba(30, 64, 175, 1.0)']   # Azul oscuro al 100% de opacidad
    ]
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=formatted_labels,
        parents=[""] * len(labels),  # No hierarchy, all assets at same level
        values=values,
        textinfo="label",
        textfont=dict(size=14, color="white", family="Arial, bold"),
        marker=dict(
            colors=values,
            colorscale=custom_colorscale,  # Usar escala personalizada
            showscale=False,
            line=dict(width=2, color="black")
        ),
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Distribución de Activos (Treemap)",
            font=dict(color="#E5E7EB", size=20)
        ),
        margin=dict(t=50, l=25, r=25, b=25),
        width=600,
        height=600,
        paper_bgcolor="#1F2937",
        font=dict(color="#E5E7EB"),
        treemapcolorway=["#374151"],  # Background for treemap
    )
    
    return fig

# Interface Design with Dark Mode Compatibility
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #BFDBFE;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #D1D5DB;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #374151;
        color: #E5E7EB;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #60A5FA;
        margin-bottom: 20px;
    }
    .info-box a {
        color: #93C5FD;
        text-decoration: underline;
    }
    .info-box a:hover {
        color: #BFDBFE;
    }
    .stButton>button {
        background-color: #60A5FA;
        color: #1F2937;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #1F2937;
        padding: 20px;
        color: #E5E7EB;
    }
    .metric-card {
        background-color: #374151;
        color: #E5E7EB;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-align: center;
    }
    .stDataFrame table {
        color: #E5E7EB !important;
    }
    .stDataFrame th {
        color: #D1D5DB !important;
    }
    .stMetric label {
        color: #D1D5DB !important;
    }
    .stMetric span {
        color: #E5E7EB !important;
    }
    .stCaption {
        color: #9CA3AF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">Optimizador de Carteras IBEX35</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    Bienvenido a la herramienta de optimización de carteras del IBEX35. Esta aplicación utiliza tres métodos avanzados para optimizar tu cartera:
    <ul>
        <li><b>Markowitz</b>: Optimización clásica basada en la teoría moderna de carteras.</li>
        <li><b>NSGA-II</b>: Algoritmo genético multiobjetivo para maximizar el retorno y minimizar el riesgo.</li>
        <li><b>RNN</b>: Red neuronal recurrente para optimizar pesos.</li>
    </ul>
    Ajusta el nivel de riesgo en la barra lateral y haz clic en "Optimizar Carteras" para comenzar.
</div>
""", unsafe_allow_html=True)

# Load data
train_data, test_data, train_weekly_returns, test_weekly_returns, full_data, risk_free_rate, current_bond_yield = load_data()

# Sidebar
with st.sidebar:
    st.header("Configuración")
    
    st.subheader("Parámetros de Optimización")
    riesgo_maximo = st.slider(
        "Nivel máximo de riesgo (volatilidad %)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.5,
        help="Define el nivel máximo de volatilidad anual que estás dispuesto a aceptar."
    ) / 100

    st.subheader("Instrucciones")
    st.markdown("""
        <div class="info-box">
            <ol>
                <li>Ajusta el nivel máximo de riesgo.</li>
                <li>Haz clic en "Optimizar Carteras" para generar las carteras.</li>
                <li>Explora los resultados en las pestañas de métricas, rendimiento, análisis estadístico y composición.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Acciones")
    optimize_button = st.button("Optimizar Carteras", key="optimize_button", help="Inicia el proceso de optimización de carteras.")

# Definir tasa libre de riesgo fija
risk_free_rate_user = risk_free_rate  # Usar la tasa libre de riesgo obtenida de load_data()

# Main content
if optimize_button:
    with st.spinner('Optimizando carteras... Esto puede tomar unos minutos'):
        markowitz_sharpe, markowitz_vol = markowitz_optimization(train_weekly_returns, risk_free_rate_user)
        nsga_sharpe, nsga_vol = nsga_ii_optimization(train_weekly_returns, risk_free_rate_user)
        cnn_sharpe, cnn_vol = cnn_optimization(train_weekly_returns, risk_free_rate_user)
        
        carteras = {
            'Markowitz Max Sharpe': markowitz_sharpe,
            'Markowitz Min Vol': markowitz_vol,
            'NSGA-II Max Sharpe': nsga_sharpe,
            'NSGA-II Min Vol': nsga_vol,
            'RNN Max Sharpe': cnn_sharpe,
            'RNN Min Vol': cnn_vol
        }
        st.session_state.carteras = carteras

if 'carteras' in st.session_state:
    carteras = st.session_state.carteras
    
    # Select best portfolio
    mejor_cartera = None
    mejor_retorno = -np.inf
    mejor_nombre = ""
    mejor_vol = 0
    mejor_sharpe = 0
    mejor_var = 0
    mejor_cvar = 0
    mejor_sortino = 0

    for nombre, pesos in carteras.items():
        ret, vol, sharpe, var, cvar, sortino = portfolio_metrics(pesos, train_weekly_returns, risk_free_rate_user)
        if vol <= riesgo_maximo and ret > mejor_retorno:
            mejor_retorno = ret
            mejor_vol = vol
            mejor_sharpe = sharpe
            mejor_var = var
            mejor_cvar = cvar
            mejor_sortino = sortino
            mejor_cartera = pesos
            mejor_nombre = nombre

    if mejor_cartera is None:
        mejor_nombre = min(carteras.keys(), key=lambda x: portfolio_metrics(carteras[x], train_weekly_returns, risk_free_rate_user)[1])
        mejor_cartera = carteras[mejor_nombre]
        mejor_retorno, mejor_vol, mejor_sharpe, mejor_var, mejor_cvar, mejor_sortino = portfolio_metrics(mejor_cartera, train_weekly_returns, risk_free_rate_user)
        st.warning(f"⚠️ Todas las carteras tienen mayor riesgo que el seleccionado. Mostrando la de menor volatilidad ({mejor_nombre})")

    st.markdown(f'<div class="section-header">Cartera Recomendada: {mejor_nombre}</div>', unsafe_allow_html=True)
    
    test_ret, test_vol, test_sharpe, test_var, test_cvar, test_sortino = portfolio_metrics(mejor_cartera, test_weekly_returns, risk_free_rate_user)

    # Tabs for results
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Métricas", "📈 Rendimiento", "🔬 Análisis Estadístico", "📋 Composición"])

    with tab1:
        st.markdown('<div class="section-header">Métricas de la Cartera</div>', unsafe_allow_html=True)
        
        metric_type = st.radio(
            "Seleccionar período:",
            ["Entrenamiento", "Prueba"],
            horizontal=True
        )

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        if metric_type == "Entrenamiento":
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Retorno Esperado Anual", f"{mejor_retorno*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Volatilidad Anual", f"{mejor_vol*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Ratio de Sharpe", f"{mejor_sharpe:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("VaR 95% Anual", f"{mejor_var*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CVaR 95% Anual", f"{mejor_cvar*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col6:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Ratio de Sortino", f"{mejor_sortino:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:  # Prueba
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Retorno Realizado Anual", f"{test_ret*100:.2f}%",
                         delta=f"{(test_ret-mejor_retorno)*100:.2f}% vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Volatilidad Realizada", f"{test_vol*100:.2f}%",
                         delta=f"{(test_vol-mejor_vol)*100:.2f}% vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Ratio de Sharpe Realizado", f"{test_sharpe:.2f}",
                         delta=f"{(test_sharpe-mejor_sharpe):.2f} vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("VaR 95% Realizado", f"{test_var*100:.2f}%",
                         delta=f"{(test_var-mejor_var)*100:.2f}% vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)
            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CVaR 95% Realizado", f"{test_cvar*100:.2f}%",
                         delta=f"{(test_cvar-mejor_cvar)*100:.2f}% vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)
            with col6:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Ratio de Sortino Realizado", f"{test_sortino:.2f}",
                         delta=f"{(test_sortino-mejor_sortino):.2f} vs entrenamiento")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Comparación de Rendimiento en Test</div>', unsafe_allow_html=True)
        
        start_date = pd.to_datetime('2024-7-7')
        end_date = pd.to_datetime('2025-7-7')
        returns_data = test_weekly_returns.loc[(test_weekly_returns.index >= start_date) & (test_weekly_returns.index <= end_date)]
        
        if returns_data.empty:
            st.error("No hay datos disponibles para el rango de fechas seleccionado.")
        else:
            # Calculate cumulative returns for the benchmark (IBEX35)
            benchmark_returns = np.expm1(returns_data.mean(axis=1).cumsum())

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                'Fecha': returns_data.index,
                'IBEX35': benchmark_returns
            })

            colors = px.colors.qualitative.Plotly

            # Calculate cumulative returns for each portfolio
            for i, (nombre, pesos) in enumerate(carteras.items()):
                portfolio_returns = np.expm1((returns_data @ pesos).cumsum())
                df_plot[nombre] = portfolio_returns

            df_plot_melted = df_plot.melt(id_vars='Fecha', var_name='Serie', value_name='Retorno')

            # Create line plot
            fig = px.line(df_plot_melted, x='Fecha', y='Retorno', color='Serie',
                         labels={'Retorno': 'Retorno Acumulado', 'Fecha': 'Fecha'},
                         color_discrete_map={
                             'IBEX35': '#6B7280',
                             **{nombre: colors[i%len(colors)] for i, nombre in enumerate(carteras.keys())}
                         })

            # Highlight the selected portfolio with a dotted line
            fig.update_traces(
                line=dict(width=3, dash='dot'),
                selector=dict(name=mejor_nombre)
            )
            fig.update_traces(
                line=dict(width=2),
                selector=dict(name=lambda x: x != mejor_nombre)
            )

            # Update layout for dark mode
            fig.update_layout(
                hovermode='x unified',
                title=dict(
                    text="Comparación de Rendimiento",
                    font=dict(color="#FFFFFF", size=20)
                ),
                legend=dict(
                    title=dict(
                        text='Serie',
                        font=dict(color="#FFFFFF", size=14)
                    ),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#FFFFFF", size=12)
                ),
                height=500,
                paper_bgcolor='#1F2937',
                plot_bgcolor='#1F2937',
                xaxis=dict(
                    gridcolor='#374151',
                    tickfont=dict(color="#FFFFFF", size=14),
                    title_font=dict(color="#FFFFFF", size=16),
                    title_text='Fecha',
                    dtick='M1',  # Intervalo de un mes
                    tickangle=45
                ),
                yaxis=dict(
                    gridcolor='#374151',
                    tickfont=dict(color="#FFFFFF", size=14),
                    title_font=dict(color="#FFFFFF", size=16),
                    title_text='Retorno Acumulado',
                    tick0=-0.1,
                    dtick=0.05,
                    range=[-0.1, 0.5]
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Análisis Estadístico de Carteras</div>', unsafe_allow_html=True)
        
        st.markdown("### Intervalos de Confianza de las Métricas (Período de Entrenamiento)")
        ci_data = []
        confidence_level = 0.95
        for nombre, pesos in carteras.items():
            returns = train_weekly_returns @ pesos
            n = len(returns)
            t_crit = t.ppf((1 + confidence_level) / 2, n - 1)
            
            # Retorno Anual
            annualized_returns = returns * 52
            mean_return = np.mean(annualized_returns)
            std_err_return = sem(annualized_returns)
            ci_return = std_err_return * t_crit
            
            # Volatilidad Anual
            portfolio_vol = returns.std() * np.sqrt(52)
            var_returns = returns.var()
            se_vol = np.sqrt(var_returns / (2 * n)) * np.sqrt(52)
            ci_vol = se_vol * t_crit
            
            # Sharpe
            sharpe = (mean_return - risk_free_rate_user) / portfolio_vol if portfolio_vol > 1e-10 else 0
            sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n)
            ci_sharpe = sharpe_se * t_crit
            
            # VaR 95% Anual
            var_95 = norm.ppf(1 - confidence_level, returns.mean(), returns.std()) * np.sqrt(52)
            var_se = returns.std() / np.sqrt(n) * norm.ppf(1 - confidence_level) * np.sqrt(52)
            ci_var = var_se * t_crit
            
            # CVaR 95% Anual
            cvar_95 = returns.mean() - returns.std() * norm.pdf(norm.ppf(confidence_level)) / (1 - confidence_level) * np.sqrt(52)
            cvar_se = returns.std() / np.sqrt(n) * norm.pdf(1 - confidence_level) * np.sqrt(52)
            ci_cvar = cvar_se * t_crit
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else portfolio_vol
            sortino = (mean_return - risk_free_rate_user) / downside_std if downside_std > 1e-10 else 0
            sortino_se = np.sqrt((1 + 0.5 * sortino**2) / n)
            ci_sortino = sortino_se * t_crit
            
            ci_data.append({
                'Cartera': nombre,
                'Retorno Media': mean_return,
                'Retorno CI Lower': mean_return - ci_return,
                'Retorno CI Upper': mean_return + ci_return,
                'Volatilidad Media': portfolio_vol,
                'Volatilidad CI Lower': portfolio_vol - ci_vol,
                'Volatilidad CI Upper': portfolio_vol + ci_vol,
                'Sharpe Media': sharpe,
                'Sharpe CI Lower': sharpe - ci_sharpe,
                'Sharpe CI Upper': sharpe + ci_sharpe,
                'VaR 95% Media': var_95,
                'VaR 95% CI Lower': var_95 - ci_var,
                'VaR 95% CI Upper': var_95 + ci_var,
                'CVaR 95% Media': cvar_95,
                'CVaR 95% CI Lower': cvar_95 - ci_cvar,
                'CVaR 95% CI Upper': cvar_95 + ci_cvar,
                'Sortino Media': sortino,
                'Sortino CI Lower': sortino - ci_sortino,
                'Sortino CI Upper': sortino + ci_sortino
            })
        
        ci_df = pd.DataFrame(ci_data)
        
        # Plot CI para cada métrica
        metrics = ['Retorno', 'Volatilidad', 'Sharpe', 'VaR 95%', 'CVaR 95%', 'Sortino']
        for metric in metrics:
            fig_ci = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, row in ci_df.iterrows():
                fig_ci.add_trace(go.Scatter(
                    x=[row['Cartera']],
                    y=[row[f'{metric} Media']],
                    error_y=dict(
                        type='data',
                        symmetric=True,
                        array=[row[f'{metric} CI Upper'] - row[f'{metric} Media']],
                        thickness=1.5,
                        width=5
                    ),
                    mode='markers',
                    marker=dict(size=10, color=colors[i % len(colors)]),
                    name=row['Cartera']
                ))
            
            y_title = {
                'Retorno': 'Retorno Anual',
                'Volatilidad': 'Volatilidad Anual',
                'Sharpe': 'Ratio de Sharpe',
                'VaR 95%': 'VaR 95% Anual',
                'CVaR 95%': 'CVaR 95% Anual',
                'Sortino': 'Ratio de Sortino'
            }[metric]
            
            fig_ci.update_layout(
                title=dict(
                    text=f"Intervalos de Confianza del 95% para {metric} (Período de Entrenamiento)",
                    font=dict(color="#FFFFFF", size=16)
                ),
                xaxis_title="Cartera",
                yaxis_title=y_title,
                legend=dict(
                    font=dict(color="#FFFFFF", size=12)
                ),
                height=500,
                showlegend=True,
                xaxis=dict(
                    tickangle=45,
                    gridcolor='#374151',
                    tickfont=dict(color="#FFFFFF"),
                    title_font=dict(color="#FFFFFF")
                ),
                yaxis=dict(
                    gridcolor='#374151',
                    tickfont=dict(color="#FFFFFF"),
                    title_font=dict(color="#FFFFFF")
                ),
                paper_bgcolor='#1F2937',
                plot_bgcolor='#1F2937',
                font=dict(color="#FFFFFF")
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        
        st.markdown("*Nota: Los intervalos que se solapan indican que las carteras no son significativamente diferentes en términos de la métrica correspondiente.*")

    with tab4:
        st.markdown('<div class="section-header">Composición de la Cartera</div>', unsafe_allow_html=True)
        
        # Dropdown to select portfolio
        portfolio_options = list(carteras.keys())
        default_index = portfolio_options.index(mejor_nombre) if mejor_nombre in portfolio_options else 0
        selected_portfolio = st.selectbox(
            "Seleccionar Cartera",
            options=portfolio_options,
            index=default_index,
            help="Selecciona una cartera para ver su composición y distribución de activos."
        )
        
        # Get weights for the selected portfolio
        selected_weights = carteras[selected_portfolio]
        
        col1, col2 = st.columns([1, 2])

        with col1:
            # Create composition dictionary, filtering out weights < 1%
            composicion = {ticker: peso for ticker, peso in zip(tickers, selected_weights) if peso > 0.01}
            composicion_ordenada = dict(sorted(composicion.items(), key=lambda item: item[1], reverse=True))
            composicion_df = pd.DataFrame.from_dict(composicion_ordenada, orient='index', columns=['Peso'])
            
            st.markdown(f"**Número de activos en la cartera: {len(composicion_df)}**")
            st.dataframe(
                composicion_df.style.format("{:.2%}"),
                use_container_width=True,
                height=len(composicion_df) * 35 + 35
            )

        with col2:
            treemap_fig = create_treemap_plot(selected_weights, tickers)
            st.plotly_chart(treemap_fig, use_container_width=True)

    # Comparación de Todas las Carteras
    st.markdown('<div class="section-header">Comparación de Todas las Carteras</div>', unsafe_allow_html=True)
    
    # Training Metrics Comparison
    st.markdown('<div class="section-header">Métricas de comparación: Train</div>', unsafe_allow_html=True)
    carteras_df_train = []
    for nombre, pesos in carteras.items():
        ret, vol, sharpe, var, cvar, sortino = portfolio_metrics(pesos, train_weekly_returns, risk_free_rate_user)
        carteras_df_train.append({
            'Método': nombre,
            'Retorno': ret,
            'Volatilidad': vol,
            'Sharpe': sharpe,
            'VaR 95%': var,
            'CVaR 95%': cvar,
            'Sortino': sortino,
            'Seleccionada': nombre == mejor_nombre
        })
    
    df_display_train = pd.DataFrame(carteras_df_train).sort_values('Sharpe', ascending=False)
    st.dataframe(
        df_display_train.style.apply(
            lambda x: ['background-color: #4B5563' if x.Seleccionada else '' for _ in x],
            axis=1   
        ).format({
            "Retorno": "{:.2%}",
            "Volatilidad": "{:.2%}",
            "Sharpe": "{:.2f}",
            "VaR 95%": "{:.2f}",
            "CVaR 95%": "{:.2f}",
            "Sortino": "{:.2f}"
        }),
        column_order=['Método', 'Retorno', 'Volatilidad', 'Sharpe', 'VaR 95%', 'CVaR 95%', 'Sortino'],
        hide_index=True,
        use_container_width=True
    )

    # Test Metrics Comparison
    st.markdown('<div class="section-header">Métricas de comparación: Test</div>', unsafe_allow_html=True)
    carteras_df_test = []
    for nombre, pesos in carteras.items():
        ret, vol, sharpe, var, cvar, sortino = portfolio_metrics(pesos, test_weekly_returns, risk_free_rate_user)
        carteras_df_test.append({
            'Método': nombre,
            'Retorno': ret,
            'Volatilidad': vol,
            'Sharpe': sharpe,
            'VaR 95%': var,
            'CVaR 95%': cvar,
            'Sortino': sortino,
            'Seleccionada': nombre == mejor_nombre
        })
    
    df_display_test = pd.DataFrame(carteras_df_test).sort_values('Sharpe', ascending=False)
    st.dataframe(
        df_display_test.style.apply(
            lambda x: ['background-color: #4B5563' if x.Seleccionada else '' for _ in x],
            axis=1   
        ).format({
            "Retorno": "{:.2%}",
            "Volatilidad": "{:.2%}",
            "Sharpe": "{:.2f}",
            "VaR 95%": "{:.2f}",
            "CVaR 95%": "{:.2f}",
            "Sortino": "{:.2f}"
        }),
        column_order=['Método', 'Retorno', 'Volatilidad', 'Sharpe', 'VaR 95%', 'CVaR 95%', 'Sortino'],
        hide_index=True,
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption("""
*Nota: Los resultados se basan en datos históricos y no garantizan rendimientos futuros. 
La diversificación no elimina completamente el riesgo de mercado.*
""")
