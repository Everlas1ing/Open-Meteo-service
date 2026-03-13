import streamlit as st
import pandas as pd
import requests
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# --- Налаштування сторінки ---
st.set_page_config(page_title="Прогноз опадів", layout="wide")
st.title("🌦️ Прогноз опадів на основі ML (Open-Meteo)")

# --- Збереження стану (Session State) ---
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None

# --- Бічна панель: Налаштування ---
st.sidebar.header("1. Отримання даних")
lat = st.sidebar.number_input("Широта (Latitude)", value=50.45) # Київ за замовчуванням
lon = st.sidebar.number_input("Довгота (Longitude)", value=30.52)

# Вибираємо період (за замовчуванням останні 90 днів)
end_date_default = datetime.date.today() - datetime.timedelta(days=1)
start_date_default = end_date_default - datetime.timedelta(days=90)

start_date = st.sidebar.date_input("Дата початку", value=start_date_default)
end_date = st.sidebar.date_input("Дата кінця", value=end_date_default)

# --- КРОК 1: Завантаження даних ---
if st.sidebar.button("Отримати дані з Open-Meteo"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max",
        "timezone": "auto"
    }
    
    with st.spinner("Завантаження даних..."):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            daily = data.get("daily", {})
            
            # Створюємо DataFrame
            df = pd.DataFrame({
                "date": daily.get("time"),
                "temp_max": daily.get("temperature_2m_max"),
                "temp_min": daily.get("temperature_2m_min"),
                "wind_speed": daily.get("wind_speed_10m_max"),
                "rain_sum": daily.get("rain_sum"),
                "precipitation_sum": daily.get("precipitation_sum")
            })
            
            # Видаляємо порожні значення
            df = df.dropna()
            
            # Зберігаємо у CSV та в стан застосунку
            df.to_csv("weather_daily.csv", index=False)
            st.session_state.df = df
            
            st.success("Дані успішно завантажено та збережено у weather_daily.csv!")
        else:
            st.error(f"Помилка API: {response.status_code}")

# Якщо дані є локально, але не в пам'яті (наприклад, після перезапуску)
if st.session_state.df is None and os.path.exists("weather_daily.csv") and os.path.getsize("weather_daily.csv") > 0:
    st.session_state.df = pd.read_csv("weather_daily.csv")

if st.session_state.df is not None:
    st.write("### Огляд даних", st.session_state.df.tail())

st.divider()

# --- КРОК 2: Навчання моделі ---
st.header("2. Навчання ML-моделі")

if st.session_state.df is not None:
    if st.button("Навчити модель"):
        df = st.session_state.df
        
        # Формування цільової змінної (0 - немає опадів, 1 - є опади)
        df['target'] = (df['precipitation_sum'] > 0).astype(int)
        
        # Ознаки (X) та цільова змінна (y)
        # УВАГА: Ми не беремо rain_sum та precipitation_sum у X!
        X = df[['temp_max', 'temp_min', 'wind_speed']]
        y = df['target']
        
        # Розбиття на train/test (80% на навчання, 20% на тест)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Навчання RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Прогноз на тестових даних
        y_pred = model.predict(X_test)
        
        # Метрики
        st.write("### Метрики якості моделі:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy (Точність)", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
        col4.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")
        
        st.session_state.model = model
        st.success("Модель успішно навчена!")
else:
    st.info("Спочатку завантажте дані (Крок 1).")

st.divider()

# --- КРОК 3: Прогноз ---
st.header("3. Зробити прогноз")

if st.session_state.model is not None:
    st.write("Введіть погодні умови для дня, на який хочете зробити прогноз:")
    
    col_a, col_b, col_c = st.columns(3)
    input_temp_max = col_a.number_input("Макс. температура (°C)", value=20.0)
    input_temp_min = col_b.number_input("Мін. температура (°C)", value=10.0)
    input_wind = col_c.number_input("Швидкість вітру (км/год)", value=15.0)
    
    if st.button("Зробити прогноз"):
        # Формуємо датафрейм для одного прикладу
        input_data = pd.DataFrame({
            'temp_max': [input_temp_max],
            'temp_min': [input_temp_min],
            'wind_speed': [input_wind]
        })
        
        model = st.session_state.model
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Ймовірність того, що опади будуть (клас 1)
        prob_rain = probabilities[1] * 100
        
        if prediction == 1:
            st.error(f"🌧️ **Очікуються опади!**")
        else:
            st.success(f"☀️ **Опадів не очікується!**")
            
        st.write(f"**Ймовірність опадів:** {prob_rain:.1f}%")
else:
    st.info("Спочатку навчіть модель (Крок 2).")