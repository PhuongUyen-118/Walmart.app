import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🔮 Dự đoán Doanh số Walmart")

# Upload file CSV
uploaded_file = st.file_uploader("Tải lên file WalmartSales.csv.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Hiển thị bảng dữ liệu
    st.subheader("📄 Dữ liệu đã tải lên:")
    st.write(df.head())

    # Tiền xử lý
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Lựa chọn biến
    X = df[["Temperature", "Fuel_Price", "CPI", "Unemployment"]]
    y = df["Weekly_Sales"]

    # Huấn luyện mô hình
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Form nhập giá trị đầu vào
    st.sidebar.header("🧪 Nhập thông tin để dự đoán:")
    temp = st.sidebar.slider("Temperature", float(X.Temperature.min()), float(X.Temperature.max()))
    fuel = st.sidebar.slider("Fuel_Price", float(X.Fuel_Price.min()), float(X.Fuel_Price.max()))
    cpi = st.sidebar.slider("CPI", float(X.CPI.min()), float(X.CPI.max()))
    unemploy = st.sidebar.slider("Unemployment", float(X.Unemployment.min()), float(X.Unemployment.max()))

    # Dự đoán
    input_data = np.array([[temp, fuel, cpi, unemploy]])
    prediction = model.predict(input_data)

    st.subheader("📈 Dự đoán doanh số:")
    st.success(f"Doanh số dự đoán: ${prediction[0]:,.2f}")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Phân tích dữ liệu Walmart")

# 1. Line chart: Weekly Sales over Time
st.subheader("1. Weekly Sales over Time")
fig1 = plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['Weekly_Sales'])
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Over Time')
st.pyplot(fig1)

# 2. Bar chart: Average Sales by Store
st.subheader("2. Average Sales by Store")
fig2 = plt.figure(figsize=(10,5))
store_avg = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
store_avg.plot(kind='bar')
plt.title('Average Weekly Sales by Store')
st.pyplot(fig2)

# 3. Boxplot: Sales distribution
st.subheader("3. Weekly Sales Distribution")
fig3 = plt.figure(figsize=(8,4))
sns.boxplot(data=df, y='Weekly_Sales')
plt.title('Sales Distribution')
st.pyplot(fig3)

# 4. Pairplot: Relationship between features
st.subheader("4. Feature Relationships (Pairplot)")
sns_plot = sns.pairplot(df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]])
st.pyplot(sns_plot)

# 5. Heatmap: Correlation matrix
st.subheader("5. Correlation Heatmap")
fig5 = plt.figure(figsize=(8,6))
sns.heatmap(df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
st.pyplot(fig5)

