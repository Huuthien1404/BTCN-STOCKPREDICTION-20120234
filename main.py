import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Cấu hình trang
st.set_page_config(page_title="Ứng dụng Dự báo Chứng khoán", layout="wide", initial_sidebar_state="expanded")

# CSS tùy chỉnh cho hình nền và kiểu chữ
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .reportview-container {
        background: url("https://example.com/background-image.jpg");
        background-size: cover;
    }
    .title {
        color: #2c3e50;
        font-size: 50px;
        font-weight: bold;
        width: 100%;
        text-align: center;
        margin-top: 20px;
    }
    .header {
        color: #34495e;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #7f8c8d;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề và tiêu đề phụ
st.markdown('<div class="title">Ứng dụng Dự báo Chứng khoán</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Họ và tên: Ngô Nguyễn Quang Tú</div>', unsafe_allow_html=True)
st.markdown('<div class="header">MSSV: 20120234</div>', unsafe_allow_html=True)

# Khoảng trắng và giới thiệu
st.write("")
st.markdown('<div class="subheader">Bài tập cá nhân Dự báo chứng khoán</div>', unsafe_allow_html=True)
st.write("Chào mừng bạn đến với Ứng dụng Dự báo Chứng khoán. Sử dụng thanh bên để chọn các tùy chọn và khu vực chính để xem kết quả.")
st.write("")

# Nhập liệu từ người dùng
stock_options = ('BTC-USD', 'ETH-USD', 'ADA-USD')
chosen_stock = st.selectbox('Chọn cặp tiền dự đoán', stock_options)
start_date_input = st.date_input('Chọn ngày bắt đầu', value=date(2018, 1, 1))
current_date = date.today().strftime("%Y-%m-%d")

# Hàm tải dữ liệu
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Tải và hiển thị dữ liệu
loading_state = st.text('Đang tải dữ liệu...')
data = fetch_data(chosen_stock, start_date_input, current_date)
loading_state.text('Đã tải dữ liệu xong!')
st.write(data.head())
st.write(data.tail())

# Chuẩn bị dữ liệu cho mô hình Prophet
df_training = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Dự báo
forecast_years = st.slider('Số năm dự đoán', 1, 4)
forecast_period = forecast_years * 365

model = Prophet()
model.fit(df_training)
future_dates = model.make_future_dataframe(periods=forecast_period)
forecast_data = model.predict(future_dates)

# Hiển thị dữ liệu dự báo
st.subheader('Dự báo dữ liệu')

@st.cache_data(ttl=24*60*60)  # cache trong 24 giờ
def dataframe_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = dataframe_to_csv(data)

st.download_button(
    label="Tải về dữ liệu",
    data=csv_data,
    file_name=f"{chosen_stock}.csv",
    mime="text/csv"
)
st.write(forecast_data)

# Hiển thị biểu đồ dự báo
st.subheader(f'Dự báo trong {forecast_years} năm')
forecast_fig = plot_plotly(model, forecast_data)
st.plotly_chart(forecast_fig)

# Hàm vẽ dữ liệu gốc
def plot_original_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Giá mở"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Giá đóng"))
    fig.layout.update(title_text='Biểu diễn biểu đồ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_original_data()

# Hiển thị các thành phần dự báo
st.subheader("Các thành phần")
components_fig = model.plot_components(forecast_data)
st.write(components_fig)
