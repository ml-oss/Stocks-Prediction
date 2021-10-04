import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet

from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2010-01-01" # start of the data
TODAY = date.today().strftime("%Y-%m-%d") # end of the data


st.title("Stock Prediction App")

stocks = ("AAPL","GOOG","MSFT","GME","TCS.NS","RBLX","BTC-USD","SBIN.NS","TSLA","DOGE-USD") # Stock Name
selected_stocks = st.selectbox("Select the stock",stocks)

n_years = st.slider("Years of prediction:",1,2) # years for prediction in the future 1 year to 4 year

period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True) #puts date as the index column
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name = "Stock Open"))
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name = "Stock Close"))
    fig.layout.update(title_text = "Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# forcasting

df_train = data[["Date","Close"]]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)
st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
