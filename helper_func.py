from datetime import datetime
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pycountry
import pandas as pd


def load_data(ticker, period, interval, date_index):
    """Load stock/forex/crypto data from Yahoo Finance"""
    comp = yf.Ticker(ticker)
    data = comp.history(period=period, interval=interval)
    if data is None or data.empty:
        return None
    data.reset_index(inplace=True)
    
    # Convert datetime to timezone-naive format
    if date_index == "Datetime":
        if pd.api.types.is_datetime64_any_dtype(data[date_index]):
            data[date_index] = data[date_index].dt.tz_localize(None)
    
    return data.bfill().ffill()


def get_company_info(ticker):
    """Get company information and currency details"""
    comp = yf.Ticker(ticker)
    comp_info = comp.info
    
    # Get country code for holidays
    try:
        comp_country_code = pycountry.countries.search_fuzzy(comp_info.get("country", ""))[0].alpha_2
    except:
        comp_country_code = None
    
    currency = comp_info.get("financialCurrency", "")
    company_name = comp_info.get('longName', ticker)
    
    return comp_country_code, currency, company_name


def create_raw_data_plots(data, date_index):
    """Create line chart and candlestick chart for raw data"""
    # Line chart
    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(
            x=data[date_index],
            y=data["Open"],
            name="opening_price",
            line=dict(color="#0000ff"),
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=data[date_index],
            y=data["Close"],
            name="closing_price",
            line=dict(color="#ff0000"),
        )
    )
    line_fig.layout.update(
        title_text="Time Series data in Line chart",
        xaxis_rangeslider_visible=True,
        hovermode="x",
    )
    line_fig.update_yaxes(title_text="Price Range")
    line_fig.update_xaxes(title_text="Date")

    # Candlestick chart
    candle_fig = go.Figure(
        data=[
            go.Candlestick(
                x=data[date_index],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    candle_fig.layout.update(
        title_text="Time Series data in Candle-sticks chart",
        xaxis_rangeslider_visible=True,
        hovermode="x",
    )
    candle_fig.update_yaxes(title_text="Price Range")
    candle_fig.update_xaxes(title_text="Date")

    return line_fig, candle_fig


def build_model(comp_country_code):
    """Build Prophet forecasting model"""
    m = Prophet(
        interval_width=0.95,
        daily_seasonality=True,
        changepoint_prior_scale=1,
    )
    if comp_country_code:
        m.add_country_holidays(country_name=comp_country_code)
    return m


def prepare_training_data(data, date_index):
    """Prepare data for training the Prophet model"""
    df_train = data[[date_index, "Close"]]
    df_train = df_train.rename(columns={date_index: "ds", "Close": "y"})
    
    # Convert ds column to datetime and remove timezone
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    if df_train['ds'].dt.tz is not None:
        df_train['ds'] = df_train['ds'].dt.tz_convert(None)
    
    return df_train


def generate_forecast(model, df_train, periods, freq):
    """Generate forecast using the trained model"""
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def prepare_forecast_data(forecast, df_train, periods):
    """Prepare forecast data for display"""
    original = df_train["y"]
    prediction = forecast["yhat"][:-periods]

    forecast_display = forecast.copy()
    
    # Extend actual prices with NaN for future periods
    actual_prices = list(df_train["y"]) + [np.nan] * periods
    forecast_display["Actual Price"] = actual_prices[:len(forecast_display)]
    
    forecast_display["Date"] = forecast_display["ds"].astype(str)
    forecast_display["Predicted Price"] = forecast_display["yhat"]
    forecast_display["Predicted Price (Lower)"] = forecast_display["yhat_lower"]
    forecast_display["Predicted Price (Upper)"] = forecast_display["yhat_upper"]

    # Get next period's prediction
    next_period_price = round(forecast_display["Predicted Price"].iloc[-1], 4)
    if next_period_price > 99:
        next_period_price = round(next_period_price, 2)
    
    # Use the last actual price from the training data (not from forecast_display)
    current_price = df_train["y"].iloc[-1]
    
    # Calculate price change percentage with error handling
    if pd.isna(current_price) or current_price == 0:
        price_change_pct = 0.0
    else:
        price_change_pct = round(((next_period_price - current_price) / current_price) * 100, 2)

    return forecast_display, next_period_price, price_change_pct


def create_forecast_plot(model, forecast):
    """Create forecast plot using Prophet's built-in plotting"""
    fig = plot_plotly(model, forecast)
    fig.layout.update(xaxis_rangeslider_visible=True, hovermode="x")
    fig.update_yaxes(title_text="Price Range")
    fig.update_xaxes(title_text="Date")
    return fig


def create_components_plot(model, forecast):
    """Create forecast components plot"""
    return model.plot_components(forecast)


def calculate_price_metrics(data, interval):
    """Calculate current and previous price metrics"""
    if interval == "1d":
        label = "Today's Closing Price"
    else:
        label = "Latest Closing Price"
    
    current_price = round(data["Close"].iloc[-1], 4)
    if current_price > 99:
        current_price = round(current_price, 2)
    
    prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
    delta_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price else 0
    
    return label, current_price, round(delta_pct, 2)


def get_interval_parameters(interval, slider_value=None):
    """Get parameters based on selected interval"""
    if interval == "5m":
        return {
            'period': f'{slider_value or 15}d',
            'date_index': 'Datetime',
            'periods': 1,
            'freq': '5min',
            'slider_max': 60,
            'slider_label': 'No. of days\' data to fetch:'
        }
    elif interval == "15m":
        return {
            'period': f'{slider_value or 15}d',
            'date_index': 'Datetime',
            'periods': 1,
            'freq': '15min',
            'slider_max': 60,
            'slider_label': 'No. of days\' data to fetch:'
        }
    elif interval == "30m":
        return {
            'period': f'{slider_value or 15}d',
            'date_index': 'Datetime',
            'periods': 1,
            'freq': '30min',
            'slider_max': 60,
            'slider_label': 'No. of days\' data to fetch:'
        }
    elif interval == "60m":
        return {
            'period': f'{slider_value or 15}d',
            'date_index': 'Datetime',
            'periods': 1,
            'freq': 'H',
            'slider_max': 60,
            'slider_label': 'No. of days\' data to fetch:'
        }
    elif interval == "1d":
        months = slider_value or 4
        return {
            'period': f'{months * 30}d',
            'date_index': 'Date',
            'periods': 1,
            'freq': 'D',
            'slider_max': 12,
            'slider_label': 'No. of months\' data to fetch:'
        }
    elif interval == "1wk":
        return {
            'period': f'{slider_value or 15}y',
            'date_index': 'Date',
            'periods': 1,
            'freq': 'W',
            'slider_max': 20,
            'slider_label': 'No. of years\' data to fetch:'
        }
    elif interval == "1mo":
        return {
            'period': f'{slider_value or 15}y',
            'date_index': 'Date',
            'periods': 1,
            'freq': 'M',
            'slider_max': 20,
            'slider_label': 'No. of years\' data to fetch:'
        }


def generate_csv_content(df):
    """Generate CSV content for download"""
    return df.to_csv(index=False).encode("utf-8")


def generate_filename(selected_stock, data_type):
    """Generate filename with timestamp"""
    now = datetime.now().strftime("%d%m%Y%H%M%S")
    return f"{selected_stock} {data_type} {now}.csv"