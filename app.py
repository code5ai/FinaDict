import streamlit as st
from helper_func import (
    load_data, get_company_info, create_raw_data_plots, build_model,
    prepare_training_data, generate_forecast, prepare_forecast_data,
    create_forecast_plot, create_components_plot, calculate_price_metrics,
    get_interval_parameters, generate_csv_content, generate_filename
)


@st.cache_data
def cached_load_data(ticker, period, interval, date_index):
    """Cached wrapper for load_data function"""
    return load_data(ticker, period, interval, date_index)


def download_csv(df, selected_stock, filename):
    """Create download button for CSV"""
    csv_content = generate_csv_content(df)
    csv_filename = generate_filename(selected_stock, filename)
    
    st.download_button(
        label="Press to Download",
        data=csv_content,
        file_name=csv_filename,
        mime="text/csv",
    )


def display_raw_data_plots(data, date_index):
    """Display raw data plots"""
    st.subheader("Raw data plots")
    line_fig, candle_fig = create_raw_data_plots(data, date_index)
    
    with st.expander("Tap to expand/collapse", expanded=True):
        st.plotly_chart(line_fig, use_container_width=True)
        st.plotly_chart(candle_fig, use_container_width=True)


def display_forecast_results(model, forecast, data, periods, df_train, currency, c2, selected_stock):
    """Display forecast results and plots"""
    st.subheader("Forecast data")
    
    forecast_display, next_period_price, price_change_pct = prepare_forecast_data(forecast, df_train, periods)
    
    # Display forecast data table
    display_columns = [
        "Date",
        "Actual Price",
        "Predicted Price",
        "Predicted Price (Lower)",
        "Predicted Price (Upper)",
    ]
    
    with st.expander("Tap to expand/collapse", expanded=True):
        st.write(forecast_display[display_columns].iloc[::-1])
        download_csv(
            forecast_display[display_columns],
            selected_stock.upper(),
            "prediction_data"
        )

    # Next period's price metric
    label = "Next Period's Closing Price"
    value = f"{next_period_price} {currency}"
    delta = f"{price_change_pct}% from current"
    
    with st.spinner("Predicting price..."):
        c2.metric(label=label, value=value, delta=delta)

    # Display forecast plot
    st.subheader("Forecast plot")
    forecast_fig = create_forecast_plot(model, forecast)
    with st.expander("Tap to expand/collapse", expanded=True):
        st.plotly_chart(forecast_fig, use_container_width=True)

    # Display forecast components
    st.subheader("Forecast components")
    components_fig = create_components_plot(model, forecast)
    with st.expander("Tap to expand/collapse", expanded=False):
        st.write(components_fig)


def setup_sidebar():
    """Setup sidebar with market selection and parameters"""
    st.sidebar.image('finadict.png')
    
    menu = ["Stocks", "Forex", "Crypto"]
    choice = st.sidebar.selectbox("Select your market choice", menu)
    
    form = st.sidebar.form("take parameters")
    
    interval_aliases = (
        "5 mins",
        "15 mins",
        "30 mins",
        "1 hour",
        "1 day",
        "1 week",
        "1 month",
    )
    interval_choices = ("5m", "15m", "30m", "60m", "1d", "1wk", "1mo")
    interval_alias = form.radio("Select interval", interval_aliases, index=4)
    interval = interval_choices[interval_aliases.index(interval_alias)]
    
    # Get interval parameters
    params = get_interval_parameters(interval)
    
    # Create slider based on interval
    if interval == "1d":
        slider_value = form.slider(
            params['slider_label'],
            2,
            params['slider_max'],
            value=4,
        )
    else:
        slider_value = form.slider(
            params['slider_label'],
            2,
            params['slider_max'],
            value=15,
        )
    
    # Update parameters with slider value
    updated_params = get_interval_parameters(interval, slider_value)
    
    form.form_submit_button("Submit")
    
    return choice, interval, updated_params


def setup_stock_interface():
    """Setup stock prediction interface"""
    st.title("Stock Prediction")
    selected_stock = st.text_input(
        "Type in a ticker symbol",
        value="TCS.NS",
    )
    if not selected_stock:
        selected_stock = "TCS.NS"
    
    comp_country_code, currency, company_name = get_company_info(selected_stock)
    
    st.write(
        f"<p style='text-decoration:none; font-size:20px'>Showing results for <strong>{company_name}</strong></p>",
        unsafe_allow_html=True,
    )
    
    return selected_stock, comp_country_code, currency


def setup_forex_interface():
    """Setup forex prediction interface"""
    st.title("Forex Prediction")
    col1, col2 = st.columns(2)
    x = col1.text_input("From", value="USD")
    y = col2.text_input("To", value="INR")
    if not x:
        x = "USD"
    if not y:
        y = "INR"
    
    selected_stock = x + y + "=X"
    currency = y.upper()
    comp_country_code = None
    
    st.write(
        f"<p style='text-decoration:none; font-size:20px'>Showing results for <strong>{x.upper()}</strong> to <strong>{y.upper()}</strong> conversion rate.</p>",
        unsafe_allow_html=True,
    )
    
    return selected_stock, comp_country_code, currency


def setup_crypto_interface():
    """Setup crypto prediction interface"""
    st.title("Crypto Prediction")
    selected_stock = st.text_input(
        "Type in a conversion string",
        value="BTC-INR",
    )
    if not selected_stock:
        selected_stock = "BTC-INR"
    
    comp_country_code = None
    parts = selected_stock.split('-')
    x = parts[0] if len(parts) > 0 else "BTC"
    y = parts[1] if len(parts) > 1 else "INR"
    currency = y.upper()
    
    st.write(
        f"<p style='text-decoration:none; font-size:20px'>Showing results for <strong>{x.upper()}</strong> to <strong>{y.upper()}</strong> conversion rate.</p>",
        unsafe_allow_html=True,
    )
    
    return selected_stock, comp_country_code, currency


def main():
    st.set_page_config(
        page_title="FINAnce preDICT",
        page_icon="•",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Setup sidebar and get parameters
    choice, interval, params = setup_sidebar()
    
    menu = ["Stocks", "Forex", "Crypto"]
    
    # Setup interface based on choice
    if choice == menu[0]:
        selected_stock, comp_country_code, currency = setup_stock_interface()
    elif choice == menu[1]:
        selected_stock, comp_country_code, currency = setup_forex_interface()
    elif choice == menu[2]:
        selected_stock, comp_country_code, currency = setup_crypto_interface()

    # Load data
    data = cached_load_data(
        selected_stock, 
        params['period'], 
        interval, 
        params['date_index']
    )
    
    if data is None or len(data) < 2:
        st.error("Not enough data available for prediction. Try a different period/interval.")
        st.stop()

    # Create columns for metrics
    c1, c2 = st.columns(2)

    # Display current price metric
    label, current_price, delta_pct = calculate_price_metrics(data, interval)
    value = f"{current_price} {currency}"
    delta = f"{delta_pct}% change"
    c1.metric(label=label, value=value, delta=delta)

    # Display raw data
    st.subheader("Raw data")
    with st.expander("Tap to expand/collapse", expanded=False):
        st.dataframe(data.iloc[::-1])
        download_csv(data, selected_stock.upper(), "raw_data")
    
    # Display raw data plots
    display_raw_data_plots(data, params['date_index'])

    # Prepare training data
    df_train = prepare_training_data(data, params['date_index'])
    
    # Generate forecast
    with st.spinner("Predicting prices..."):
        model = build_model(comp_country_code)
        forecast = generate_forecast(model, df_train, params['periods'], params['freq'])
        
        # Display forecast results
        display_forecast_results(
            model, forecast, data, params['periods'], 
            df_train, currency, c2, selected_stock
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()