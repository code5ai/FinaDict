# 📈 Financial Price Prediction

**FINAnce preDICT (FINADICT)** is an interactive **Streamlit** web app that leverages **yFinance**, **Prophet**, **Plotly** and **Streamlit** to fetch, visualize, and forecast market data for **stocks**, **forex**, and **cryptocurrencies**; all packaged in a single **Docker** image. Users can select financial instruments, time intervals, and historical ranges to generate:

* 📈 **Live and historical visualizations** (line and candlestick charts)
* 🔮 **Next-day/hour price predictions** with confidence intervals using Facebook Prophet
* 📥 **Downloadable CSV reports** for raw and predicted data
* 📉 **Forecast accuracy metrics** like Mean Accuracy and RMSPE

---

## 🛠 Installation

### 🐧 In VS Code Termial, with python installed

```bash
git clone https://github.com/subhayu99/finadict.git
cd finadict
pip3 install -r requirements.txt
streamlit run app.py
````
