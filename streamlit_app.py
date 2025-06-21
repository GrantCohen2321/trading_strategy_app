import streamlit as st
import pandas as pd

from trading_strategy_app.trading_strategy_app import run_scan

st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

st.title("Trading Strategy Dashboard")

if st.button("Run Daily Scan"):
    st.info("Running scan...")
    results = run_scan()
    if results is None:
        st.warning("Scan completed, but no results returned.")
    else:
        st.success("Scan completed!")
        st.write(results)
