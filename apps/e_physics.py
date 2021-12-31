import streamlit as st
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

def app():
    # title of the app
    st.markdown('Artillery Physics')
    