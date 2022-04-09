import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
def app():
    # title of the app
    st.markdown('Get your position from the location of the sun')
    c1,c2 = st.columns((1,3))
    with c1:
        now_utc = datetime.now(timezone.utc) # UTC time
        rightnow = datetime.now(timezone.utc)
        dt = rightnow.astimezone() # local time
        year = rightnow.year
        yeard = st.number_input('Year: ',1900,2100,year)
        setday = rightnow - timedelta(days=365*(year - yeard))
    with c2:
        st.write(str(setday)+'UTC')
        

    