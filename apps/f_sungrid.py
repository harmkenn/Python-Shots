import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
def app():
    # title of the app
    st.markdown('Get your position from the location of the sun')
    c1,c2 = st.columns((1,3))
    with c1:
        now_utc = datetime.now(timezone.utc) # UTC time
        rightnow = datetime.now()
        dt = rightnow.astimezone() # local time
        year = rightnow.year
        if st.button('Now'):
            rightnow = datetime.now()
            year = rightnow.year
        year = st.number_input('Year: ',1900,2100,year)
    with c2:
        st.write(year)
        

    