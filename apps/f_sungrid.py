import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
def app():
    # title of the app
    st.markdown('Get your position from the location of the sun')
    c1,c2 = st.columns((1,3))
    with c1:
        rightnow = datetime.now(timezone.utc) #Pull current zulu time
        setyear = st.number_input('Year: ',1900,2100,rightnow.year)
        setmonth = st.number_input('Month: ',1,12,rightnow.month)
        setday = st.number_input('Day: ',1,31,rightnow.day)
        sethour = st.number_input('Hour: ',0,23,rightnow.hour)
        setmin = st.number_input('Minute: ',0,59,rightnow.minute)
        setsec = st.number_input('Second: ',0,59,rightnow.second)
        
        setday = datetime(setyear,setmonth,setday,sethour,setmin,setsec)

        

    with c2:
        if st.button('Now'):
            setday = datetime.now(timezone.utc)
        st.write(str(setday)+'UTC')

        
        

    