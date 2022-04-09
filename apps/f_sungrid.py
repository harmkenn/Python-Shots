import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
def app():
    # title of the app
    st.markdown('Get your position from the location of the sun')
    c1,c2 = st.columns((1,3))
    with c1:
        
        setyear = st.number_input('Year: ',1900,2100,2022)
        setmonth = st.number_input('Month: ',1,12,4)
        setday = st.number_input('Day: ',1,31,9)
        sethour = st.number_input('Hour: ',0,23,22)
        setmin = st.number_input('Minute: ',0,59,40)
        setsec = st.number_input('Second: ',0,59,0)
        
        setday = datetime(setyear,setmonth,setday,sethour,setmin,setsec)

        

    with c2:
        if st.button('Now'):
            setday = datetime.now(timezone.utc)
        st.write(str(setday)+'UTC')
        sslon = 180-(setday.hour+setday.minute/60+setday.second/3600)*15+22/60
        st.write(sslon)
        julian = int(setday.strftime('%j')) + (setday.hour+setday.minute/60+setday.second/3600)/24
        a = 360/365.24*(julian - 2)*np.pi/180
        b = (360/365.24*(julian+10)+360/np.pi*.0167*np.sin(a))*np.pi/180
        c = np.sin(-23.44*np.pi/180)*np.cos(b)
        sslat = np.arcsin(c)*180/np.pi - .07
        st.write(julian)
        st.write(sslat)

        
        

    