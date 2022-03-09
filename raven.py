
import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from PIL import Image 


st.write(""" 
# RAVEN: AIRCRAFT INDICATED DAMAGE PREDICTION APP!
""")
image=Image.open('birdstrike.jpg')
st.image(image, caption='Aircraft/Avian Hazard')
model = pickle.load(open('birdy.pkl', 'rb'))
scaler=pickle.load(open('tweeny.pkl', 'rb'))
enc_loaded=pickle.load(open('Encoda.obj','rb'))
df2= pd.read_csv('Birdstrikesml.csv', low_memory=False)

Risks=df2['RISK'].unique().tolist()
Faaregion=df2['FAAREGION'].unique().tolist()
Airport=df2['AIRPORT'].unique().tolist()
Aircraft=df2['AIRCRAFT'].unique().tolist()
Damage_level=df2['DAMAGE_LEVEL'].unique().tolist() 
Size=df2['SIZE'].unique().tolist()
Warned=df2['WARNED'].unique().tolist()
Phase_of_flight=df2['PHASE_OF_FLIGHT'].unique().tolist()
Type_eng=df2['TYPE_ENG'].unique().tolist() 
Effect=df2['EFFECT'].unique().tolist() 
 





                        
st.sidebar.header('User Input Parameters')
def user_input_features():
    Risk=st.sidebar.selectbox('Risk Level', Risks)
    Faaregions=st.sidebar.selectbox('Region', Faaregion)
    Airports=st.sidebar.selectbox('Airport',Airport)
    Aircrafts=st.sidebar.selectbox('Aircraft', Aircraft)
    Damage_levels=st.sidebar.selectbox('Level of damages',Damage_level )
    Sizes=st.sidebar.selectbox('Bird Size',Size)
    Incident_year=st.number_input('What is the incident year',max_value=2022,min_value=2012,step=1)
    Warning=st.sidebar.selectbox('Warnings', Warned)
    Height=st.number_input('How high is the aircraft',min_value=0)
    Phase_of_flights=st.sidebar.selectbox('Flight Phase',Phase_of_flight )
    Type_engs=st.sidebar.selectbox('Engine Type',Type_eng )
    Effects=st.sidebar.selectbox('Effect',Effect )
    Number_of_Engine=st.number_input('Number_of_engines',1,3)
    Position_of_engines1=st.number_input('Engine_Position',1,7) 
    Incident_day=st.number_input('What is the incident day',max_value=31,min_value=12,step=1)
    Incident_month=st.number_input('What is the incident month',max_value=12,min_value=1,step=1)
    
  
        
        
        


    
    data={'RISK':Risk,
          'FAAREGION':Faaregions,
          'AIRPORT':Airports,
          'AIRCRAFT':Aircrafts,
          'DAMAGE_LEVEL':Damage_levels,
          'SIZE':Sizes, 
          'INCIDENT_YEAR':Incident_year, 
          'WARNED':Warning,
          'PHASE_OF_FLIGHT':Phase_of_flights,
          'HEIGHT':Height,
          'TYPE_ENG':Type_engs,  
          'EFFECT':Effects,
          'NUM_ENGS':Number_of_Engine,
          'ENG_1_POS':Position_of_engines1,
          'INCIDENT_MONTH':Incident_day,
          'INCIDENT_DAY': Incident_month}
    
    features = pd.DataFrame(data, index=[0])
    return features


input_df =user_input_features()
input_df=enc_loaded.transform(input_df)
input_df =scaler.transform(input_df)




if st.button('PREDICT'):
    y_out=model.predict(input_df)                    
    if y_out[0]==1:
        st.write(f'Aircraft shows considerable damage, take precaution immediately')                
    else:
        st.write(f'Aircraft shows no considerable damage, it is safe to keep flying')                

                     
                     


