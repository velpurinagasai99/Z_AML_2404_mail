import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data
data=load_model()
regressor_loaded=data['model']
le_Geo=data['le_Geo']
le_Inc=data['le_Inc']
le_Var=data['le_Var']

def show_predict_page():
    st.title("Poppulation Prediction")
    st.write("""### We need some information to predict the Population""")
    GeoRegions=('Western Asia', 'Western Africa', 'Eastern Africa', 'Central Asia',
       'Eastern Asia', 'Caribbean', 'Middle Africa', 'Southern Asia',
       'Eastern Europe', 'Central America', 'Northern Africa',
       'South-eastern Asia', 'South America', 'Melanesia',
       'Southern Africa', 'Micronesia', 'Southern Europe',
       'Western Europe', 'Northern Europe', 'Australia and New Zealand',
       'Northern America')
    IncomeGroup=('Low income', 'Lower middle income', 'Upper middle income',
       'High income')
    Elevation=('Elevation under 5 meters', 'Elevation between 50 and 100 meters',
       'Elevation between 800 and 1500 meters',
       'Elevation between 25 and 50 meters',
       'Elevation between 10 and 25 meters',
       'Elevation between 200 and 400 meters',
       'Elevation between 100 and 200 meters',
       'Elevation between 5 and 10 meters',
       'Elevation between 1500 and 3000 meters',
       'Elevation between 400 and 800 meters',
       'Elevation between 3000 and 5000 meters')
    
    region=st.selectbox('GeoRegions',GeoRegions)
    inc=st.selectbox('Income Group',IncomeGroup)
    ele=st.selectbox('Evelation Range',Elevation)
    area=st.slider("Land Area",10,40000,500)
    sel=st.button("Calculate Population")
    if sel:
        x=np.array([[region,inc,ele,area]])
        x[:,0]=le_Geo.transform(x[:,0])
        x[:,1]=le_Inc.transform(x[:,1])
        x[:,2]=le_Var.transform(x[:,2])
        x=x.astype(float)
        population=regressor_loaded.predict(x)
        st.subheader(f"The estimated population is ${population[0]:.2f}")

