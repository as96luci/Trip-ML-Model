

import streamlit as st
import pandas as pd
import joblib as jb
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import time



st.title('Trip Time Prediction')

model = jb.load('tripml_model.pkl')
trip_data = pd.read_csv('Trip-Data-for-ML-Model.csv')  

day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,"Friday": 4, "Saturday": 5, "Sunday": 6}

label_encoder = jb.load('label_encoder.pkl')

def time_to_float(t):
    # st.write(f"function {t.hour} : {t.minute/ 60}")
    
    return t.hour + t.minute / 60  + t.second / 3600

def float_to_time(f):

    hours = int(f)

    minutes = int((f - hours) * 60)

    seconds = int((f - hours - minutes / 60) * 3600)

    return time(hours, minutes)



def make_prediction(Operated_Day, Route_Name, Actual_Trip_Start):

        Operated_Day = day_mapping[Operated_Day]
        
        Route_Name = label_encoder.transform([Route_Name])[0]

        Actual_Trip_Start = datetime.strptime(Actual_Trip_Start, '%m/%d/%Y %H:%M').time()
        Actual_Trip_Start = time_to_float(Actual_Trip_Start)

        # st.write(f"{Actual_Trip_Start}")

        trip_data = pd.DataFrame({
            'Actual_Trip_Start': [Actual_Trip_Start],
            'Route_Name': [Route_Name],
            'Operated_Day': [Operated_Day],  
        })

        # st.write(trip_data)
        # st.write(model)

        prediction = model.predict(trip_data)
        return prediction

Operated_Day = st.selectbox('Select Operated Day', list(day_mapping.keys()))
Route_Name = st.selectbox('Select Route Name', trip_data['Route_Name'].unique()) 
Actual_Trip_Start = st.text_input('Enter Actual Start Time (M:D:Y HH:MM)')
Actual_Trip_End = st.text_input('Enter Actual End Time for Evaluation (optional)')


if st.button('Make Prediction'):
    if Operated_Day and Route_Name and Actual_Trip_Start:
        prediction = make_prediction(Operated_Day, Route_Name, Actual_Trip_Start)

        if prediction is not None:

            # st.write(f'Predicted Trip End Time: {prediction[0]}')
            

            if Actual_Trip_End:

                Actual_Trip_End = datetime.strptime(Actual_Trip_End, '%m/%d/%Y %H:%M').time()
                Actual_Trip_End = time_to_float(Actual_Trip_End)
                # st.write(Actual_Trip_End)
                
                # Calculate Mean Absolute Error (MAE)
                mae = mean_absolute_error([Actual_Trip_End], prediction)
                # st.write(f'Mean Absolute Error (MAE): {mae}')
        
                if prediction[0] >Actual_Trip_End:
                    predicted_end_time = float_to_time(prediction[0]-mae)
                else:
                    predicted_end_time = float_to_time(prediction[0]+mae)
                
                st.write(f'Predicted Trip End Time: {predicted_end_time}')
            
            else:
                predicted_end_time = float_to_time(prediction[0])
                st.write(f'Predicted Trip End Time: {predicted_end_time}')
                    
    else:
        st.write('Please fill in all fields.')

    
