# import numpy as np
# import pandas as pd
# import streamlit as st
# st.title("This is my first streamlit application---!")
# st.image(r"C:\Users\VICTUS\OneDrive\Desktop\Ev_vehicle project\e_vehicle.jpg")
# df=pd.read_csv(r"C:\Users\VICTUS\Downloads\EV_Dataset.csv")
# st.write(EV)
# st.line_chart(EV)
import numpy as np
import pandas as pd
import streamlit as st

# Title
st.title("🚗 Electric Vehicle Analysis Dashboard")

# Show image
st.image("e_vehicle.jpg", caption="Electric Vehicle Charging", use_container_width=True)

# Load dataset
df = pd.read_csv("EV_Dataset.csv")

# Display dataset
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Line chart
st.subheader("📈 EV Sales Over Time")
st.line_chart(df.set_index(df.columns[0]))
