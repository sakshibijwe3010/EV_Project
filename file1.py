import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Loading the CSV File
EV=pd.read_csv(r"C:\Users\VICTUS\OneDrive\Desktop\Ev_vehicle project\EV_Dataset.csv")
EV
EV.head()
EV.isna().sum()
EV.describe()
EV.duplicated().sum()
# Assigning Index Name & Starts from 1
EV.index.name = "Id"
EV.index = EV.index+1
EV.info()
# Checing data types
EV.dtypes
# Change Year Column data type 
EV.Year = EV.Year.astype(int)
# The data in the Month_Name column first letter capitalized for better view
EV.Month_Name = EV.Month_Name.str.capitalize()
# Date colum datetype changes into datetime format
EV.Date = pd.to_datetime(EV.Date,format="%m/%d/%Y")
EV.Date
# Change dateype of EV_Sales Quantity Column
EV.EV_Sales_Quantity = EV.EV_Sales_Quantity.astype(int)
EV
#Visulization of data
# Create function for unique value Checking
def unique_value_checking(EV_column):
    print("List of unique values : " ,EV_column.unique())
    print("No of unique values : ", len(EV_column.unique()))
# Year Column checking
unique_value_checking(EV.Year)
# Month_Name Column Checking if there Unusual Data
unique_value_checking(EV.Month_Name)
# Date Column Uniqueness Cheking
unique_value_checking(EV.Date)
# State Column Cheking
unique_value_checking(EV.State)
unique_value_checking(EV.Vehicle_Class)
# Cheking Vehicle_Type	Column
unique_value_checking(EV.Vehicle_Type)
# Checking Vehicle_Category	Column
unique_value_checking(EV.Vehicle_Category)
# Checking EV_Sales_Quantity Column
unique_value_checking(EV.EV_Sales_Quantity)
unique_columns_counts = []
for i in EV.columns:
    unique_columns_counts.append(len(EV[i].unique()))
# Bar Graph  for unique Value checking
plt.figure(figsize = (12,6))
sns.set_style("white")
bars = plt.bar(EV.columns,unique_columns_counts)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             str(bar.get_height()), ha='center', va='bottom', fontsize=9)
plt.title("Colums_unique_Values", fontweight = "bold")
plt.xlabel("Columns_Value")
plt.ylabel("Value_Counts")
plt.show()
# After checking uniqueness, the Vehicle_Type column does not need to be categorized.
EDA_df = EV.drop(columns = "Vehicle_Type")
# Drop rows where EV_Sales_Quantity is 0 to ensure accurate analysis.
list_of_row_deletion = EDA_df.EV_Sales_Quantity[EDA_df.EV_Sales_Quantity == 0].index
EDA_df_filtered = EDA_df.drop(list_of_row_deletion)
# Function for reset index and assign indes name and 
def reset_index(df):
    df = EV.reset_index(drop = True)
    EV.index = EV.index+1
    EV.index.name = "Id"
    return EV
# Reset the index
EDA_df_filtered = reset_index(EDA_df_filtered)
EDA_df_filtered
# we are going to analyse the data from the EDA_df_filtered
EV = EDA_df_filtered.copy()
EV
EV.EV_Sales_Quantity.sum()
year_wise_sold = EV[["Year","EV_Sales_Quantity"]].groupby(by = "Year").sum()
print(year_wise_sold)
sns.set_style("whitegrid")
plt.figure(figsize = (12,6))
bars = plt.bar(year_wise_sold.index,year_wise_sold.EV_Sales_Quantity, color = "blue")
plt.title("Total EV vehicles sold year-wise", fontweight = 'bold')
plt.xlabel("Year")
plt.ylabel("Quantity")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             str(bar.get_height()), ha='center', va='bottom', fontsize=9)
plt.show()
month_wise_sold = EV[["Month_Name","EV_Sales_Quantity"]].groupby( by = "Month_Name").sum().sort_values(by = "EV_Sales_Quantity" , ascending =False)
print(month_wise_sold)
sns.set_style("whitegrid")
plt.figure(figsize = (12,6))
bars = sns.barplot(x = month_wise_sold.index,y = month_wise_sold.EV_Sales_Quantity, data = month_wise_sold, palette = "viridis" )
plt.title("Total EV vehicles sold month-wise", fontweight = 'bold')
plt.xlabel("Month_Name")
plt.ylabel("Quantity")
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.0f}', 
                (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()
date_wise_sold = EV[["Date","EV_Sales_Quantity"]].groupby("Date").sum()
print(date_wise_sold)
sns.set_style("darkgrid")
plt.figure(figsize = (12,6))
sns.lineplot(x = date_wise_sold.index, y = date_wise_sold.EV_Sales_Quantity, data = date_wise_sold, marker = ".", linewidth = 2)
plt.title("Date wise EV_Sale Trent", fontweight = 'bold')
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.grid (True)
plt.show()
State_wise_sold = EV[["State","EV_Sales_Quantity"]].groupby("State").sum().sort_values("EV_Sales_Quantity", ascending = False)
print(State_wise_sold)
Vehicle_Class_wise_sold = EV[["Vehicle_Class","EV_Sales_Quantity"]].groupby("Vehicle_Class").sum().sort_values("EV_Sales_Quantity", ascending = False)
print(Vehicle_Class_wise_sold)
Vehicle_Category_wise_sold = EV[["Vehicle_Category","EV_Sales_Quantity"]].groupby("Vehicle_Category").sum()
print(Vehicle_Category_wise_sold)
plt.figure(figsize = (7,6))
colors = sns.color_palette("pastel")
plt.pie(Vehicle_Category_wise_sold.EV_Sales_Quantity, labels=Vehicle_Category_wise_sold.index,
        autopct='%1.1f%%', colors = colors , startangle = -20, wedgeprops={'edgecolor': 'black'})
plt.title("Vechicle Category Pie Chart", fontweight = "bold")
plt.legend(Vehicle_Category_wise_sold.index, loc="upper right", bbox_to_anchor=(1.2, 1))
forecast_df = EV[["Year","EV_Sales_Quantity"]].groupby("Year").sum().reset_index()
forecast_df.Year = pd.to_datetime(forecast_df.Year, format = "%Y")
forecast_df = forecast_df.rename(columns={"Year" : "ds", "EV_Sales_Quantity" : "y"})
forecast_df
forecast_df = EV[["Year","EV_Sales_Quantity"]].groupby("Year").sum().reset_index()
forecast_df.Year = pd.to_datetime(forecast_df.Year, format = "%Y")
forecast_df = forecast_df.rename(columns={"Year" : "ds", "EV_Sales_Quantity" : "y"})
forecast_df
EV= EDA_df[["Vehicle_Class","EV_Sales_Quantity"]]
EV
State_pivot = EDA_df.pivot_table(index = "State", columns = "Year", values = "EV_Sales_Quantity", aggfunc = sum )
plt.figure(figsize = (10,10))
sns.heatmap(State_pivot, annot=True, fmt=".0f")
plt.show()
print(State_wise_sold.describe())
plt.figure(figsize = (6,5))
sns.heatmap(State_wise_sold.tail(10),annot=True,cmap="coolwarm",linewidths=0.5,fmt=".0f")
plt.title("Key Area to moitor",fontweight = "bold")
plt.show()
# import Libraries
#build regression model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Convert categorical variable to numeric using encoding

df = pd.get_dummies(EDA_df, columns = ["State", "Vehicle_Class", "Vehicle_Category"],drop_first = True)

# future & target variables

X = df.drop(columns = ["Month_Name", "Date", "EV_Sales_Quantity"])  # independent variable

y = df["EV_Sales_Quantity"] # dependent variable

# split the data to taraining & testing 80% train, 20% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Train the model

model = LinearRegression() # Initialize
model.fit(X_train, y_train) # Train
# Predict sales

y_pred = model.predict(X_test)
y_pred
sns.set_style("darkgrid")
plt.figure(figsize = (10,10))
sns.scatterplot( x = y_test, y= y_pred)
sns.lineplot( x = y_test, y = y_test, color = 'red',linestyle = 'dashed')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Regression Model",fontweight = "bold")
plt.show()
# Calculate errors
mae = float(mean_absolute_error(y_test, y_pred))
mse = float(mean_squared_error(y_test, y_pred))
rmse = float(np.sqrt(mse))
r2 = float((r2_score(y_test, y_pred)))

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

Ev = pd.DataFrame ({"Error" : ["mae", "mse", "rmse", "r2"],"Values" : [mae, mse, rmse, r2]})
sns.heatmap(Ev.set_index("Error")[["Values"]], cmap="plasma", annot=True, fmt=".5f")
plt.show()

plt.show()
forecast_df = EDA_df[["Year","EV_Sales_Quantity"]].groupby("Year").sum().reset_index()
forecast_df.Year = pd.to_datetime(forecast_df.Year, format = "%Y")
forecast_df = forecast_df.rename(columns={"Year" : "ds", "EV_Sales_Quantity" : "y"})
forecast_df
#evalution the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

# Prepare the data for modeling
X = EV.drop(columns=['EV_Sales_Quantity'])
X = pd.get_dummies(X, drop_first=True)
y = EV['EV_Sales_Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()   

model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mae
