## The code for cleaning the status data and extracting the information relevant for our case 

# The intermediate files are being saved locally for inspection
import pandas as pd 

## Clean the status csv
status = pd.read_csv("../data/status.csv",parse_dates=[3],dayfirst=True)

status['time'] = pd.to_datetime(status['time'])
start_date = pd.to_datetime('09/01/2014') # m/d/y
end_date = pd.to_datetime('09/01/2015') # m/d/y
status = status.loc[(status['time'] > start_date) & (status['time'] < end_date)]

# status.to_csv("new_status.csv")

# #status = pd.read_csv("new_status.csv",parse_dates=[3],dayfirst=True)

status["time"] = pd.to_datetime(status["time"])

avail_bikes = status.groupby([status['time'].dt.date, status['time'].dt.hour,status['station_id']], as_index=False).last()

#avail_bikes.to_csv('rel_data.csv')

#avail_bikes = pd.read_csv("rel_data.csv",parse_dates=[3],dayfirst=True,index_col=0)

avail_bikes['time'] = pd.to_datetime(avail_bikes['time'].astype(str)) + pd.DateOffset(hours=1)
avail_bikes["time"] = pd.to_datetime(avail_bikes["time"]).dt.strftime('%d/%m/%Y %H')
avail_bikes.columns = ['Station_ID','bikes_available', 'docks_available','Time']

avail_bikes.to_csv('avail_bikes.csv')
