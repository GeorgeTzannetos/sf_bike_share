import pandas as pd 
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


trips = pd.read_csv("../data/trip_data.csv",parse_dates=[1,3],dayfirst=True)
weather = pd.read_csv("../data/weather_data.csv")
stations = pd.read_csv("../data/station_data.csv")


dates = pd.date_range(start="9/1/2014",end="9/1/2015",freq="H")
weekdays = dates.weekday
only_date = dates.date

#Find business days and holidays
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=dates.date.min(), end=dates.date.max())

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
business_days = pd.DatetimeIndex(start=dates.date.min(), end=dates.date.max(), freq=us_bd)

business_days = pd.to_datetime(business_days, format='%d/%m/%Y').date
holidays = pd.to_datetime(holidays, format='%d/%m/%Y').date


date_df = pd.DataFrame(dates,columns=["Time"])
date_df["Weekday"] = weekdays
date_df["Date"] = only_date

# Add the business and holidays 

date_df['Business_day'] = date_df['Date'].isin(business_days)
date_df['Holiday'] = date_df['Date'].isin(holidays)

date_df.Business_day = date_df.Business_day.map(lambda x: 1 if x == True else 0)
date_df.Holiday = date_df.Holiday.map(lambda x: 1 if x == True else 0)

# Change format 
dates = pd.to_datetime(dates).strftime('%d/%m/%Y %H')
only_date = pd.to_datetime(only_date).strftime('%d/%m/%Y')
# Replace with correct format
date_df["Date"] = only_date
date_df["Time"] = dates

## Match the location of the stations with the zip of the weather data
# {94107: 'San Francisco', 94063: 'Redwood City', 94301: 'Palo Alto', 94041: 'Mountain View', 95113: 'San Jose'}

stations["Zip"] = np.nan
stations.loc[stations['City'] == 'San Jose','Zip'] = 95113
stations.loc[stations['City'] == 'Redwood City','Zip'] = 94063
stations.loc[stations['City'] == 'San Francisco','Zip'] = 94107
stations.loc[stations['City'] == 'Palo Alto','Zip'] = 94301
stations.loc[stations['City'] == 'Mountain View','Zip'] = 94041 

# Concatenate all the stations together for every day and hour of the desired time range
dates_df = pd.concat([date_df]*len(stations.index), ignore_index=True)

## List of indices and list of stations
loi = np.arange(0,674597,8761)

station_ids = []
dock_count = []
zip_loc = []
for _,row in stations.iterrows():
    
    station_ids.append(row["Id"])
    dock_count.append(row["Dock Count"])
    zip_loc.append(row["Zip"])

for i in range(len(loi)-1):
    dates_df.loc[loi[i]:loi[i+1]-1, "Station_ID"] = station_ids[i]
    dates_df.loc[loi[i]:loi[i+1]-1, "Dock_count"] = dock_count[i]
    dates_df.loc[loi[i]:loi[i+1]-1, "Zip"] = zip_loc[i]


trips_dates = pd.to_datetime(trips["Start Date"])
start_dates = trips_dates.apply(lambda x: x.strftime('%d/%m/%Y %H'))
trips["New_dates"]= start_dates
#print(start_dates)

start = trips["Start Station"].groupby(trips['New_dates']).value_counts().to_frame()
start['Time'] = start.index.get_level_values('New_dates') 
start['Station_ID'] = start.index.get_level_values('Start Station') 
start.columns = ['Start_CountTrips', 'Time','Station_ID']
start.reset_index(drop=True,inplace=True)
#print(test)

merged = pd.merge(dates_df,start,on=["Time","Station_ID"],how="outer")
merged['Start_CountTrips'].fillna(0,inplace=True)
#print(merged)

#Calculate the same way the end trips 
trips_copy = trips.copy()
stop_trips_dates = pd.to_datetime(trips_copy["End Date"])
end_dates = stop_trips_dates.apply(lambda x: x.strftime('%d/%m/%Y %H'))
#print(end_dates)
trips_copy["New_dates"]= end_dates

stop = trips_copy["End Station"].groupby(trips_copy['New_dates']).value_counts().to_frame()
stop['Time'] = stop.index.get_level_values('New_dates') 
stop['Station_ID'] = stop.index.get_level_values('End Station') 
stop.columns = ['Stop_CountTrips', 'Time','Station_ID']
stop.reset_index(drop=True,inplace=True)


final_merged = pd.merge(merged,stop,on=["Time","Station_ID"],how="outer")
final_merged['Stop_CountTrips'].fillna(0,inplace=True)

final_merged["Net_Rate"] = final_merged["Stop_CountTrips"] - final_merged["Start_CountTrips"]

# Match weather data with the location and the date of the stations 
final_merged_weather = pd.merge(final_merged,weather,on=["Date","Zip"],how="outer")

## Add the available bikes at each station at the beginning on the hour
avail_bikes = pd.read_csv("avail_bikes.csv",parse_dates=[3], dayfirst=True,index_col=0)
final_merged_weather_avail = pd.merge(final_merged_weather,avail_bikes,on=["Time","Station_ID"],how="outer")


# Drop the data samples with dates 1/9/2015 and then date column 
final_merged_weather_avail.drop(final_merged_weather_avail[final_merged_weather_avail['Date'] == '01/09/2015'].index, inplace=True)
final_merged_weather_avail.drop(columns=['Date'],inplace=True)

print(final_merged_weather_avail)

#Save train dataset 
final_merged_weather_avail = final_merged_weather_avail.fillna(0)
final_merged_weather_avail.to_csv("net_change_weather_avail.csv")
