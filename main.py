import csv
import numpy as np
import os
import sys
import fnmatch
import pandas as pd
from colorama import Fore, Back, Style
from colorama import init
from termcolor import colored
from datetime import datetime,date,time
import numpy as np
import seaborn as sns
from helper import *
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data_dir = "./data"
all_csv_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(('.csv')):
            print(root,dirs,filename)
            all_csv_files.append(root + '/' + filename)


custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")


def plot_imputed_data_vs_present_data(df,imputed,not_imputed,station_name):
    """
    plot colum wise imputed result and save in in plot folder , file name is like:
    <station_id>_<column_name>.png
    """
    df_temp = df
    #station_name =  list(df["station_id"])[0]
    #df_temp = df_temp.drop(['station_id','callsign','longitude','latitude','qc_flag','tp','qc_flag','thtp','sprtp'],axis = 1, inplace=True)
    for c in df_temp.columns:
        print(colored(f"imputing and plotting for col: {c}", 'blue', 'on_green'))
        plt.suptitle(f'{c} imputed plot at station {station_name}')
        plt.plot(df_temp[c],color = 'r')
        plt.plot(not_imputed[c],color = 'r', marker = 'o')
        plt.plot(imputed[c],linestyle = '',color = 'b',marker = 'd')
        plt.savefig(f'./plots/station_{station_name}_{c}imputation_results.png')
        #plt.show()


def impute_missing_data(df,method = "time"):
    """
    imputing dataframe missing row values using cubic spline as default/ time method
    """
    station_name =  list(df["station_id"])[0]
    na_indexs = df.isnull()
    df.drop(['station_id','time','callsign','longitude','latitude','qc_flag','tp','qc_flag','thtp','sprtp'],axis = 1, inplace=True)
    df = df.interpolate(method=method,axis = 0).ffill().bfill()
    imputed = df[na_indexs]
    not_imputed = df[~na_indexs]
    df_temp = df.copy()
    df_temp["date"] = df_temp.index
    df_temp.to_csv('./imputed_data/'+station_name+'.csv',index=False)
    plot_imputed_data_vs_present_data(df,imputed,not_imputed,station_name)
    return df


def generate_time_series(one_group):
    #print(one_group)
    min_date = one_group.index.min(axis=0)
    max_date = one_group.index.max(axis=0)
    diff_days = max_date - min_date
    diff = diff_days.days
    print(f" min date and max date is : {min_date} & {max_date}")
    print(f" total number of rows found in data is : {len(one_group.index)}")
    print(f"index and diff must mactch if not make new indexes: {diff} , number of missing dates from dataframe is {diff - len(one_group.index)}")

    if diff > 0:
        one_group = one_group.reindex(pd.date_range(min_date,max_date))
        print("this is the len of indexes after merge must match",len(one_group.index))
        print(one_group.head())
        print("missing dates: ",pd.date_range(start = min_date, end = max_date ).difference(one_group.index))

    return one_group


def load_data(station,csv_like):
    print(all_csv_files)
    this_file = [x for x in all_csv_files if csv_like in x][0]
    print(this_file)
    df = pd.read_csv(this_file,parse_dates=["date"],index_col=["date"],date_parser=custom_date_parser)
    df_groups = df.groupby('station_id')
    print("default index created on this columns: ",df.index.names)
    for one_group in df_groups:
        print(colored(f"starting for station {one_group[0]}", 'green', 'on_red'))
        ch = input(colored(f"Press 1 to continue and 0 to exit",'red','on_green'))
        if int(ch) != 1:
            print("EXIT...")
            sys.exit()
        df = generate_time_series(one_group[1])
        df = impute_missing_data(df,method = "time")
        do_train_test_if_model_does_not_exit(df,one_group[0])


if __name__ == "__main__":

    if len(sys.argv) < 4:
        raise("arguments : python main.py m1 p poly hour")
    station = sys.argv[1]
    typ = sys.argv[2]
    impute_method = sys.argv[3]
    csv_like = sys.argv[4]


    this_df = load_data(station,csv_like)



