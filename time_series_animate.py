import pandas as pd
import os
import bar_chart_race as bcr


col_of_interest = ['waveheight']#,'waveperiod']


data_dir = "./imputed_data"
all_csv_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(('.csv')):
            print(root,dirs,filename)
            all_csv_files.append(root + '/' + filename)


def make_all_station_date_same(lar_df,fln_largest):
    min_date = lar_df.index.min(axis=0)
    max_date = lar_df.index.max(axis=0)
    diff_days = max_date - min_date
    diff = diff_days.days
    lar_df.drop(lar_df.columns.difference(col_of_interest),1,inplace= True)
    #lar_df['station'] = fln_largest
    lar_df.rename(columns= {'waveheight':fln_largest.split('/')[-1].replace('.csv','')},inplace = True)
    for one in all_csv_files:
        if fln_largest in one:
            continue
        fln = one.split('/')[-1].replace('.csv','')
        one_group = pd.read_csv(one,parse_dates=["date"],index_col=["date"])
        one_group = one_group.reindex(pd.date_range(min_date,max_date))
        one_group = one_group.fillna(0)
        one_group.drop(one_group.columns.difference(col_of_interest),1,inplace= True)
        #one_group['station'] = fln
        one_group.rename(columns= {'waveheight':fln},inplace = True)
        print(one,"this is the len of indexes after merge must match",one_group.head())

        lar_df = pd.merge(lar_df,one_group,left_index = True,right_index = True)

    return lar_df

print(all_csv_files)
largest = -9999999
largest_df = None
fln_largest = ""
for one in all_csv_files:
    df = pd.read_csv(one,parse_dates=["date"],index_col=["date"])
    l = len(df.index)
    fln = one.split('}/')[-1].replace('.csv','')
    if l > largest:
        largest = l
        largest_df = df
        fln_larget = fln

largest_df['station'] = fln_largest
largest_df = make_all_station_date_same(largest_df,fln_larget)
print(largest_df.tail())


cum_sum_df = largest_df.cumsum(axis = 0)
df_temp = cum_sum_df.copy()
df_temp["date"] = df_temp.index
df_temp.to_csv('cum_sum_df.csv',index = False)
#bcr.bar_chart_race(cum_sum_df, sort='asc', steps_per_period=30, figsize=(5, 3))

