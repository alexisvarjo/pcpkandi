import refinitiv.data as rd
import pandas as pd
import calendar
from datetime import datetime
from datetime import timedelta
import pandas_market_calendars as mcal

session = rd.open_session()

print(session)

def get_risk_free_rates(start_date, ric_list1, ric_list2):
    end_date = datetime.now().strftime('%Y-%m-%d')

    final_data = pd.DataFrame()

    for ric in ric_list1:
        print(ric)
        data = rd.get_history(ric,
                                interval='daily',
                                start=start_date,
                                end=end_date,
                                fields=['FIXING_1'])
        final_data[ric] = data["FIXING_1"]

    for ric in ric_list2:
        print(ric)
        data = rd.get_history(ric,
                                interval='daily',
                                start=start_date,
                                end=end_date,
                                fields=['ZERO_YLD1'])
        final_data[ric] = data["ZERO_YLD1"]

    final_data.fillna(method='ffill', inplace=True)

    print(final_data)
    return final_data


list1 = ["OINOKSWD=", "OINOK1MD=", "OINOK2MD=", "OINOK3MD=", "OINOK6MD=",
    "STISEKTNDFI=", "STISEK1WDFI=", "STISEK1MDFI=", "STISEK2MDFI=", "STISEK3MDFI=", "STISEK6MDFI=",
    "CIDKKSWD=", "CIDKK1MD=", "CIDKK3MD=", "CIDKK6MD=", "CIDKK1YD=", 'CIDKK3MD=']

list2 = ["NOKABQOD1YZ=R", "NOKONZ=R", "DKKONZ=R", "DKK2MZ=R", "DKK9MZ=R", "DKKABQCD1Y3MZ=R", "SEGOV1Y3MZ=R", "SEGOV1YZ=R", "SEK9MZ=R",
            "NOK9MZ=R", "NOK1YZ=R", "NOK1Y3MZ=R"]

data = get_risk_free_rates("2010-01-01", list1, list2)

column_order = ['NOKONZ=R','OINOKSWD=', 'OINOK1MD=', 'OINOK2MD=', 'OINOK3MD=', 'OINOK6MD=', 'NOK9MZ=R', "NOK1YZ=R",'NOK1Y3MZ=R',
    'STISEKTNDFI=', 'STISEK1WDFI=', 'STISEK1MDFI=', 'STISEK2MDFI=', 'STISEK3MDFI=', 'STISEK6MDFI=', 'SEK9MZ=R', 'SEGOV1YZ=R','SEGOV1Y3MZ=R',
    'DKKONZ=R','CIDKKSWD=', 'CIDKK1MD=', 'DKK2MZ=R', 'CIDKK6MD=','DKK9MZ=R', 'CIDKK1YD=','DKKABQCD1Y3MZ=R']

data = data.reindex(columns=column_order)
print(data)


data.to_csv('risk_free_rates2.csv')
