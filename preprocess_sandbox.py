import pandas as pd
from data import COLUMNS

"""
current file is only used to inspect data and find patterns throught the data, it should be used in actual preprocessing 
"""
    
if __name__ == "__main__":
    data = pd.read_csv("welddb/welddb.data", delim_whitespace=True, header=None,names=COLUMNS)
    #N stands for missing data:
    data=data.replace("N",pd.NA)

    #first idea is to show missing data according to type of weld
    weld_types=data["Type of weld"].unique()
    
    #MMA and ShMA are the same
    data["Type of weld"]=data["Type of weld"].replace("ShMA","MMA")

    #Now trying to find which columns are useless for each weld type
    NO_DATA={} # weld_type:list[feature] where feature is never given for weld_type instance
    #
    missing_values_by_weld_type=pd.DataFrame() #with (index,column) we can see the probability of "index missing" when column is the weld_type
    for weld_type in weld_types:
        df=data[data["Type of weld"]==weld_type]
        missing_values=df.isnull().mean()
        missing_features=missing_values[missing_values==1].index
        NO_DATA[weld_type]=missing_features

        missing_values_by_weld_type[weld_type]=missing_values
    
    # we should be checking at least std, maybe min max etc
    std=missing_values_by_weld_type.std(axis=1)

    breakpoint()

