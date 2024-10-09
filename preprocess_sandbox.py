import pandas as pd
from data import COLUMNS,TARGET_FEATURES

"""
current file is only used to inspect data and find patterns throught the data, it should not be used in actual preprocessing 
"""
    
data = pd.read_csv("welddb/welddb.data", delim_whitespace=True, header=None,names=COLUMNS)
#MMA and ShMA are the same
data["Type of weld"]=data["Type of weld"].replace("ShMA","MMA")
#N stands for missing data:
data=data.replace("N",pd.NA)
def weld_type_to_rename(columns:list[str]):
    selected_data=data[columns+["Type of weld"]]
    #first idea is to show missing data according to type of weld
    weld_types=selected_data["Type of weld"].unique()

    #Now trying to find which columns are useless for each weld type
    no_data={} # weld_type:list[feature] where feature is never given for weld_type instance
    #
    missing_values_by_weld_type=pd.DataFrame() #with (index,column) we can see the probability of "index missing" when column is the weld_type
    for weld_type in weld_types:
        df=selected_data[selected_data["Type of weld"]==weld_type]
        missing_values=df.isnull().mean()
        missing_features=missing_values[missing_values==1].index
        no_data[weld_type]=missing_features

        missing_values_by_weld_type[weld_type]=missing_values

    # we should be checking at least std, maybe min max etc
    return(missing_values_by_weld_type,no_data)


if __name__ == "__main__":
    missing_values_by_weld_type,no_data=weld_type_to_rename(columns=TARGET_FEATURES)
    breakpoint()