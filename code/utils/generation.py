import pandas as pd

def generation(df):
    
    if (df['Generation']<= 76) & (df['Generation']>=58):
        val = 'Boomer'
    elif (df['Generation']<= 57) & (df['Generation']>=42):
        val = 'GenX'
    elif (df['Generation']<= 41) & (df['Generation']>=26):
        val = 'GenY'
    elif (df['Generation']<= 25) & (df['Generation']>=10):
        val = 'GenZ'
    else:
        val = 'Unknown'
    return val