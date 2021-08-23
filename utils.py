# Function Definition Section
def clean_dataset(df):
    df = df.dropna()
    return df

def printfo(*arg):
    print("[*] %s" % ' '.join([str(s) for s in arg]))