def prepare_data(df):
    x=df.drop("y",axis=1)
    y=df["y"]
    return x,y