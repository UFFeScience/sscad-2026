import argparse
import pandas as pd
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()

parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--k",type=int)

args=parser.parse_args()

X=pd.read_csv(args.input)

model=KMeans(
    n_clusters=args.k,
    random_state=0,
    n_init="auto"
)

labels=model.fit_predict(X)

X["cluster"]=labels

X.to_csv(args.output,index=False)