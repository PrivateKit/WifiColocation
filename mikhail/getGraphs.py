import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("json", help="Path to json file containing readings")
parser.add_argument("metric", help="Metric to be returned, either 'jaccard' or 'pearson'")
args = parser.parse_args()

"""
usage 
    Return graphs of changes in Jaccard similarity or Pearson correlation
    based on distance from a point
    
    :-json (str) 
        Path to JSON file to load readings from
    :-metric (str)
        Chosen metric, either Jaccard or Pearson
"""


with open(args.json) as f:
    logs = json.loads(f.read())


def jaccard(macs1, macs2):
    """Returns jaccard similarity between two lists of MAC Addresses"""
    union = list(set(macs1) | set(macs2)) 
    intersection = list(set(macs1) & set(macs2))
    return len(intersection) / len(union)



def get_metrics():
    """Returns list of Jaccard similarities and Pearson correlations relative to first reading"""
    
    loc1_cluster = logs["WIFI_DATA"][0]["scanInfo"]
    loc1_macs = [item['BSSID'] for item in loc1_cluster]

    correlations = []
    jaccards = []

    for i in range(len(logs["WIFI_DATA"])):

        loc2_cluster = logs["WIFI_DATA"][i]["scanInfo"]

        loc2_macs = [item['BSSID'] for item in loc2_cluster]

        intersections = list(set(loc1_macs) & set(loc2_macs))

        jaccards.append(jaccard(loc1_macs, loc2_macs))

        cluster1_values = [item['level'] for item in loc1_cluster if item['BSSID'] in intersections]
        cluster2_values = [item['level'] for item in loc2_cluster if item['BSSID'] in intersections]

        pearson = list(stats.pearsonr(cluster1_values, cluster2_values))

        correlations.append(pearson)


    return [round(x, 3) for x in jaccards], [round(x[0], 3) for x in correlations]


if __name__ == "__main__":
    jaccards, correlations = get_metrics()
    
    if(args.metric == 'jaccard'):
        plt.title('Jaccard')
        plt.plot(jaccards, marker="o")
        plt.ylabel("Jaccard similarity with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, 0, 1])
        plt.show()
    else:
        plt.title('Pearson')
        plt.plot(correlations, marker="o", color="g")
        plt.ylabel("Pearson correlation with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, 0.90, 1])
        plt.show()
    





