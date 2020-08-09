import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import math
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("json", help="Path to json file containing readings")
parser.add_argument("metric", help="Metric to be returned, either 'jaccard' or 'pearson'")
args = parser.parse_args()

"""
usage 
    Return graphs of changes in Jaccard similarity, Pearson correlation, "Das" feature, or "New" Feature
    based on distance from a point
    
    python getGraphs.py <json> <metric>

    :-json (str) 
        Path to JSON file to load readings from
    :-metric (str)
        Chosen metric, either 'Jaccard', 'Pearson', 'Das', or 'New'
"""


with open(args.json) as f:
    logs = json.loads(f.read())


def jaccard(macs1, macs2):
    """Returns jaccard similarity between two lists of MAC Addresses"""
    union = list(set(macs1) | set(macs2)) 
    intersection = list(set(macs1) & set(macs2))
    return len(intersection) / len(union)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_metrics():
    """Returns list of chosen metric relative to first reading"""
    
    loc1_cluster = logs["WIFI_DATA"][0]["scanInfo"]
    loc1_macs = [item['BSSID'] for item in loc1_cluster]

    correlations = []
    jaccards = []
    das_feature_list = []
    new_feature_list = []

    for i in range(len(logs["WIFI_DATA"])):

        loc2_cluster = logs["WIFI_DATA"][i]["scanInfo"]
        loc2_macs = [item['BSSID'] for item in loc2_cluster]
        intersections = list(set(loc1_macs) & set(loc2_macs))

        jaccard1 = jaccard(loc1_macs, loc2_macs)
        jaccards.append(jaccard1)

        cluster1_values = [item['level'] for item in loc1_cluster if item['BSSID'] in intersections]
        cluster2_values = [item['level'] for item in loc2_cluster if item['BSSID'] in intersections]


        #Das feature calculation
        adjusted_cluster1 = [item - cluster1_values[0] for item in cluster1_values]
        adjusted_cluster2 = [item - cluster2_values[0] for item in cluster2_values]
        signal_strength = [item1 - item2 for (item1, item2) in zip(adjusted_cluster1, adjusted_cluster2)]
        gain = sum(signal_strength) / len(intersections)
        if(gain == 0):
            gain = 1
        das_feature = jaccard1 / gain
        das_feature_list.append(das_feature)

        #pearson correlation
        pearson = list(stats.pearsonr(cluster1_values, cluster2_values))
        correlations.append(pearson)

        #new feature calculation
        new_feature = jaccard1 / (1 - pearson[0])

        if(new_feature == np.inf):
            new_feature = 256
        new_feature_list.append(new_feature)

    return [round(x, 3) for x in jaccards], [round(x[0], 3) for x in correlations], [round(x, 3) for x in das_feature_list], [round((x / new_feature_list[1]), 3) for x in new_feature_list], len(logs["WIFI_DATA"])


if __name__ == "__main__":
    jaccards, correlations, das_features, new_features, length = get_metrics()
    
    if(args.metric == 'Jaccard'):
        plt.title('Jaccard')
        plt.plot(jaccards, marker="o")
        plt.ylabel("Jaccard similarity with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, 0, 1])
        plt.show()
    elif(args.metric == 'Pearson'):
        plt.title('Pearson')
        plt.plot(correlations, marker="o", color="g")
        plt.ylabel("Pearson correlation with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, 0.90, 1])
        plt.show()
    elif(args.metric == 'Das'):
        plt.title('Das')
        plt.plot(das_features, marker="o", color="c")
        plt.ylabel("Das with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, -1, 1])
        plt.show()
    elif(args.metric == 'New'):
        plt.title('New')
        plt.plot(new_features, marker="o", color="c")
        plt.ylabel("New Feature with reading at starting point")
        plt.xlabel("Distance in ft from AP")
        plt.axis([0, 26, 0, 1])
        plt.show()





