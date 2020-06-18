#co location
import json
import numpy as np
import pandas as pd
from scipy import stats

with open('./edited_june_17.json') as f:
    new_logs = json.loads(f.read())


def find_correlations(wifi_logs):
    correlations = []
    for i in range(len(wifi_logs) - 1):
        cluster1 = wifi_logs[i]
        cluster2 = wifi_logs[i + 1]

        cluster1_macs = [item['BSSID'] for item in cluster1]
        cluster2_macs = [item['BSSID'] for item in cluster2]

        intersections = list(set(cluster1_macs) & set(cluster2_macs))

        cluster1_values = [item['level'] for item in cluster1 if item['BSSID'] in intersections]
        cluster2_values = [item['level'] for item in cluster2 if item['BSSID'] in intersections]

        correlations.append(list(stats.pearsonr(cluster1_values, cluster2_values)))
    return correlations

print(find_correlations(new_logs))


