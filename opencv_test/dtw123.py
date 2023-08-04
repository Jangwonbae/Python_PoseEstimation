from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from dtaidistance import dtw_ndim
from scipy import stats
#series1 = np.array([[0, 0], [0, 1],   [2, 1],    [0, 1], [0, 0]], dtype=np.double)
series1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0,2,10,1,4,5,6])
series2 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0,4,5,6])
az = stats.zscore(series1)
bz = stats.zscore(series2)
"""_summary_

series1 = np.empty((0,2), dtype=np.double)
series1 = np.append(series1, np.array([[0,0]]), axis=0)
series1 = np.append(series1, np.array([[0,1]]), axis=0)
series1 = np.append(series1, np.array([[2,1]]), axis=0)
series1 = np.append(series1, np.array([[0,1]]), axis=0)
series1 = np.append(series1, np.array([[0,0]]), axis=0)
series1 = np.append(series1, np.array([[0,0]]), axis=0)
series1 = np.append(series1, np.array([[1,2]]), axis=0)
series1 = np.append(series1, np.array([[2,3]]), axis=0)
series1 = np.append(series1, np.array([[3,4]]), axis=0)

series2 = np.array([[0, 0],
                    [2, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4]], dtype=np.double)
d = dtw_ndim.distance(series1, series2)
print(d)
path = dtw_ndim.warping_path(series1, series2)
print(path)
print(series1)

dtwvis.plot_warping(series1, series2, path, filename="warp2.png")
"""
path = dtw.warping_path(az, bz)
dtwvis.plot_warping(az, bz, path, filename="warp4.png")
