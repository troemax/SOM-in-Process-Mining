import os
os.environ['OMP_NUM_THREADS'] = '1' # I added this cause of the sklearn warning: UserWarning: KMeans is known to have a memory leak on Windows with MKL,
                                    # when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
                                    # This has to be done before import numpy to work properly.
import numpy as np
rng = np.random.default_rng()
import scipy
import sklearn
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from datetime import datetime

import core.EventLog as EventLog
from ui.FeatureSelectorPage import get_train_data
from core.Logger import logger

# Global Variables
X = 0 # width
Y = 0 # height
n = 0 # number of nodes (=X*Y)

df_train = None # shape = N * p
arr_train = None # shape = N * p
normalization_scalers = None # dictionary that holds sklearn scalers
dict_df_to_arr = None # which column of df_train corresponds to which column of arr_train
N = 0
p = 0 # number of features
N_var = None
variants_distances = None # shape = N_var * N_var, dtype = int
variants_weight = None # shape = N_var, dtype = int

arr_MCoords = None # shape = n * 2
arr_DistanceMatrix = None # shape = n * n
arr_M = None # shape = n * p OR n, dtype = float OR int(N_var)

BMUs = None # shape = N OR N_var, dtype = int(n)
quantization_error_1 = -1
quantization_error_2 = -1
topographic_error = -1

hex_neighbor_pairs = None # Used for colored lines between hexagons

def initialize_random():
    global arr_M
    if n > N:
        init_indices = rng.choice(N, size=n, replace=True)
    else:
        init_indices = rng.choice(N, size=n, replace=False)
    arr_M = arr_train[init_indices,:].astype(float) # astype also copies the array
    
def initialize_linear():
    # so-called autocorrelation matrix
    me = np.mean(arr_train, axis=0)
    D = arr_train - me
    A = (D.T @ D) / N # symmetric matrix -> use np.linalg.eigh instead of np.linalg.eig
    eigvals, V = np.linalg.eigh(A) # V[:,i] are already normalized
    reodering = np.argsort(eigvals)
    eigvals = eigvals[reodering[[-1,-2]]]
    V = V[:,reodering[[-1,-2]]]
    V = V * np.sqrt(eigvals)
    # scale coordinates -1 to 1
    coords = arr_MCoords.copy()
    c_min = np.min(coords, axis=0)
    c_max = np.max(coords, axis=0)
    coords = 2 * (coords - c_min) / (c_max-c_min) - 1
    # Build arr_M
    global arr_M
    arr_M = np.full((n, p), me)
    for i in range(n):
        arr_M[i,:] += coords[i,0] * V[:,0].T + coords[i,1] * V[:,1].T

def create_new(X_, Y_, initialization):
    global X, Y, n, arr_MCoords, arr_DistanceMatrix
    X = X_
    Y = Y_
    n = X*Y
    arr_MCoords = np.empty((n, 2), dtype=float)
    for i in range(n):
        x = i % X
        y = i // X
        if y % 2 == 0:
            arr_MCoords[i,:] = (x, y*np.sqrt(0.75))
        else:
            arr_MCoords[i,:] = (x+0.5, y*np.sqrt(0.75))
    arr_DistanceMatrix = scipy.spatial.distance_matrix(arr_MCoords, arr_MCoords)
    
    global df_train, arr_train, normalization_scalers, dict_df_to_arr, N, p
    df_train = get_train_data().astype(float) # set all to float
    
    # Hard coded duration and NEvents to be transformed by the logarithm if they are in the training dataset. One should come up with another solution for that.
    if "MAIN_Duration" in df_train.columns:
        df_train.loc[:,"MAIN_Duration"] = np.log(df_train.loc[:,"MAIN_Duration"])
        logger.info(f"DANGER! Hard-coded logarithm was used on MAIN_Duration. See SOM.py file for more info.")
    if "MAIN_NEvents" in df_train.columns:
        df_train.loc[:,"MAIN_NEvents"] = np.log(df_train.loc[:,"MAIN_NEvents"])
        logger.info(f"DANGER! Hard-coded logarithm was used on MAIN_NEvents. See SOM.py file for more info.")
    
    # normalize
    normalization_scalers = {}
    for c in df_train.columns:
        if not c.startswith("DECL_"): # DECL_ columns need no normalization
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            df_train.loc[:,c] = scaler.fit_transform(df_train.loc[:,c].to_numpy().reshape(-1, 1))
            normalization_scalers[c] = scaler
    arr_train = df_train.to_numpy()
    dict_df_to_arr = {col: i for i, col in enumerate(df_train.columns)}
    logger.info(f"arr_train.shape={arr_train.shape}")
    N = arr_train.shape[0]
    p = arr_train.shape[1]
    if initialization == "Random":
        initialize_random() # It ensures 0<=x<=1 for every feature x (here, x=0 or x=1)
    else:
        initialize_linear()
    
    calculate_BMUs_and_Errors()

def calculate_BMUs_and_Errors():
    global BMUs
    # Idea: maximize -2xy+y^2 instead of (x-y)^2
    d = np.dot(arr_train, arr_M.T) # shape = N * n
    d *= -2
    m2 = np.einsum('ij,ij->i', arr_M, arr_M) # shape = n
    d += m2.reshape(1, -1)
    BMUs = np.full(N, -1, dtype=int)
    
    d_partition2 = np.partition(d, kth=1, axis=1) # second column contains second-smallest value -> first column contains smallest value
    minima_places = np.isclose(d, d_partition2[:, 0, np.newaxis]) # shape = N * n, dtype=bool; gets updated over time
    
    minima_uniqueness = np.where(np.sum(minima_places, axis=1)==1)[0] # indices
    BMUs[minima_uniqueness] = np.argwhere(minima_places[minima_uniqueness])[:,1]
    open_indices = np.setdiff1d(np.arange(len(BMUs)), minima_uniqueness)
    
    r = 1
    tol = 0.001 # tolerance
    while len(open_indices) > 0:
        if r > np.sqrt(X**2 + 0.75*Y**2): # approximately the diagonal of the SOM
            logger.info("Training step stopped with non-unique minima :(")
            for i in open_indices:
                BMUs[i] = rng.choice(np.where(minima_places[i])[0])
            break
        
        # 1. For every open N: Find neighborhood
        activation_matrix = arr_DistanceMatrix <= r + tol
        
        # 2. Calculate sum of neighborhood distance to N
        d_sums = d[open_indices] @ activation_matrix
        d_sums /= np.sum(activation_matrix, axis=0)
        d_sums[~minima_places[open_indices]] = np.inf # only look at minima from last iteration
        
        # 3. Check again for minima
        minima_places[open_indices] = np.isclose(d_sums, np.min(d_sums, axis=1)[:, np.newaxis])
        minima_uniqueness = np.where(np.sum(minima_places[open_indices], axis=1)==1)[0]
        BMUs[open_indices[minima_uniqueness]] = np.argwhere(minima_places[open_indices[minima_uniqueness]])[:,1]
        open_indices = np.setdiff1d(open_indices, open_indices[minima_uniqueness])
        
        r += 1
    
    # Topological Error
    top_error = np.ones(BMUs.shape[0], dtype=bool) # shape = N
    for i in range(BMUs.shape[0]):
        bmu = BMUs[i]
        second_smallest_value = d_partition2[i, 1] # could be the same as the smallest value
        minima2_places = np.where(np.isclose(d[i,:], second_smallest_value))[0]
        if np.any(np.isclose(arr_DistanceMatrix[bmu, minima2_places], 1)):
            top_error[i] = 0
    
    global quantization_error_1, quantization_error_2, topographic_error
    BMU_vecs = arr_M[BMUs] # shape = N * p
    quantization_error_1 = np.linalg.norm(arr_train - BMU_vecs, axis=1, ord=1).mean()
    quantization_error_2 = np.linalg.norm(arr_train - BMU_vecs, axis=1).mean()
    topographic_error = np.average(top_error)

def train(sigma):
    # From SOMPY: The following calculation of arr_M is based on SOM Toolbox for Matlab by Helsinky University
    neighborhood = np.exp(-arr_DistanceMatrix**2/(2*sigma**2))
    P = scipy.sparse.csr_matrix((np.ones(N, dtype=int),(BMUs, np.arange(N))), shape=(n,N))
    S = P.dot(arr_train)
    nom = neighborhood.dot(S)
    nV = P.sum(axis=1).reshape(1, n)
    denom = nV.dot(neighborhood).reshape(n, 1)
    arr_M[:,:] = np.divide(nom, denom) # [:,:] is used to not need global arr_M
    
    calculate_BMUs_and_Errors() # calculate BMUs for next iteration to visualize them already now

def train_multi(sigma_start, sigma_end, T):
    for t in range(T):
        sigma = sigma_start * (sigma_end / sigma_start)**(t/(T-1))
        train(sigma)

def get_feature_values(column_name):
    if column_name not in dict_df_to_arr:
        return None
    feature_values = arr_M[:,dict_df_to_arr[column_name]]
    # TODO If colorbars will be shown then we should reverse the normalization
    #if not column_name.startswith("DECL_"):
    #    feature_values = normalization_scalers[column_name].inverse_transform(feature_values.reshape(-1, 1))
    return feature_values

def calculate_UMatrix():
    UMatrix = np.empty(n)
    for i in range(n):
        nj = np.where(np.isclose(arr_DistanceMatrix[i], 1))[0] # The isclose method checks equality to 1 with a tolerance
        diffs = arr_M[i] - arr_M[nj] # numpy broadcasting allows differences of matrices with shapes (1,p) minus (_,p)
        diffs = np.linalg.norm(diffs, axis=1)
        UMatrix[i] = np.mean(diffs)
    return UMatrix

def get_hex_support():
    return np.bincount(BMUs, minlength=n) # shape = n

def calculate_test_data_DECL(declare_id):
    if EventLog.log.declareConstraint.isOneVar(declare_id):
        num_classes = 2
        classes = EventLog.log.df_data[f"DECL_{declare_id}"]
    else:
        num_classes = 3
        declare_id_AtLeastOne = EventLog.log.declareConstraint.getPartner(declare_id)
        data = EventLog.log.df_data[f"DECL_{declare_id}"] # two-variable Declare constraint
        data_AtLeastOne = EventLog.log.df_data[f"DECL_{declare_id_AtLeastOne}"] # corresponding AtLeastOne constraint
        classes = np.where(data & data_AtLeastOne, 2, data) # trick to distinguish in three cases:
        # (data,data_AtLeastOne): (0,0) is not possible; (0,1) becomes 0, (1,0) becomes 1, and (1,1) becomes 2
    
    value = np.zeros((n, num_classes), dtype=int)
    np.add.at(value, (BMUs, classes), 1) # crazy numpy solution - never saw it before
    
    return value, num_classes

def calculate_test_data_categoric(attribute_name):
    classes = EventLog.log.df_data[attribute_name]
    list_classes = classes.unique().tolist()
    classes_to_position = {c:i for i, c in enumerate(list_classes)}
    num_classes = len(classes_to_position)
    classes = classes.map(classes_to_position)
    
    value = np.zeros((n, num_classes), dtype=int)
    np.add.at(value, (BMUs, classes), 1) # crazy numpy solution - never saw it before
    
    return value, num_classes, list_classes

def calculate_test_data_numeric(attribute_name, num_classes):
    weights = EventLog.log.df_data[attribute_name]
    classes = np.floor(num_classes * np.argsort(np.argsort(weights)) / N).astype(int) # shape = N; double sorting is like a magic trick
    
    value = np.zeros((n, num_classes), dtype=int)
    np.add.at(value, (BMUs, classes), 1) # crazy numpy solution - never saw it before
    
    return value, np.percentile(weights, np.linspace(0,100,11))

last_clustering_hashes = {}
last_clustering_labels = {}
did_it_already=False
def getClustering(option, k):
    parameters_hash = hash((option, k))
    arr_M_hash = hash(arr_M.tobytes())
    
    if parameters_hash in last_clustering_hashes and last_clustering_hashes[parameters_hash] == arr_M_hash:
        return last_clustering_labels[parameters_hash]
    if option == "KMeans":
        labels = sklearn.cluster.KMeans(n_clusters=k).fit(arr_M).labels_
    elif option == "Agglomerative":
        model = sklearn.cluster.AgglomerativeClustering(n_clusters=k, metric='manhattan', linkage='complete', compute_distances=True).fit(arr_M)
        labels = model.fit(arr_M).labels_
        # show dendrogram
        global did_it_already
        if not did_it_already:
            did_it_already = True
            plt.title("Agglomerative Clustering Dendrogram")
            plot_dendrogram(model, color_threshold=0) #, truncate_mode="level", p=3)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            filename = "outputs/"+"AgglomerativeClustering"+datetime.now().strftime(" %Y-%m-%d %H_%M_%S")+".png"
            plt.savefig(filename, dpi=200)
            plt.close()
    else:
        raise Exception(f"The option {option} is not a valid clustering option here!")
    last_clustering_hashes[parameters_hash] = arr_M_hash
    last_clustering_labels[parameters_hash] = labels
    return labels

def getInterestingWeightsByVariance():
    return np.argsort(np.var(arr_M, axis=0))[::-1]

# copied from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, **kwargs)