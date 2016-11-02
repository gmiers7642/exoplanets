import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import clean1 as c

################################################################################
# Cleaning Functions
################################################################################

def standardize_all(df):
    for col in df.columns:
        df.loc[:, col] = ( df.loc[:, col] - df.loc[:, col].mean() ) / df.loc[:, col].std()
    return df

def log_all(dfi, logcols):
    df = dfi.copy()
    for col in df.columns:
        if col in logcols:
            df.loc[:,col] = df.loc[:,col].apply(lambda x: np.log(x + 1))
    return df

def fill_nan(df, value):
    for col in df.columns:
        df.loc[:,col] = df.loc[:,col].fillna(value=value)
    return df

def fill_avg(dfi):
    df = dfi.copy()
    for col in df.columns:
        df.loc[:,col] = df.loc[:,col].fillna(value=df[col].mean())
    return df

################################################################################
# Clustering Functions
################################################################################

def kmeans_centroid_fill(df, n_clusters, max_iterations):
    # Pre-process the data, apply lof to all columns
    df_p_log = log_all(df, df.columns)
    df_p_avg = fill_avg(df_p_log)

    # Set up for clustering
    Xm = df_p_avg
    valid = np.isfinite(Xm)
    mu = np.nanmean(Xm, 0, keepdims=1)
    X_hat = np.where(valid, Xm, mu)

    # Clustering iterations
    for ii in range(max_iterations):

        # perform clustering on filled-in data
        clus = KMeans(n_clusters, n_jobs=-1)
        labels_hat = clus.fit_predict(X_hat)

        # Update missing values based on cluster centroids
        X_hat[~valid] = clus.cluster_centers_[labels_hat][~valid]

    # Return the labels output by KMeans
    return clus.labels_, Xm

def agg_clustering(df, cols, n_clusters):
    df_select = df[cols]
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit(df_select)
    return ac, df_select

################################################################################
# Plotting Functions
################################################################################

def plot_pcs(A, rows, cols, T=False, colors=['blue']):
    fig = plt.figure(figsize=(20,8))
    for i in range(rows):
        for j in range(cols):
            plot_num = i*cols + j + 1
            ax = plt.subplot(rows, cols, plot_num)
            if T == True:
                ax.scatter(A[plot_num-1,:], A[plot_num,:], alpha=0.1, c=colors, s=30)
            elif T == False:
                ax.scatter(A[:,plot_num], A[:,plot_num+1], alpha=0.2, c=colors, s=30)
            ax.set_title("(" + str(plot_num-1) + "," + str(plot_num) + ")")
    plt.suptitle("Principal Components Plot", fontsize=15)
    plt.show()

def plot_pc_features(vt, num, cols, title=""):
    fig = plt.figure(figsize=(20,8))
    for i in range(num):
        ax = plt.subplot(1,4,i+1)
        ax.bar(left=np.arange(len(cols)), height=vt[:,i])
        ax.set_xticklabels(cols, rotation='vertical')
    plt.suptitle("Principal component plot for " + title)
    plt.show()

def color_coded_pc_plot(u, labels, xlim=(-0.021, -0.01), ylim=(-0.07, 0.03)):
    fig = plt.figure(figsize=(12,12))
    plt.scatter(u[:,0], u[:,1], c=labels, s=45, alpha=0.1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

################################################################################
# Unit tests
################################################################################

# To test Kmeans centroid clustering workflwo on full 21 features
def unit001():
    print "Importing data"
    df = c.import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = c.get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    print "Creating clusters"
    labels, df_imputed = kmeans_centroid_fill(df_p, 3, 10)

    # Split columns into two groups
    cph1 = cols_phys[0:10]
    cph2 = cols_phys[11:]

    print "Creating plots"
    c.physical_scatterplot(df_p, cph1, cph2, logcols=df_p.columns, \
                           colors=labels, alpha=0.2)

# To test Aglorrerative clustering workflwo on full 21 features (non-imputed data)
def unit002():
    print "Importing data"
    df = c.import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = c.get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    # Pre-process the data, apply lof to all columns
    df_p_log = log_all(df_p, df_p.columns)
    df_p_avg = fill_avg(df_p_log)

    # Actual agglomerative clustering
    n_clusters = 3
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit(df_p_avg)

    # Scatterplotting of agglomerative clusters
    cph1 = cols_phys[0:10]
    cph2 = cols_phys[11:]
    ac_labels = ac.labels_
    c.physical_scatterplot(df_p, cph1, cph2, logcols=df_p.columns, \
                                colors=ac_labels, alpha=0.2)

# To test plotting of columns with the most relevance from an svd
def unit003():
    print "Importing data..."
    df = c.import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = c.get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    # Pre-process the data, imputation, and create svd for plotting
    print "Applying pre-processing"
    km_labels, df_imputed = kmeans_centroid_fill(df_p, 3, 10)
    u,s,vt = svd(df_imputed)

    # Create plot of most important components
    print "Creating feature plots"
    plot_pc_features(vt, 4, df_imputed.columns, "21 columns")

# To test plotting of color coded pcs
def unit004():
    print "Importing data..."
    df = c.import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = c.get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    # Pre-process the data, imputation, and create svd for plotting
    print "Applying pre-processing..."
    km_labels, df_imputed = kmeans_centroid_fill(df_p, 3, 10)

    # Columns selected from previous analysi of 21 columns
    print "Clustering..."
    cols = ['st_age', 'pl_orbsmax', 'pl_bmassj', 'st_plx', 'st_dist', 'st_lum', 'st_mass']
    n_clusters = 3
    ac, df_select = agg_clustering(df_imputed, cols, n_clusters)
    labels = ac.labels_

    # Create svd
    u,s,vt = svd(df_select)

    # Create the color codeded pc plot
    print "Creating pc plot..."
    color_coded_pc_plot(u, labels)

# To test selection of most relevant features
def unit005():
    print "Importing data..."
    df = c.import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = c.get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    # Pre-process the data, imputation, and create svd for plotting
    print "Applying pre-processing..."
    km_labels, df_imputed = kmeans_centroid_fill(df_p, 3, 10)

    # Columns selected from previous analysi of 21 columns
    print "Clustering..."
    cols = ['st_age', 'pl_orbsmax', 'pl_bmassj', 'st_plx', 'st_dist', 'st_lum', 'st_mass']
    n_clusters = 3
    ac, df_select = agg_clustering(df_imputed, cols, n_clusters)
    labels = ac.labels_

    # Create svd
    u,s,vt = svd(df_select)

if __name__ == '__main__':
    # unit001 - To test Kmeans centroid clustering workflwo on full 21 features
    #unit001()

    # unit002 - To test Aglorrerative clustering workflwo on full 21 features (non-imputed data)
    #unit002()

    # unit003 - To test plotting of columns with the most relevance from an svd
    #unit003()

    # unit004 - To test plotting of color coded pcs
    unit004()
