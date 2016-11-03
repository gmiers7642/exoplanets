import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import clean1 as c
import clustering as cl

if __name__ == '__main__':

    ### Import data, and create imputed KMeans data

    # Extract columns
    print "Importing data..."
    df = c.import_data('../data/planets.csv')
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
    print "Imputing data..."
    km_labels, df_imputed = cl.kmeans_centroid_fill(df_p, 3, 10)


    ### Transit and radial velocity group analysis

    # Split the data into transit and radial velocity groups
    print "Splitting into transit and radial velocity sets..."
    df_imputed['pl_discmethod'] = df['pl_discmethod']
    df_transit = df[df['pl_discmethod'] == 'Transit']
    df_radialv = df[df['pl_discmethod'] == 'Radial Velocity']
    df_p_transit = df_imputed[df_imputed['pl_discmethod'] == 'Transit'].drop('pl_discmethod', 1)
    df_p_radialv = df_imputed[df_imputed['pl_discmethod'] == 'Radial Velocity'].drop('pl_discmethod', 1)

    # SVDs for feature selection
    print "Creating feature seleciton SVDs..."
    ut,st,vtt = svd(df_p_transit)
    ur,sr,vtr = svd(df_p_radialv)

    # Feature selection, 11 for transiting, 7 for radial velocity, based on prior analysis
    print "Performing feature selection..."
    transit_relevances = cl.get_n_best(vtt,21, df_p_transit.columns)
    radial_relevances = cl.get_n_best(vtr,21, df_p_radialv.columns)
    cols_transit = radial_relevances['features'].values[0:11]
    cols_radialv = radial_relevances['features'].values[0:7]

    ### Agglomerative clustering based on optimal parameters from prior analysis
    print "Creating clusters..."
    n_clusters = 3

    # Transit
    ac_transit, df_select_transit = cl.agg_clustering(df_p_transit, cols_transit, n_clusters)
    labels_transit = ac_transit.labels_

    # Radial velocity
    ac_radialv, df_select_radialv = cl.agg_clustering(df_p_radialv, cols_radialv,
                                    n_clusters, linkage='average', affinity='cosine')
    labels_radialv = ac_radialv.labels_

    ### Data cleanup and addition of ancillary information
    print "Data cleanup..."
    df_p_transit['label'] = labels_transit
    df_p_radialv['label'] = labels_radialv

    ### Plot the data clusters for both the transit and radial velocity cases to make sure that everything went ok
    print "Creating plots..."
    fig = plt.figure(figsize=(20,8))
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(ut[:,0], ut[:,1], c=labels_transit, s=45, alpha=0.05)
    ax1.set_title("Transit clusters")
    ax1.set_xlim((-0.022, -0.016))
    ax1.legend(labels_transit)
    ax2 = plt.subplot(1,2,2)
    ax2.scatter(ur[:,0], ur[:,1], c=labels_radialv, s=45, alpha=0.2)
    ax2.set_title("Radial Velocity clusters")
    ax2.legend(labels_radialv)
    plt.suptitle("Cluster labeled scatterplots for transit and radial velocity discoveries")
    plt.show()
