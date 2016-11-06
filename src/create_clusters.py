import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import DBSCAN
from tsne import bh_sne
import matplotlib.pyplot as plt
import clean1 as c
import clustering as cl

def create_clusters_agg():
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
    plt.savefig("../data/QC001_Clusters_rad_and_trans.png")

    ### Data cleanup and addition of ancillary information
    print "Data cleanup..."
    df_p_transit['label'] = labels_transit
    df_p_radialv['label'] = labels_radialv

    ### Merge data frames back together and output them to disk
    print "Exporting data..."
    merged = df_p_transit.merge(df_p_radialv, how='outer')
    merged.to_csv("../data/planets_physical_w_labels.csv", index=False)

def create_clusters_dbscan():
    np.random.seed(seed=12509234)

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
    km_labels, df_imputed = cl.kmeans_centroid_fill(df_p, 3, 10)

    # Create TSNE embedding
    vis_data_transit = bh_sne(df_imputed, perplexity=40)
    vis_x_transit = vis_data_transit[:, 0]
    vis_y_transit = vis_data_transit[:, 1]

    # Create a background plot of TSNE embedding
    fig = plt.figure(figsize=(12,8))
    plt.scatter(vis_y_transit, vis_x_transit, c=['blue'], cmap=plt.cm.get_cmap("jet", 10), alpha=0.2)
    plt.savefig("../data/QC010_TSNE_background.png")

    # DBSCAN clustering
    X = np.array([vis_x_transit, vis_y_transit]).T
    dbs = DBSCAN(eps=2.1, min_samples=12)
    dbs.fit(X)

    # Generate clustering plot from TSNE
    n_clusters = len(np.unique(dbs.labels_))
    fig = plt.figure(figsize=(15,12))
    plt.scatter(vis_y_transit, vis_x_transit, c=dbs.labels_, cmap=plt.cm.get_cmap("jet", n_clusters), alpha=1.0, s=10*dbs.labels_ + 1)
    plt.colorbar(ticks=range(n_clusters))
    plt.clim(-0.5, n_clusters - 0.5)
    plt.savefig("../data/QC011_TSNE_clustering_w_sizes.png")

# Perform svd, clustering, attribute comparison
def svd_cluster_analysis():
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

    # Pre-process the data, apply lof to all columns
    print "Creating imputed data..."
    km_labels, df_imputed = cl.kmeans_centroid_fill(df_p, 3, 10)

    print "Separating transit and radial velocity data..."
    df_imputed['pl_discmethod'] = df['pl_discmethod']
    df_transit = df[df['pl_discmethod'] == 'Transit']
    df_radialv = df[df['pl_discmethod'] == 'Radial Velocity']
    df_p_transit = df_imputed[df_imputed['pl_discmethod'] == 'Transit'].drop('pl_discmethod', 1)
    df_p_radialv = df_imputed[df_imputed['pl_discmethod'] == 'Radial Velocity'].drop('pl_discmethod', 1)

    # SVDs
    print "Determining relevances of features..."
    ut,st,vtt = svd(df_p_transit)
    ur,sr,vtr = svd(df_p_radialv)

    transit_relevances = cl.get_n_best(vtt,21, df_p_transit.columns)
    radial_relevances = cl.get_n_best(vtr,21, df_p_radialv.columns)

    ### So, 11 pcs for transits
    cols_transit = radial_relevances['features'].values[0:11]
    ### And, 7 for the radial velocity cases
    cols_radialv = radial_relevances['features'].values[0:7]

    print "Performing cluster analysis..."
    n_clusters = 3

    # Transit
    ac_transit, df_select_transit = cl.agg_clustering(df_p_transit, cols_transit, n_clusters)
    labels_transit = ac_transit.labels_

    # Radial velocity
    ac_radialv, df_select_radialv = cl.agg_clustering(df_p_radialv, cols_radialv, n_clusters)
    labels_radialv = ac_radialv.labels_

    print "Creating plotting attributes for pca cluster analysis..."
    uts, sts, vtts = svd(df_p_transit)
    urs, srs, vtrs = svd(df_p_radialv)

    # Plotting the first two pc clusters, commented out until ready to export
    #cl.color_coded_pc_plot(uts[:,0:], labels_transit, xlim=None, ylim=None)

    df_p_transit['label'] = labels_transit + 1

    print "Creating separated groups for svd clusters..."
    df1 = df_p_transit[df_p_transit['label'] == 1]
    df2 = df_p_transit[df_p_transit['label'] == 2]
    df3 = df_p_transit[df_p_transit['label'] == 3]

    print "Creating normailzed summary statistics..."
    df_sst = pd.DataFrame({'full_mean':df_p_transit.mean(),
                           'full_std':df_p_transit.std(),
                           'g1_mean':df1.mean(),
                           'g1_std':df1.std(),
                           'g2_mean':df2.mean(),
                           'g2_std':df2.std(),
                           'g3_mean':df3.mean(),
                           'g3_std':df3.std()
                          }).T

    df_means = df_sst.loc[['full_mean', 'g1_mean', 'g2_mean', 'g3_mean'],:]
    df_means_n = df_means.copy()
    disp_cols = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_bmassj', 'pl_radj', 'st_lum', 'pl_rvamp', 'st_vsini']
    df_means_n = df_means_n[disp_cols]

    for col in df_means_n:
        df_means_n.loc['g1_mean',col] = df_means_n.loc['g1_mean',col] / df_means_n.loc['full_mean',col]
        df_means_n.loc['g2_mean',col] = df_means_n.loc['g2_mean',col] / df_means_n.loc['full_mean',col]
        df_means_n.loc['g3_mean',col] = df_means_n.loc['g3_mean',col] / df_means_n.loc['full_mean',col]

    # Bar plot highlighting differences in features between the columns, commented out until ready
    #print "Creating bar plot of svd clusters..."
    #df_means_n.T[['g1_mean', 'g2_mean', 'g3_mean']].plot(figsize=(15,8), kind='bar');
    #plt.show()

    print "Entering data on the Earth / Sun system for comparison..."
    df_s = pd.DataFrame(
        {'pl_orbper':365.24,
         'pl_orbsmax':1.00001018,
         'pl_orbeccen':0.0167086,
         'pl_orbincl':1.578690,
         'pl_bmassj':0.0911301,
         'pl_radj':0.00314442,
         'pl_dens':5.514,
         'st_dist':0.0,
         'st_optmag':4.75,
         'st_teff':5777.0,
         'st_mass':1.0,
         'st_rad':1.0,
         'st_logg':1.447468,
         'st_dens':1.41,
         'st_lum':0.0,
         'pl_rvamp':0.1,
         'pl_eqt':300.0,
         'st_plx':21600.0,
         'st_age':4.6,
         'st_vsini':0.46511,
         'st_acts':0.65},
                        index=['sun'])

    df_s = df_s.loc[:,disp_cols]
    df_means_n = df_means_n.append(df_s)

    # Critical normalizing step
    for col in df_means_n.columns:
        df_means_n.loc['sun',col] = np.log(df_means_n.loc['sun',col] + 1) / df_means_n.loc['full_mean',col]

    # Plotting comparison attributes with the sun / earth system included, commented out until ready
    df_means_n.T[['g1_mean', 'g2_mean', 'g3_mean', 'sun']].plot(figsize=(15,7), kind='bar');
    plt.show()

if __name__ == '__main__':
    #create_clusters_agg()
    #create_clusters_dbscan()
    svd_cluster_analysis()
