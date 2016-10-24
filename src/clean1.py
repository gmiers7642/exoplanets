import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_physical_columns(df):
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad']
    return df[cols_phys]

def plot_physical_histograms(df, logcols):
    df = df.copy()
    plt.figure(figsize=(20,20))
    for col in df.columns:
        color = 'blue'
        ax = plt.subplot(4, 3, list(df.columns).index(col)+1)
        if col in logcols:
            temp = df.loc[:,col].apply(lambda x: np.log(x+1)).fillna(value=0.)
            color = 'red'
            print "log applied to:", col
        else:
            temp = df.loc[:,col].fillna(value=0.)
        #df_p.hist(ax=ax, bins=25, xlabelsize=15, xrot=-45, ylabelsize=15, color=color)
        ax.hist(temp, bins=25, color=color)
        ax.set_title(col + "   warning! log scale!")
    plt.tight_layout()
    plt.show()

def convert_to_polar(df):
    # Reference:
    # http://math.stackexchange.com/questions/15323/how-do-i-calculate-the-cartesian-coordinates-of-stars
    df_c = df.loc[:,['ra', 'dec', 'st_dist']]
    df_c['X'] = ( df_c['st_dist'] * np.cos(df_c['dec']) ) * np.cos(df_c['ra'])
    df_c['Y'] = ( df_c['st_dist'] * np.cos(df_c['dec']) ) * np.sin(df_c['ra'])
    df_c['Z'] = df_c['st_dist'] * np.sin(df_c['dec'])
    return df_c

def plot_XYZ_histograms(df):
    #fig = plt.figure(figsize=(20,8))
    df_c.hist(bins=25)
    plt.suptitle("XYZ Coordinate Convervion Check")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Extract columns
    df = pd.read_csv('../data/planets.csv')
    df_p = get_physical_columns(df)
    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff']
    plot_physical_histograms(df_p, logcols)

    # Convert from polar to XYZ
    '''df_c = convert_to_polar(df)
    plot_XYZ_histograms(df_c)'''
