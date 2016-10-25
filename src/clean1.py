import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_physical_columns(df, cols):
    return df[cols]

def plot_physical_histograms(df, logcols, num_rows, num_cols):
    df = df.copy()
    plt.figure(figsize=(20,20))
    for col in df.columns:
        color = 'blue'
        ax = plt.subplot(num_rows, num_cols, list(df.columns).index(col)+1)
        if col in logcols:
            temp = df.loc[:,col].apply(lambda x: np.log(x+1)).dropna()
            color = 'red'
            label = col + ' --->warning! log scale!'
        else:
            temp = df.loc[:,col].dropna()
            label = col
        ax.hist(temp, bins=35, color=color)
        ax.set_title(label, fontsize=10)
    plt.tight_layout()
    plt.show()

def physical_scatterplot(df, hcols, vcols, logcols):
    fig = plt.figure(figsize=(20,20))
    total_plots = len(hcols) * len(vcols)
    for col in df.columns:
        if col in logcols:
            df.loc[:,col] = df.loc[:,col].apply(lambda x: np.log(x+1))
    i = 1
    for hcol in hcols:
        for vcol in vcols:
            ax = plt.subplot(len(hcols), len(vcols), i)
            ax.scatter(df[hcol], df[vcol])
            ax.set_title('(' + hcol + "," + vcol + ')')
            i += 1
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
    plt.suptitle("XYZ Coordinate Conversion Check")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = get_physical_columns(df, cols_phys)

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']
    #plot_physical_histograms(df_p, logcols, 5, 5)

    # Convert from polar to XYZ
    '''df_c = convert_to_polar(df)
    plot_XYZ_histograms(df_c)'''

    # Scatterplot matrix for physical data
    cph1 = cols_phys[0:10]
    cph2 = cols_phys[11:]
    physical_scatterplot(df_p, cph1, cph2, logcols)
