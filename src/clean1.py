# Standard plotting and data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# plotly
import plotly.plotly as py
from plotly.graph_objs import *
import json

################################################################################
# Cleaning Functions
################################################################################

def import_data(filename):
    return pd.read_csv(filename)

def get_physical_columns(df, cols):
    return df[cols]

def convert_to_polar(df):
    # Reference:
    # http://math.stackexchange.com/questions/15323/how-do-i-calculate-the-cartesian-coordinates-of-stars
    df_c = df.loc[:,['ra', 'dec', 'st_dist']]
    df_c['X'] = ( df_c['st_dist'] * np.cos(df_c['dec']) ) * np.cos(df_c['ra'])
    df_c['Y'] = ( df_c['st_dist'] * np.cos(df_c['dec']) ) * np.sin(df_c['ra'])
    df_c['Z'] = df_c['st_dist'] * np.sin(df_c['dec'])
    return df_c

def create_correlation_matrix(df):
    return np.corrcoef(df)

def get_nan_pct(df):
    pcts = [1.0 - sum(np.isnan(df[col]) ) / float(df.shape[0]) for col in df.columns]
    df_miss = pd.DataFrame({
                       'Attribute': df.columns,
                       'Percent_missing': pcts
                        })
    return df_miss.sort_values(by='Percent_missing', ascending=True)

def add_multiplanet_column(df):
    temp = df.groupby('pl_hostname', as_index=False)['rowid'].count()
    temp.columns = [['pl_hostname', 'Multiple']]
    df = df.merge(temp, on='pl_hostname', how='inner')
    df.rename(columns={'Multiple_y':'Multiple'}, inplace=True)
    return df

def add_habitable_col(df, df_hc):
    df_hc = df_hc.iloc[0:10,:]
    print df_hc.shape
    df_hc['Name'] = df_hc['Name'].apply(lambda x: x.strip('*'))
    df_hc['letter'] = df_hc['Name'].apply(lambda x: x[-1])
    col = np.zeros((df.shape[0], 1))
    for i in range(df.shape[0]):
        for j in range(df_hc.shape[0]):
            if df.ix[i,'pl_hostname'] in df_hc.ix[j,'Name'] and df.ix[i,'pl_letter'] == df_hc.ix[j,'letter']:
                col[i] = 1.0
    return col

################################################################################
# Plotting Functions
################################################################################

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

def plot_physical_kdes(dfi, logcols, num_rows, num_cols, bandwidth):
    df = dfi.copy()
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
        temp = df[col].apply(lambda x: np.log(x+1)).dropna()
        kde = gaussian_kde(temp)
        kde.set_bandwidth(bandwidth)
        support = np.linspace(np.min(temp), np.max(temp), 300)
        ax.plot(support, kde(support), color=color)
        ax.fill_between(support, kde(support), alpha=0.2, color=color)
        ax.set_title(label, fontsize=15)
    plt.tight_layout()
    plt.show()

def physical_scatterplot(df, hcols, vcols, logcols, colors=['blue'], alpha=1.0):
    fig = plt.figure(figsize=(20,20))
    total_plots = len(hcols) * len(vcols)
    for col in df.columns:
        if col in logcols:
            df.loc[:,col] = df.loc[:,col].apply(lambda x: np.log(x+1))
    i = 1
    for hcol in hcols:
        for vcol in vcols:
            ax = plt.subplot(len(hcols), len(vcols), i)
            ax.scatter(df[hcol], df[vcol], alpha=alpha, c=colors)
            ax.set_title('(' + hcol + "," + vcol + ')')
            i += 1
    plt.tight_layout()
    plt.show()

def plot_XYZ_histograms(df):
    #fig = plt.figure(figsize=(20,8))
    df.hist(bins=25)
    plt.suptitle("XYZ Coordinate Conversion Check")
    plt.tight_layout()
    plt.show()

def set_plotly_credentials(filename):
    with open('plotly_credentials.json') as data_file:
        creds = json.load(data_file)
    plotly.tools.set_credentials_file(username=creds['username'], api_key=creds['api_key'] )

def create_random_mask(df, num):
    mask = np.random.choice(range(df.shape[0]), size=num, replace=False)
    return df.iloc[mask,:]

def plot_3D_planets(coords, dfi, sizecol, colorcol, logcols):
    df = dfi.copy()
    for col in df.columns:
        if col in logcols:
            df.loc[:,col] = df.loc[:,col].apply(lambda x: np.log(x+1))
    trace1 = Scatter3d(
        x=coords['X'],
        y=coords['Y'],
        z=coords['Z'],
        #text=df[textcol],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            #sizeref=750,
            size=df[sizecol],
            color = df[colorcol],
            colorscale = 'Viridis',
            colorbar = dict(title = 'Planet<br>Properties'),
            line=dict(color='rgb(140, 140, 170)')
        )
    )

    theTitle = "3D planet plot of X,Y,Z vs." + sizecol + " as size, and " + colorcol + " as color"

    data=[trace1]
    layout=dict(height=800, width=800, title=theTitle)
    fig=dict(data=data, layout=layout)
    py.iplot(fig, filename='3DBubble')

################################################################################
# Unit tests
################################################################################

# unit001 - For testing physical data import
def unit001():
    df = import_data('../data/planets.csv')

    # Extract columns
    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = get_physical_columns(df, cols_phys)

    print "Physical data, first few rows:"
    print df_p.head()

# unit002 - For testing polar to rectangular conversion
def unit002():
    df = import_data('../data/planets.csv')

    # Convert from polar to XYZ
    df_c = convert_to_polar(df)
    plot_XYZ_histograms(df_c)

# unit003 - For testing the scatterplot
def unit003():
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

    # Scatterplot matrix for physical data, for one selected quadrant
    cph1 = cols_phys[0:10]
    cph2 = cols_phys[11:]
    physical_scatterplot(df_p, cph1, cph2, logcols, alpha=0.2)

# unit004 - For testing testing histogramming
def unit004():
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
    plot_physical_histograms(df_p, logcols, 5, 5)

# unit005 - For testing 3D plotting
def unit005():
    df = pd.read_csv('../data/planets.csv')
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
    coords = convert_to_polar(df)
    plot_3D_planets(create_random_mask(coords, 200),
                    create_random_mask(df_p, 200),
                    'pl_orbper', 'st_teff', logcols)

# unit006 - For testing kde plots
def unit006():
    df = pd.read_csv('../data/planets.csv')

    cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                 'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                 'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                 'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                 'st_acts']
    df_p = df[cols_phys]

    logcols = ['pl_bmassj', 'pl_dens', 'pl_orbper', 'pl_orbsmax',
               'pl_radj', 'st_dist', 'st_rad', 'st_teff', 'st_dens',
               'pl_rvamp', 'st_plx', 'st_vsini', 'st_acts']

    plot_physical_kdes(df_p, logcols, 5, 5, 0.15)

# unit007 - For testing calculation of fraction of missing values
def unit007():
        df = pd.read_csv('../data/planets.csv')

        cols_phys = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl',
                     'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist',
                     'st_optmag', 'st_teff', 'st_mass', 'st_rad',
                     'st_logg', 'st_dens', 'st_lum', 'pl_rvamp',
                     'pl_eqt', 'st_plx', 'st_age', 'st_vsini',
                     'st_acts']
        df_p = df[cols_phys]
        df_miss = get_nan_pct(df_p)
        print "Missing column information:"
        print df_miss.head()

# unit008 - For testing the additon of multi-planet flag 'Multiple'
def unit008():
    df = pd.read_csv('../data/planets.csv')
    df = add_multiplanet_column(df)
    print "DataFrame with multi-planet column added:"
    print df[['pl_hostname', 'Multiple']].head(13)

# unit009 - For testing addition of habitable column 'Habitable_c'
def unit009():
    # Load data
    df = import_data('../data/planets.csv')
    df_hc = pd.read_csv('../data/Habitalbe_list_conservative.csv')

    # Test the add_habitable_col() function
    col = add_habitable_col(df, df_hc)
    df['Habitable_c'] = col

    # Print unique values, this should be 0.0 and 1.0
    print "Unique values in Habitable_c column"
    print df['Habitable_c'].value_counts()

################################################################################
# Main
################################################################################

if __name__ == '__main__':
    # For testing physical data import
    #unit001()

    # For testing polar to rectangular conversion
    #unit002()

    # For testing the scatterplot
    unit003()

    # For testing testing histogramming
    #unit004()

    # For testing 3D plotting
    #unit005()

    # For testing kde plots
    #unit006()

    # For testing calculation of fraction of missing values
    #unit007()

    # For testing the additon of multi-planet flag 'Multiple'
    #unit008()

    # For testing addition of habitable column 'Habitable_c'
    #unit009()
