<img src="images/opening-background.png" alt="alt text" align="middle">

Code documentation can be found <a href="https://github.com/gmiers7642/exoplanets/blob/master/Code_Documentation.md">here.</a>

# Outline
1. [Unsupervised Learning](#1Unsupervised Learning)
2. [Exoplanets](#2Exoplanets)
3. [Data Challenges](#3Data Challenges)
4. [Methodology](#4Methodology)
    [4.1 Data Wrangling and Processing](#4_1 Data Wrangling and Processing)
    [4.2 Data Imputation](#4_2 Data Imputation)
    [4.3 Clustering](#4_3 Clustering)
    [4.4 Dimensionality Reduction](#4_4 Dimensionality Reduction)
    [4.5 Data Interpretation](#4_5 Data Interpretation)
5. [Major Results and Interpretation](#5 Major Results and Interpretation)
6. [Future work](#6Future work)
7. [Tools Used](#7Tools Used)

<hr>

### 1. <a id="1Unsupervised Learning">Unsupervised Learning</a>
* Very powerful technique for exploratory data analysis (EDA)
* There are many applications for unsupervised learning
    * Marketing
    * Genetics
    * Politics
    * Many others
* Here, unsupervised learning has been used to detect interesting structure in exoplanet data available from the <a href="http://exoplanetarchive.ipac.caltech.edu/">NASA exoplanet archive,</a>

<hr>

### 2. <a id="2Exoplanets">What are Exoplanets?</a>
* Exoplanets are planets that we have discovered outside of our solar system
* There is a LOT of data available on exoplanets, for instance:
  * http://exoplanetarchive.ipac.caltech.edu/index.html
* What can we quickly discover about exoplanets using out-of-the box tools in Python?
    * What are some common characteristics of exoplanets?
    * How do they compare to the Earth and Sun?
    * What are Habitable planets, and how do they compare to the Earth and Sun?

<hr>

### 3. <a id="3Data Challenges">Data Challenges</a>
* There are many features in the exoplanet data set
    * Over 150 features in total
        * These are physical measurements that have been made by astronomers using various
    * 2/3 of these are error bars associated with the physical measurements contained in the data
* Much missing data
    * Imputation required
* Data is large scale, so transformations are required
    * For this analysis, logarithms were applied to all of the features
        * This reveals far more structure in the data
        * However, it can make interpretation more difficult
* There are <a href="https://en.wikipedia.org/wiki/Methods_of_detecting_exoplanets">many methods</a> by which exoplanets are discovered, the two most common two are:
    * Transit
    * Radial Velocity
* At this time, this project considers only the transiting planets, since they represent the bulk of the data,

<hr>

### 4. <a id="4Methodology">Methodology</a>

<hr>

### 5. <a id="5Major Results and Interpretation">Major Results and Interpretation</a>

<hr>

### 6. <a id="6Future work">Future work</a>
* There is a lot more work to be done here.
    * There is a lot of interesting structure present in the data that is still left to be explored.  
        * This is a plot of several features used here plotted as colors on a scatterplot of <> which shows three distinct clusters. <div>
        <img src="images/3_clusters.png" alt="alt text" align="bottom">
        * These structures could easily be investigated using further cluster analysis.
    * Performing the same analysis with the planets discovered using radial velocity

<hr>

### 7. <a id="7Tools Used">Tools Used</a>
* ScikitLearn KMeans clustering
* Numpy Singular Value decomposition (SVD)
* ScikitLearn Agglomerative clustering
* Matplotlib for plotting and interpretation of data
* Pandas and Numpy for quick and convenient data manipulation



![python logo](images/python-logo.png) ![skl_logo](images/skl_logo.png) ![pandas_logo](images/pandas_logo.png)
![numpy_logo](images/numpy_logo.jpg) ![matplotlib_logo](images/matplotlib_logo.png)
