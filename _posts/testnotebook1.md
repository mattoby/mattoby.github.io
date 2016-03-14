---
layout: post
title: "Exploring lead levels in Michigan"
published: true
---

# Analysis of lead bll levels (Arnhold)


```python
# imports:
from __future__ import division
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nan
import seaborn as sns
import time
#import time
import pickle

#from scipy.stats import ttest_ind
#from scipy.stats import ranksums
#import statsmodels.formula.api as sm
#from functools import reduce

# for running notebook:
%load_ext autoreload
%autoreload 2
%matplotlib inline  

## sklearn imports:
#import sklearn.linear_model
#import sklearn.cross_validation
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
#from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import sklearn.linear_model as linear_model
from sklearn.cross_validation import train_test_split

#import sys
sys.path.append('/Users/matto/Documents/taxidata/')
sys.path.append('/Users/matto/Dropbox/Insight/datasciencetools/')
sys.path.append('/Users/matto/Documents/censusdata/zipcodes/cb_2014_us_zcta510_500k/')
sys.path.append('/Users/matto/Documents/censusdata/')


## set options:

# style for plots:
sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

# so that i can print as many lines as i want
np.set_printoptions(threshold='nan') 

# so i can see all rows of dfs:
pd.set_option('display.max_columns', None) # 500
pd.set_option('display.max_rows', 1000)

# import proper division:
from __future__ import division

## import my toolboxes:
#import taxitools as tt
#import memorytools as mt
import datasciencetools as dst
import leadtools as lt
import geotools as gt

```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## A. Initial setup


```python
# import lead bll dataset for michigan (by zip and zcta):
dfl, dfm = lt.build_michigan_lead_dataset(savetoFile=False)
```

    processing file ACS_14_5YR_S1701_with_ann.csv...
    processing file ACS_14_5YR_S1702_with_ann.csv...
    processing file ACS_14_5YR_S0501_with_ann.csv...
    processing file ACS_14_5YR_S1401_with_ann.csv...
    processing file ACS_14_5YR_S1501_with_ann.csv...
    processing file ACS_14_5YR_S1601_with_ann.csv...
    processing file ACS_14_5YR_S2701_with_ann.csv...
    processing file ACS_14_5YR_DP04_with_ann.csv...
    Num dfl cols, pre filtering: 569
    Num dfl cols, post-GEO.id2 filtering: 561
    Num dfl cols, post-allnan filtering: 462
    Num dfl cols, post-perc>100 filtering: 443
    Histogram of # non-nans in each column of the data



![png](testnotebook1_files/testnotebook1_3_1.png)


    Cutoff of # non-nan values: 800
    Num dfl cols, post-nan filtering: 368
    Filter for nan rows in bll counts:
    Rows in dfl, pre-filtering: 970 
    Rows in dfl, post-filtering bll>=5 rows: 943 
    Rows in dfl, post-filtering bll>=10 rows: 941 
    
    'perc_bll_ge5' and 'perc_bll_ge10' are the lead features to predict.
    
    Take 1 row per zcta (should combine - do later):
    Rows in dfl, pre-filtering: 941 
    number of unique zctas:  938
    Rows in dfl, post-filtering: 938 



```python
# import shapes & locations of zip codes:
zctacodes, zctashapes = lt.zcta_geographies()
dfl_zctashapes, dfl_zctacenters, dfl_cities = lt.zctas_for_dfl(zctacodes, zctashapes, dfl)
```


```python
# add new features to dfl based on nearby zctas:
dfl_closezctas, dfl_closezctadists = lt.find_neighboring_zctas(dfl, dfl_distmat, Nneighbors=5)
dfl, dfm = lt.build_neighboring_zcta_features(dfl, dfm, dfl_closezctas)

```

    Note: only desc, outcode, and coltype are updated in dfm



```python
# distances between zctas:
fromScratch = False
if fromScratch:
    dfl_shapecenters, dfl_distmat = gt.build_geodist_matrix(dfl['zcta'].values, dfl_zctashapes, distmetric='haversine')
else:
    basedir = '/Users/matto/Documents/censusdata/arnhold_challenge/'
    dfl_distmat = pickle.load( open( basedir + 'dfl_distmat.p', "rb" ) )
    dfl_zctashapes = pickle.load( open( basedir + 'dfl_zctashapes.p', "rb" ) )
    dfl_zctacenters = pickle.load( open( basedir + 'dfl_zctacenters.p', "rb" ) )
    dfl = pickle.load( open( basedir + 'dfl.p', "rb" ) )

```

    building row 0 of 938
    building row 50 of 938
    building row 100 of 938
    building row 150 of 938
    building row 200 of 938
    building row 250 of 938
    building row 300 of 938
    building row 350 of 938
    building row 400 of 938
    building row 450 of 938
    building row 500 of 938
    building row 550 of 938
    building row 600 of 938
    building row 650 of 938
    building row 700 of 938
    building row 750 of 938
    building row 800 of 938
    building row 850 of 938
    building row 900 of 938



```python
# save the data:
#basedir = '/Users/matto/Documents/censusdata/arnhold_challenge/'
#pickle.dump( dfl_distmat, open( basedir + 'dfl_distmat.p', "wb" ) )
#pickle.dump( dfl_zctashapes, open( basedir + 'dfl_zctashapes.p', "wb" ) )
#pickle.dump( dfl_zctacenters, open( basedir + 'dfl_zctacenters.p', "wb" ) )
#pickle.dump( dfl, open( basedir + 'dfl.p', "wb" ) )

```


```python
dfl.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip</th>
      <th>zcta</th>
      <th>perc_pre1950_housing__CLPPP</th>
      <th>children_under6__CLPPP</th>
      <th>children_tested__CLPPP</th>
      <th>perc_tested__CLPPP</th>
      <th>bll_lt5__CLPPP</th>
      <th>bll_5to9__CLPPP</th>
      <th>capillary_ge10__CLPPP</th>
      <th>venous_10to19__CLPPP</th>
      <th>venous_20to44__CLPPP</th>
      <th>venous_ge45__CLPPP</th>
      <th>blltot_ge5__CLPPP</th>
      <th>blltot_ge10__CLPPP</th>
      <th>perc_bll_ge5__CLPPP</th>
      <th>perc_bll_ge10__CLPPP</th>
      <th>HC03_EST_VC01__S1701</th>
      <th>HC03_EST_VC03__S1701</th>
      <th>HC03_EST_VC04__S1701</th>
      <th>HC03_EST_VC05__S1701</th>
      <th>HC03_EST_VC06__S1701</th>
      <th>HC03_EST_VC09__S1701</th>
      <th>HC03_EST_VC10__S1701</th>
      <th>HC03_EST_VC13__S1701</th>
      <th>HC03_EST_VC14__S1701</th>
      <th>HC03_EST_VC20__S1701</th>
      <th>HC03_EST_VC22__S1701</th>
      <th>HC03_EST_VC23__S1701</th>
      <th>HC03_EST_VC26__S1701</th>
      <th>HC03_EST_VC27__S1701</th>
      <th>HC03_EST_VC28__S1701</th>
      <th>HC03_EST_VC29__S1701</th>
      <th>HC03_EST_VC30__S1701</th>
      <th>HC03_EST_VC33__S1701</th>
      <th>HC03_EST_VC34__S1701</th>
      <th>HC03_EST_VC35__S1701</th>
      <th>HC03_EST_VC36__S1701</th>
      <th>HC03_EST_VC37__S1701</th>
      <th>HC03_EST_VC38__S1701</th>
      <th>HC03_EST_VC39__S1701</th>
      <th>HC03_EST_VC42__S1701</th>
      <th>HC03_EST_VC43__S1701</th>
      <th>HC03_EST_VC44__S1701</th>
      <th>HC03_EST_VC45__S1701</th>
      <th>HC03_EST_VC54__S1701</th>
      <th>HC03_EST_VC55__S1701</th>
      <th>HC03_EST_VC56__S1701</th>
      <th>HC03_EST_VC60__S1701</th>
      <th>HC03_EST_VC61__S1701</th>
      <th>HC03_EST_VC62__S1701</th>
      <th>HC02_EST_VC01__S1702</th>
      <th>HC04_EST_VC01__S1702</th>
      <th>HC06_EST_VC01__S1702</th>
      <th>HC02_EST_VC02__S1702</th>
      <th>HC04_EST_VC02__S1702</th>
      <th>HC06_EST_VC02__S1702</th>
      <th>HC02_EST_VC06__S1702</th>
      <th>HC04_EST_VC06__S1702</th>
      <th>HC06_EST_VC06__S1702</th>
      <th>HC02_EST_VC07__S1702</th>
      <th>HC04_EST_VC07__S1702</th>
      <th>HC06_EST_VC07__S1702</th>
      <th>HC02_EST_VC16__S1702</th>
      <th>HC04_EST_VC16__S1702</th>
      <th>HC06_EST_VC16__S1702</th>
      <th>HC02_EST_VC18__S1702</th>
      <th>HC04_EST_VC18__S1702</th>
      <th>HC06_EST_VC18__S1702</th>
      <th>HC02_EST_VC19__S1702</th>
      <th>HC04_EST_VC19__S1702</th>
      <th>HC06_EST_VC19__S1702</th>
      <th>HC02_EST_VC21__S1702</th>
      <th>HC04_EST_VC21__S1702</th>
      <th>HC06_EST_VC21__S1702</th>
      <th>HC02_EST_VC23__S1702</th>
      <th>HC04_EST_VC23__S1702</th>
      <th>HC06_EST_VC23__S1702</th>
      <th>HC02_EST_VC24__S1702</th>
      <th>HC04_EST_VC24__S1702</th>
      <th>HC06_EST_VC24__S1702</th>
      <th>HC02_EST_VC27__S1702</th>
      <th>HC04_EST_VC27__S1702</th>
      <th>HC02_EST_VC28__S1702</th>
      <th>HC04_EST_VC28__S1702</th>
      <th>HC06_EST_VC28__S1702</th>
      <th>HC02_EST_VC29__S1702</th>
      <th>HC04_EST_VC29__S1702</th>
      <th>HC06_EST_VC29__S1702</th>
      <th>HC02_EST_VC30__S1702</th>
      <th>HC04_EST_VC30__S1702</th>
      <th>HC02_EST_VC33__S1702</th>
      <th>HC04_EST_VC33__S1702</th>
      <th>HC06_EST_VC33__S1702</th>
      <th>HC02_EST_VC34__S1702</th>
      <th>HC04_EST_VC34__S1702</th>
      <th>HC06_EST_VC34__S1702</th>
      <th>HC02_EST_VC35__S1702</th>
      <th>HC04_EST_VC35__S1702</th>
      <th>HC02_EST_VC39__S1702</th>
      <th>HC04_EST_VC39__S1702</th>
      <th>HC06_EST_VC39__S1702</th>
      <th>HC02_EST_VC40__S1702</th>
      <th>HC04_EST_VC40__S1702</th>
      <th>HC06_EST_VC40__S1702</th>
      <th>HC02_EST_VC41__S1702</th>
      <th>HC04_EST_VC41__S1702</th>
      <th>HC02_EST_VC45__S1702</th>
      <th>HC04_EST_VC45__S1702</th>
      <th>HC06_EST_VC45__S1702</th>
      <th>HC02_EST_VC46__S1702</th>
      <th>HC04_EST_VC46__S1702</th>
      <th>HC06_EST_VC46__S1702</th>
      <th>HC02_EST_VC47__S1702</th>
      <th>HC04_EST_VC47__S1702</th>
      <th>HC06_EST_VC47__S1702</th>
      <th>HC02_EST_VC48__S1702</th>
      <th>HC04_EST_VC48__S1702</th>
      <th>HC02_EST_VC01__S1401</th>
      <th>HC03_EST_VC01__S1401</th>
      <th>HC02_EST_VC02__S1401</th>
      <th>HC03_EST_VC02__S1401</th>
      <th>HC02_EST_VC03__S1401</th>
      <th>HC03_EST_VC03__S1401</th>
      <th>HC02_EST_VC04__S1401</th>
      <th>HC03_EST_VC04__S1401</th>
      <th>HC02_EST_VC05__S1401</th>
      <th>HC03_EST_VC05__S1401</th>
      <th>HC02_EST_VC06__S1401</th>
      <th>HC03_EST_VC06__S1401</th>
      <th>HC02_EST_VC07__S1401</th>
      <th>HC03_EST_VC07__S1401</th>
      <th>HC02_EST_VC08__S1401</th>
      <th>HC03_EST_VC08__S1401</th>
      <th>HC02_EST_VC09__S1401</th>
      <th>HC03_EST_VC09__S1401</th>
      <th>HC01_EST_VC12__S1401</th>
      <th>HC02_EST_VC12__S1401</th>
      <th>HC03_EST_VC12__S1401</th>
      <th>HC01_EST_VC13__S1401</th>
      <th>HC02_EST_VC13__S1401</th>
      <th>HC03_EST_VC13__S1401</th>
      <th>HC01_EST_VC14__S1401</th>
      <th>HC02_EST_VC14__S1401</th>
      <th>HC03_EST_VC14__S1401</th>
      <th>HC01_EST_VC15__S1401</th>
      <th>HC02_EST_VC15__S1401</th>
      <th>HC03_EST_VC15__S1401</th>
      <th>HC01_EST_VC16__S1401</th>
      <th>HC02_EST_VC16__S1401</th>
      <th>HC03_EST_VC16__S1401</th>
      <th>HC01_EST_VC17__S1401</th>
      <th>HC02_EST_VC17__S1401</th>
      <th>HC03_EST_VC17__S1401</th>
      <th>HC01_EST_VC18__S1401</th>
      <th>HC02_EST_VC18__S1401</th>
      <th>HC03_EST_VC18__S1401</th>
      <th>HC01_EST_VC19__S1401</th>
      <th>HC02_EST_VC19__S1401</th>
      <th>HC03_EST_VC19__S1401</th>
      <th>HC02_EST_VC22__S1401</th>
      <th>HC03_EST_VC22__S1401</th>
      <th>HC02_EST_VC24__S1401</th>
      <th>HC03_EST_VC24__S1401</th>
      <th>HC02_EST_VC26__S1401</th>
      <th>HC03_EST_VC26__S1401</th>
      <th>HC02_EST_VC29__S1401</th>
      <th>HC03_EST_VC29__S1401</th>
      <th>HC02_EST_VC31__S1401</th>
      <th>HC03_EST_VC31__S1401</th>
      <th>HC02_EST_VC33__S1401</th>
      <th>HC03_EST_VC33__S1401</th>
      <th>HC01_EST_VC16__S1501</th>
      <th>HC02_EST_VC16__S1501</th>
      <th>HC03_EST_VC16__S1501</th>
      <th>HC01_EST_VC17__S1501</th>
      <th>HC02_EST_VC17__S1501</th>
      <th>HC03_EST_VC17__S1501</th>
      <th>HC02_EST_VC01__S1601</th>
      <th>HC03_EST_VC01__S1601</th>
      <th>HC02_EST_VC03__S1601</th>
      <th>HC03_EST_VC03__S1601</th>
      <th>HC02_EST_VC04__S1601</th>
      <th>HC03_EST_VC04__S1601</th>
      <th>HC02_EST_VC05__S1601</th>
      <th>HC03_EST_VC05__S1601</th>
      <th>HC02_EST_VC10__S1601</th>
      <th>HC03_EST_VC10__S1601</th>
      <th>HC02_EST_VC12__S1601</th>
      <th>HC03_EST_VC12__S1601</th>
      <th>HC02_EST_VC14__S1601</th>
      <th>HC03_EST_VC14__S1601</th>
      <th>HC02_EST_VC16__S1601</th>
      <th>HC03_EST_VC16__S1601</th>
      <th>HC02_EST_VC28__S1601</th>
      <th>HC03_EST_VC28__S1601</th>
      <th>HC02_EST_VC30__S1601</th>
      <th>HC03_EST_VC30__S1601</th>
      <th>HC02_EST_VC31__S1601</th>
      <th>HC03_EST_VC31__S1601</th>
      <th>HC02_EST_VC32__S1601</th>
      <th>HC03_EST_VC32__S1601</th>
      <th>HC03_EST_VC01__S2701</th>
      <th>HC05_EST_VC01__S2701</th>
      <th>HC03_EST_VC04__S2701</th>
      <th>HC03_EST_VC05__S2701</th>
      <th>HC03_EST_VC06__S2701</th>
      <th>HC03_EST_VC08__S2701</th>
      <th>HC03_EST_VC11__S2701</th>
      <th>HC03_EST_VC12__S2701</th>
      <th>HC03_EST_VC15__S2701</th>
      <th>HC03_EST_VC16__S2701</th>
      <th>HC03_EST_VC22__S2701</th>
      <th>HC03_EST_VC24__S2701</th>
      <th>HC03_EST_VC25__S2701</th>
      <th>HC03_EST_VC28__S2701</th>
      <th>HC03_EST_VC29__S2701</th>
      <th>HC03_EST_VC30__S2701</th>
      <th>HC03_EST_VC31__S2701</th>
      <th>HC03_EST_VC34__S2701</th>
      <th>HC03_EST_VC35__S2701</th>
      <th>HC03_EST_VC36__S2701</th>
      <th>HC03_EST_VC37__S2701</th>
      <th>HC03_EST_VC38__S2701</th>
      <th>HC03_EST_VC41__S2701</th>
      <th>HC03_EST_VC42__S2701</th>
      <th>HC03_EST_VC43__S2701</th>
      <th>HC03_EST_VC44__S2701</th>
      <th>HC03_EST_VC45__S2701</th>
      <th>HC03_EST_VC48__S2701</th>
      <th>HC03_EST_VC49__S2701</th>
      <th>HC03_EST_VC50__S2701</th>
      <th>HC03_EST_VC51__S2701</th>
      <th>HC03_EST_VC54__S2701</th>
      <th>HC03_EST_VC55__S2701</th>
      <th>HC03_EST_VC56__S2701</th>
      <th>HC03_EST_VC57__S2701</th>
      <th>HC03_EST_VC58__S2701</th>
      <th>HC03_EST_VC59__S2701</th>
      <th>HC03_EST_VC62__S2701</th>
      <th>HC03_EST_VC63__S2701</th>
      <th>HC03_EST_VC64__S2701</th>
      <th>HC03_EST_VC65__S2701</th>
      <th>HC05_EST_VC68__S2701</th>
      <th>HC05_EST_VC69__S2701</th>
      <th>HC05_EST_VC70__S2701</th>
      <th>HC05_EST_VC71__S2701</th>
      <th>HC05_EST_VC72__S2701</th>
      <th>HC05_EST_VC73__S2701</th>
      <th>HC05_EST_VC74__S2701</th>
      <th>HC05_EST_VC75__S2701</th>
      <th>HC05_EST_VC76__S2701</th>
      <th>HC05_EST_VC77__S2701</th>
      <th>HC05_EST_VC78__S2701</th>
      <th>HC05_EST_VC79__S2701</th>
      <th>HC05_EST_VC80__S2701</th>
      <th>HC05_EST_VC81__S2701</th>
      <th>HC05_EST_VC82__S2701</th>
      <th>HC05_EST_VC83__S2701</th>
      <th>HC05_EST_VC84__S2701</th>
      <th>HC03_VC04__DP04</th>
      <th>HC03_VC05__DP04</th>
      <th>HC03_VC14__DP04</th>
      <th>HC03_VC15__DP04</th>
      <th>HC03_VC16__DP04</th>
      <th>HC03_VC17__DP04</th>
      <th>HC03_VC18__DP04</th>
      <th>HC03_VC19__DP04</th>
      <th>HC03_VC20__DP04</th>
      <th>HC03_VC21__DP04</th>
      <th>HC03_VC22__DP04</th>
      <th>HC03_VC27__DP04</th>
      <th>HC03_VC28__DP04</th>
      <th>HC03_VC29__DP04</th>
      <th>HC03_VC30__DP04</th>
      <th>HC03_VC31__DP04</th>
      <th>HC03_VC32__DP04</th>
      <th>HC03_VC33__DP04</th>
      <th>HC03_VC34__DP04</th>
      <th>HC03_VC35__DP04</th>
      <th>HC03_VC40__DP04</th>
      <th>HC03_VC41__DP04</th>
      <th>HC03_VC42__DP04</th>
      <th>HC03_VC43__DP04</th>
      <th>HC03_VC44__DP04</th>
      <th>HC03_VC45__DP04</th>
      <th>HC03_VC46__DP04</th>
      <th>HC03_VC47__DP04</th>
      <th>HC03_VC48__DP04</th>
      <th>HC03_VC54__DP04</th>
      <th>HC03_VC55__DP04</th>
      <th>HC03_VC56__DP04</th>
      <th>HC03_VC57__DP04</th>
      <th>HC03_VC58__DP04</th>
      <th>HC03_VC59__DP04</th>
      <th>HC03_VC64__DP04</th>
      <th>HC03_VC65__DP04</th>
      <th>HC03_VC74__DP04</th>
      <th>HC03_VC75__DP04</th>
      <th>HC03_VC76__DP04</th>
      <th>HC03_VC77__DP04</th>
      <th>HC03_VC78__DP04</th>
      <th>HC03_VC79__DP04</th>
      <th>HC03_VC84__DP04</th>
      <th>HC03_VC85__DP04</th>
      <th>HC03_VC86__DP04</th>
      <th>HC03_VC87__DP04</th>
      <th>HC03_VC92__DP04</th>
      <th>HC03_VC93__DP04</th>
      <th>HC03_VC94__DP04</th>
      <th>HC03_VC95__DP04</th>
      <th>HC03_VC96__DP04</th>
      <th>HC03_VC97__DP04</th>
      <th>HC03_VC98__DP04</th>
      <th>HC03_VC99__DP04</th>
      <th>HC03_VC100__DP04</th>
      <th>HC03_VC105__DP04</th>
      <th>HC03_VC106__DP04</th>
      <th>HC03_VC107__DP04</th>
      <th>HC03_VC112__DP04</th>
      <th>HC03_VC113__DP04</th>
      <th>HC03_VC114__DP04</th>
      <th>HC03_VC119__DP04</th>
      <th>HC03_VC120__DP04</th>
      <th>HC03_VC121__DP04</th>
      <th>HC03_VC122__DP04</th>
      <th>HC03_VC123__DP04</th>
      <th>HC03_VC124__DP04</th>
      <th>HC03_VC125__DP04</th>
      <th>HC03_VC126__DP04</th>
      <th>HC03_VC132__DP04</th>
      <th>HC03_VC133__DP04</th>
      <th>HC03_VC138__DP04</th>
      <th>HC03_VC139__DP04</th>
      <th>HC03_VC140__DP04</th>
      <th>HC03_VC141__DP04</th>
      <th>HC03_VC142__DP04</th>
      <th>HC03_VC143__DP04</th>
      <th>HC03_VC144__DP04</th>
      <th>HC03_VC148__DP04</th>
      <th>HC03_VC149__DP04</th>
      <th>HC03_VC150__DP04</th>
      <th>HC03_VC151__DP04</th>
      <th>HC03_VC152__DP04</th>
      <th>HC03_VC158__DP04</th>
      <th>HC03_VC159__DP04</th>
      <th>HC03_VC160__DP04</th>
      <th>HC03_VC161__DP04</th>
      <th>HC03_VC162__DP04</th>
      <th>HC03_VC168__DP04</th>
      <th>HC03_VC169__DP04</th>
      <th>HC03_VC170__DP04</th>
      <th>HC03_VC171__DP04</th>
      <th>HC03_VC172__DP04</th>
      <th>HC03_VC173__DP04</th>
      <th>HC03_VC174__DP04</th>
      <th>HC03_VC182__DP04</th>
      <th>HC03_VC183__DP04</th>
      <th>HC03_VC184__DP04</th>
      <th>HC03_VC185__DP04</th>
      <th>HC03_VC186__DP04</th>
      <th>HC03_VC187__DP04</th>
      <th>HC03_VC188__DP04</th>
      <th>HC03_VC197__DP04</th>
      <th>HC03_VC198__DP04</th>
      <th>HC03_VC199__DP04</th>
      <th>HC03_VC200__DP04</th>
      <th>HC03_VC201__DP04</th>
      <th>HC03_VC202__DP04</th>
      <th>perc_pre1950_housing__CLPPP_close</th>
      <th>HC03_EST_VC01__S1701_close</th>
      <th>HC03_EST_VC03__S1701_close</th>
      <th>HC03_EST_VC04__S1701_close</th>
      <th>HC03_EST_VC05__S1701_close</th>
      <th>HC03_EST_VC06__S1701_close</th>
      <th>HC03_EST_VC09__S1701_close</th>
      <th>HC03_EST_VC10__S1701_close</th>
      <th>HC03_EST_VC13__S1701_close</th>
      <th>HC03_EST_VC14__S1701_close</th>
      <th>HC03_EST_VC20__S1701_close</th>
      <th>HC03_EST_VC22__S1701_close</th>
      <th>HC03_EST_VC23__S1701_close</th>
      <th>HC03_EST_VC26__S1701_close</th>
      <th>HC03_EST_VC27__S1701_close</th>
      <th>HC03_EST_VC28__S1701_close</th>
      <th>HC03_EST_VC29__S1701_close</th>
      <th>HC03_EST_VC30__S1701_close</th>
      <th>HC03_EST_VC33__S1701_close</th>
      <th>HC03_EST_VC34__S1701_close</th>
      <th>HC03_EST_VC35__S1701_close</th>
      <th>HC03_EST_VC36__S1701_close</th>
      <th>HC03_EST_VC37__S1701_close</th>
      <th>HC03_EST_VC38__S1701_close</th>
      <th>HC03_EST_VC39__S1701_close</th>
      <th>HC03_EST_VC42__S1701_close</th>
      <th>HC03_EST_VC43__S1701_close</th>
      <th>HC03_EST_VC44__S1701_close</th>
      <th>HC03_EST_VC45__S1701_close</th>
      <th>HC03_EST_VC54__S1701_close</th>
      <th>HC03_EST_VC55__S1701_close</th>
      <th>HC03_EST_VC56__S1701_close</th>
      <th>HC03_EST_VC60__S1701_close</th>
      <th>HC03_EST_VC61__S1701_close</th>
      <th>HC03_EST_VC62__S1701_close</th>
      <th>HC02_EST_VC01__S1702_close</th>
      <th>HC04_EST_VC01__S1702_close</th>
      <th>HC06_EST_VC01__S1702_close</th>
      <th>HC02_EST_VC02__S1702_close</th>
      <th>HC04_EST_VC02__S1702_close</th>
      <th>HC06_EST_VC02__S1702_close</th>
      <th>HC02_EST_VC06__S1702_close</th>
      <th>HC04_EST_VC06__S1702_close</th>
      <th>HC06_EST_VC06__S1702_close</th>
      <th>HC02_EST_VC07__S1702_close</th>
      <th>HC04_EST_VC07__S1702_close</th>
      <th>HC06_EST_VC07__S1702_close</th>
      <th>HC02_EST_VC16__S1702_close</th>
      <th>HC04_EST_VC16__S1702_close</th>
      <th>HC06_EST_VC16__S1702_close</th>
      <th>HC02_EST_VC18__S1702_close</th>
      <th>HC04_EST_VC18__S1702_close</th>
      <th>HC06_EST_VC18__S1702_close</th>
      <th>HC02_EST_VC19__S1702_close</th>
      <th>HC04_EST_VC19__S1702_close</th>
      <th>HC06_EST_VC19__S1702_close</th>
      <th>HC02_EST_VC21__S1702_close</th>
      <th>HC04_EST_VC21__S1702_close</th>
      <th>HC06_EST_VC21__S1702_close</th>
      <th>HC02_EST_VC23__S1702_close</th>
      <th>HC04_EST_VC23__S1702_close</th>
      <th>HC06_EST_VC23__S1702_close</th>
      <th>HC02_EST_VC24__S1702_close</th>
      <th>HC04_EST_VC24__S1702_close</th>
      <th>HC06_EST_VC24__S1702_close</th>
      <th>HC02_EST_VC27__S1702_close</th>
      <th>HC04_EST_VC27__S1702_close</th>
      <th>HC02_EST_VC28__S1702_close</th>
      <th>HC04_EST_VC28__S1702_close</th>
      <th>HC06_EST_VC28__S1702_close</th>
      <th>HC02_EST_VC29__S1702_close</th>
      <th>HC04_EST_VC29__S1702_close</th>
      <th>HC06_EST_VC29__S1702_close</th>
      <th>HC02_EST_VC30__S1702_close</th>
      <th>HC04_EST_VC30__S1702_close</th>
      <th>HC02_EST_VC33__S1702_close</th>
      <th>HC04_EST_VC33__S1702_close</th>
      <th>HC06_EST_VC33__S1702_close</th>
      <th>HC02_EST_VC34__S1702_close</th>
      <th>HC04_EST_VC34__S1702_close</th>
      <th>HC06_EST_VC34__S1702_close</th>
      <th>HC02_EST_VC35__S1702_close</th>
      <th>HC04_EST_VC35__S1702_close</th>
      <th>HC02_EST_VC39__S1702_close</th>
      <th>HC04_EST_VC39__S1702_close</th>
      <th>HC06_EST_VC39__S1702_close</th>
      <th>HC02_EST_VC40__S1702_close</th>
      <th>HC04_EST_VC40__S1702_close</th>
      <th>HC06_EST_VC40__S1702_close</th>
      <th>HC02_EST_VC41__S1702_close</th>
      <th>HC04_EST_VC41__S1702_close</th>
      <th>HC02_EST_VC45__S1702_close</th>
      <th>HC04_EST_VC45__S1702_close</th>
      <th>HC06_EST_VC45__S1702_close</th>
      <th>HC02_EST_VC46__S1702_close</th>
      <th>HC04_EST_VC46__S1702_close</th>
      <th>HC06_EST_VC46__S1702_close</th>
      <th>HC02_EST_VC47__S1702_close</th>
      <th>HC04_EST_VC47__S1702_close</th>
      <th>HC06_EST_VC47__S1702_close</th>
      <th>HC02_EST_VC48__S1702_close</th>
      <th>HC04_EST_VC48__S1702_close</th>
      <th>HC02_EST_VC01__S1401_close</th>
      <th>HC03_EST_VC01__S1401_close</th>
      <th>HC02_EST_VC02__S1401_close</th>
      <th>HC03_EST_VC02__S1401_close</th>
      <th>HC02_EST_VC03__S1401_close</th>
      <th>HC03_EST_VC03__S1401_close</th>
      <th>HC02_EST_VC04__S1401_close</th>
      <th>HC03_EST_VC04__S1401_close</th>
      <th>HC02_EST_VC05__S1401_close</th>
      <th>HC03_EST_VC05__S1401_close</th>
      <th>HC02_EST_VC06__S1401_close</th>
      <th>HC03_EST_VC06__S1401_close</th>
      <th>HC02_EST_VC07__S1401_close</th>
      <th>HC03_EST_VC07__S1401_close</th>
      <th>HC02_EST_VC08__S1401_close</th>
      <th>HC03_EST_VC08__S1401_close</th>
      <th>HC02_EST_VC09__S1401_close</th>
      <th>HC03_EST_VC09__S1401_close</th>
      <th>HC01_EST_VC12__S1401_close</th>
      <th>HC02_EST_VC12__S1401_close</th>
      <th>HC03_EST_VC12__S1401_close</th>
      <th>HC01_EST_VC13__S1401_close</th>
      <th>HC02_EST_VC13__S1401_close</th>
      <th>HC03_EST_VC13__S1401_close</th>
      <th>HC01_EST_VC14__S1401_close</th>
      <th>HC02_EST_VC14__S1401_close</th>
      <th>HC03_EST_VC14__S1401_close</th>
      <th>HC01_EST_VC15__S1401_close</th>
      <th>HC02_EST_VC15__S1401_close</th>
      <th>HC03_EST_VC15__S1401_close</th>
      <th>HC01_EST_VC16__S1401_close</th>
      <th>HC02_EST_VC16__S1401_close</th>
      <th>HC03_EST_VC16__S1401_close</th>
      <th>HC01_EST_VC17__S1401_close</th>
      <th>HC02_EST_VC17__S1401_close</th>
      <th>HC03_EST_VC17__S1401_close</th>
      <th>HC01_EST_VC18__S1401_close</th>
      <th>HC02_EST_VC18__S1401_close</th>
      <th>HC03_EST_VC18__S1401_close</th>
      <th>HC01_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC19__S1401_close</th>
      <th>HC03_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC22__S1401_close</th>
      <th>HC03_EST_VC22__S1401_close</th>
      <th>HC02_EST_VC24__S1401_close</th>
      <th>HC03_EST_VC24__S1401_close</th>
      <th>HC02_EST_VC26__S1401_close</th>
      <th>HC03_EST_VC26__S1401_close</th>
      <th>HC02_EST_VC29__S1401_close</th>
      <th>HC03_EST_VC29__S1401_close</th>
      <th>HC02_EST_VC31__S1401_close</th>
      <th>HC03_EST_VC31__S1401_close</th>
      <th>HC02_EST_VC33__S1401_close</th>
      <th>HC03_EST_VC33__S1401_close</th>
      <th>HC01_EST_VC16__S1501_close</th>
      <th>HC02_EST_VC16__S1501_close</th>
      <th>HC03_EST_VC16__S1501_close</th>
      <th>HC01_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC17__S1501_close</th>
      <th>HC03_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC01__S1601_close</th>
      <th>HC03_EST_VC01__S1601_close</th>
      <th>HC02_EST_VC03__S1601_close</th>
      <th>HC03_EST_VC03__S1601_close</th>
      <th>HC02_EST_VC04__S1601_close</th>
      <th>HC03_EST_VC04__S1601_close</th>
      <th>HC02_EST_VC05__S1601_close</th>
      <th>HC03_EST_VC05__S1601_close</th>
      <th>HC02_EST_VC10__S1601_close</th>
      <th>HC03_EST_VC10__S1601_close</th>
      <th>HC02_EST_VC12__S1601_close</th>
      <th>HC03_EST_VC12__S1601_close</th>
      <th>HC02_EST_VC14__S1601_close</th>
      <th>HC03_EST_VC14__S1601_close</th>
      <th>HC02_EST_VC16__S1601_close</th>
      <th>HC03_EST_VC16__S1601_close</th>
      <th>HC02_EST_VC28__S1601_close</th>
      <th>HC03_EST_VC28__S1601_close</th>
      <th>HC02_EST_VC30__S1601_close</th>
      <th>HC03_EST_VC30__S1601_close</th>
      <th>HC02_EST_VC31__S1601_close</th>
      <th>HC03_EST_VC31__S1601_close</th>
      <th>HC02_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC01__S2701_close</th>
      <th>HC05_EST_VC01__S2701_close</th>
      <th>HC03_EST_VC04__S2701_close</th>
      <th>HC03_EST_VC05__S2701_close</th>
      <th>HC03_EST_VC06__S2701_close</th>
      <th>HC03_EST_VC08__S2701_close</th>
      <th>HC03_EST_VC11__S2701_close</th>
      <th>HC03_EST_VC12__S2701_close</th>
      <th>HC03_EST_VC15__S2701_close</th>
      <th>HC03_EST_VC16__S2701_close</th>
      <th>HC03_EST_VC22__S2701_close</th>
      <th>HC03_EST_VC24__S2701_close</th>
      <th>HC03_EST_VC25__S2701_close</th>
      <th>HC03_EST_VC28__S2701_close</th>
      <th>HC03_EST_VC29__S2701_close</th>
      <th>HC03_EST_VC30__S2701_close</th>
      <th>HC03_EST_VC31__S2701_close</th>
      <th>HC03_EST_VC34__S2701_close</th>
      <th>HC03_EST_VC35__S2701_close</th>
      <th>HC03_EST_VC36__S2701_close</th>
      <th>HC03_EST_VC37__S2701_close</th>
      <th>HC03_EST_VC38__S2701_close</th>
      <th>HC03_EST_VC41__S2701_close</th>
      <th>HC03_EST_VC42__S2701_close</th>
      <th>HC03_EST_VC43__S2701_close</th>
      <th>HC03_EST_VC44__S2701_close</th>
      <th>HC03_EST_VC45__S2701_close</th>
      <th>HC03_EST_VC48__S2701_close</th>
      <th>HC03_EST_VC49__S2701_close</th>
      <th>HC03_EST_VC50__S2701_close</th>
      <th>HC03_EST_VC51__S2701_close</th>
      <th>HC03_EST_VC54__S2701_close</th>
      <th>HC03_EST_VC55__S2701_close</th>
      <th>HC03_EST_VC56__S2701_close</th>
      <th>HC03_EST_VC57__S2701_close</th>
      <th>HC03_EST_VC58__S2701_close</th>
      <th>HC03_EST_VC59__S2701_close</th>
      <th>HC03_EST_VC62__S2701_close</th>
      <th>HC03_EST_VC63__S2701_close</th>
      <th>HC03_EST_VC64__S2701_close</th>
      <th>HC03_EST_VC65__S2701_close</th>
      <th>HC05_EST_VC68__S2701_close</th>
      <th>HC05_EST_VC69__S2701_close</th>
      <th>HC05_EST_VC70__S2701_close</th>
      <th>HC05_EST_VC71__S2701_close</th>
      <th>HC05_EST_VC72__S2701_close</th>
      <th>HC05_EST_VC73__S2701_close</th>
      <th>HC05_EST_VC74__S2701_close</th>
      <th>HC05_EST_VC75__S2701_close</th>
      <th>HC05_EST_VC76__S2701_close</th>
      <th>HC05_EST_VC77__S2701_close</th>
      <th>HC05_EST_VC78__S2701_close</th>
      <th>HC05_EST_VC79__S2701_close</th>
      <th>HC05_EST_VC80__S2701_close</th>
      <th>HC05_EST_VC81__S2701_close</th>
      <th>HC05_EST_VC82__S2701_close</th>
      <th>HC05_EST_VC83__S2701_close</th>
      <th>HC05_EST_VC84__S2701_close</th>
      <th>HC03_VC04__DP04_close</th>
      <th>HC03_VC05__DP04_close</th>
      <th>HC03_VC14__DP04_close</th>
      <th>HC03_VC15__DP04_close</th>
      <th>HC03_VC16__DP04_close</th>
      <th>HC03_VC17__DP04_close</th>
      <th>HC03_VC18__DP04_close</th>
      <th>HC03_VC19__DP04_close</th>
      <th>HC03_VC20__DP04_close</th>
      <th>HC03_VC21__DP04_close</th>
      <th>HC03_VC22__DP04_close</th>
      <th>HC03_VC27__DP04_close</th>
      <th>HC03_VC28__DP04_close</th>
      <th>HC03_VC29__DP04_close</th>
      <th>HC03_VC30__DP04_close</th>
      <th>HC03_VC31__DP04_close</th>
      <th>HC03_VC32__DP04_close</th>
      <th>HC03_VC33__DP04_close</th>
      <th>HC03_VC34__DP04_close</th>
      <th>HC03_VC35__DP04_close</th>
      <th>HC03_VC40__DP04_close</th>
      <th>HC03_VC41__DP04_close</th>
      <th>HC03_VC42__DP04_close</th>
      <th>HC03_VC43__DP04_close</th>
      <th>HC03_VC44__DP04_close</th>
      <th>HC03_VC45__DP04_close</th>
      <th>HC03_VC46__DP04_close</th>
      <th>HC03_VC47__DP04_close</th>
      <th>HC03_VC48__DP04_close</th>
      <th>HC03_VC54__DP04_close</th>
      <th>HC03_VC55__DP04_close</th>
      <th>HC03_VC56__DP04_close</th>
      <th>HC03_VC57__DP04_close</th>
      <th>HC03_VC58__DP04_close</th>
      <th>HC03_VC59__DP04_close</th>
      <th>HC03_VC64__DP04_close</th>
      <th>HC03_VC65__DP04_close</th>
      <th>HC03_VC74__DP04_close</th>
      <th>HC03_VC75__DP04_close</th>
      <th>HC03_VC76__DP04_close</th>
      <th>HC03_VC77__DP04_close</th>
      <th>HC03_VC78__DP04_close</th>
      <th>HC03_VC79__DP04_close</th>
      <th>HC03_VC84__DP04_close</th>
      <th>HC03_VC85__DP04_close</th>
      <th>HC03_VC86__DP04_close</th>
      <th>HC03_VC87__DP04_close</th>
      <th>HC03_VC92__DP04_close</th>
      <th>HC03_VC93__DP04_close</th>
      <th>HC03_VC94__DP04_close</th>
      <th>HC03_VC95__DP04_close</th>
      <th>HC03_VC96__DP04_close</th>
      <th>HC03_VC97__DP04_close</th>
      <th>HC03_VC98__DP04_close</th>
      <th>HC03_VC99__DP04_close</th>
      <th>HC03_VC100__DP04_close</th>
      <th>HC03_VC105__DP04_close</th>
      <th>HC03_VC106__DP04_close</th>
      <th>HC03_VC107__DP04_close</th>
      <th>HC03_VC112__DP04_close</th>
      <th>HC03_VC113__DP04_close</th>
      <th>HC03_VC114__DP04_close</th>
      <th>HC03_VC119__DP04_close</th>
      <th>HC03_VC120__DP04_close</th>
      <th>HC03_VC121__DP04_close</th>
      <th>HC03_VC122__DP04_close</th>
      <th>HC03_VC123__DP04_close</th>
      <th>HC03_VC124__DP04_close</th>
      <th>HC03_VC125__DP04_close</th>
      <th>HC03_VC126__DP04_close</th>
      <th>HC03_VC132__DP04_close</th>
      <th>HC03_VC133__DP04_close</th>
      <th>HC03_VC138__DP04_close</th>
      <th>HC03_VC139__DP04_close</th>
      <th>HC03_VC140__DP04_close</th>
      <th>HC03_VC141__DP04_close</th>
      <th>HC03_VC142__DP04_close</th>
      <th>HC03_VC143__DP04_close</th>
      <th>HC03_VC144__DP04_close</th>
      <th>HC03_VC148__DP04_close</th>
      <th>HC03_VC149__DP04_close</th>
      <th>HC03_VC150__DP04_close</th>
      <th>HC03_VC151__DP04_close</th>
      <th>HC03_VC152__DP04_close</th>
      <th>HC03_VC158__DP04_close</th>
      <th>HC03_VC159__DP04_close</th>
      <th>HC03_VC160__DP04_close</th>
      <th>HC03_VC161__DP04_close</th>
      <th>HC03_VC162__DP04_close</th>
      <th>HC03_VC168__DP04_close</th>
      <th>HC03_VC169__DP04_close</th>
      <th>HC03_VC170__DP04_close</th>
      <th>HC03_VC171__DP04_close</th>
      <th>HC03_VC172__DP04_close</th>
      <th>HC03_VC173__DP04_close</th>
      <th>HC03_VC174__DP04_close</th>
      <th>HC03_VC182__DP04_close</th>
      <th>HC03_VC183__DP04_close</th>
      <th>HC03_VC184__DP04_close</th>
      <th>HC03_VC185__DP04_close</th>
      <th>HC03_VC186__DP04_close</th>
      <th>HC03_VC187__DP04_close</th>
      <th>HC03_VC188__DP04_close</th>
      <th>HC03_VC197__DP04_close</th>
      <th>HC03_VC198__DP04_close</th>
      <th>HC03_VC199__DP04_close</th>
      <th>HC03_VC200__DP04_close</th>
      <th>HC03_VC201__DP04_close</th>
      <th>HC03_VC202__DP04_close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>924.000000</td>
      <td>936.000000</td>
      <td>938.000000</td>
      <td>936.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>937.000000</td>
      <td>930.000000</td>
      <td>930.000000</td>
      <td>936.000000</td>
      <td>936.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>880.000000</td>
      <td>866.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>930.000000</td>
      <td>937.000000</td>
      <td>935.000000</td>
      <td>931.000000</td>
      <td>936.000000</td>
      <td>936.000000</td>
      <td>936.000000</td>
      <td>935.000000</td>
      <td>926.000000</td>
      <td>914.000000</td>
      <td>905.000000</td>
      <td>937.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>936.000000</td>
      <td>935.000000</td>
      <td>933.000000</td>
      <td>931.000000</td>
      <td>924.000000</td>
      <td>924.000000</td>
      <td>932.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>915.000000</td>
      <td>930.000000</td>
      <td>927.000000</td>
      <td>900.000000</td>
      <td>937.000000</td>
      <td>937.00000</td>
      <td>915.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>913.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>913.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>901.000000</td>
      <td>931.000000</td>
      <td>931.000000</td>
      <td>869.000000</td>
      <td>928.000000</td>
      <td>926.000000</td>
      <td>860.000000</td>
      <td>916.000000</td>
      <td>896.000000</td>
      <td>819.000000</td>
      <td>933.000000</td>
      <td>932.000000</td>
      <td>889.000000</td>
      <td>911.000000</td>
      <td>900.000000</td>
      <td>933.000000</td>
      <td>932.000000</td>
      <td>882.000000</td>
      <td>930.000000</td>
      <td>930.000000</td>
      <td>883.00000</td>
      <td>923.000000</td>
      <td>921.000000</td>
      <td>935.000000</td>
      <td>934.000000</td>
      <td>898.000000</td>
      <td>927.000000</td>
      <td>925.000000</td>
      <td>889.000000</td>
      <td>898.000000</td>
      <td>890.000000</td>
      <td>934.000000</td>
      <td>933.000000</td>
      <td>904.000000</td>
      <td>931.000000</td>
      <td>929.000000</td>
      <td>884.000000</td>
      <td>902.000000</td>
      <td>898.000000</td>
      <td>930.000000</td>
      <td>927.000000</td>
      <td>841.000000</td>
      <td>931.000000</td>
      <td>928.000000</td>
      <td>892.000000</td>
      <td>933.000000</td>
      <td>932.000000</td>
      <td>855.000000</td>
      <td>903.000000</td>
      <td>902.000000</td>
      <td>931.000000</td>
      <td>931.000000</td>
      <td>876.000000</td>
      <td>876.000000</td>
      <td>930.000000</td>
      <td>930.000000</td>
      <td>870.000000</td>
      <td>870.000000</td>
      <td>919.000000</td>
      <td>919.000000</td>
      <td>921.000000</td>
      <td>921.000000</td>
      <td>923.000000</td>
      <td>923.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>820.000000</td>
      <td>820.000000</td>
      <td>897.000000</td>
      <td>860.000000</td>
      <td>860.000000</td>
      <td>919.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>923.000000</td>
      <td>923.000000</td>
      <td>923.000000</td>
      <td>921.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>902.000000</td>
      <td>888.000000</td>
      <td>888.000000</td>
      <td>921.000000</td>
      <td>881.000000</td>
      <td>881.000000</td>
      <td>929.000000</td>
      <td>855.000000</td>
      <td>855.000000</td>
      <td>937.000000</td>
      <td>879.000000</td>
      <td>879.000000</td>
      <td>921.000000</td>
      <td>921.000000</td>
      <td>895.000000</td>
      <td>895.000000</td>
      <td>908.000000</td>
      <td>908.000000</td>
      <td>898.000000</td>
      <td>898.000000</td>
      <td>848.000000</td>
      <td>848.000000</td>
      <td>865.000000</td>
      <td>865.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>908.000000</td>
      <td>908.000000</td>
      <td>853.000000</td>
      <td>853.000000</td>
      <td>864.000000</td>
      <td>864.000000</td>
      <td>853.000000</td>
      <td>853.000000</td>
      <td>811.000000</td>
      <td>811.000000</td>
      <td>864.000000</td>
      <td>864.000000</td>
      <td>817.000000</td>
      <td>817.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>902.000000</td>
      <td>902.000000</td>
      <td>828.000000</td>
      <td>828.000000</td>
      <td>878.00000</td>
      <td>878.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>930.000000</td>
      <td>936.000000</td>
      <td>936.000000</td>
      <td>924.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>881.000000</td>
      <td>937.000000</td>
      <td>866.000000</td>
      <td>937.000000</td>
      <td>884.000000</td>
      <td>858.000000</td>
      <td>810.000000</td>
      <td>937.000000</td>
      <td>930.000000</td>
      <td>937.000000</td>
      <td>935.000000</td>
      <td>931.000000</td>
      <td>937.000000</td>
      <td>936.000000</td>
      <td>936.000000</td>
      <td>926.000000</td>
      <td>936.000000</td>
      <td>937.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>936.000000</td>
      <td>937.000000</td>
      <td>932.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>921.000000</td>
      <td>916.000000</td>
      <td>937.000000</td>
      <td>931.000000</td>
      <td>928.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.00000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>934.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>935.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>916.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.00000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>937.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>937.000000</td>
      <td>937.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>936.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.00000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.00000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.00000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
      <td>938.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48970.251599</td>
      <td>48970.309168</td>
      <td>29.890693</td>
      <td>766.302350</td>
      <td>156.778252</td>
      <td>19.102885</td>
      <td>150.718550</td>
      <td>5.094883</td>
      <td>0.270789</td>
      <td>0.550107</td>
      <td>0.130064</td>
      <td>0.013859</td>
      <td>6.059701</td>
      <td>0.694030</td>
      <td>2.652665</td>
      <td>0.171429</td>
      <td>15.366382</td>
      <td>22.241075</td>
      <td>21.733978</td>
      <td>15.094658</td>
      <td>8.041880</td>
      <td>14.284312</td>
      <td>16.405656</td>
      <td>15.132551</td>
      <td>14.441302</td>
      <td>24.819545</td>
      <td>23.907275</td>
      <td>14.115688</td>
      <td>12.266275</td>
      <td>23.764839</td>
      <td>13.905229</td>
      <td>10.913369</td>
      <td>5.022127</td>
      <td>10.742842</td>
      <td>8.023184</td>
      <td>6.847009</td>
      <td>9.159572</td>
      <td>30.761555</td>
      <td>28.784245</td>
      <td>33.576685</td>
      <td>13.707577</td>
      <td>3.210695</td>
      <td>17.426738</td>
      <td>19.708013</td>
      <td>27.430053</td>
      <td>24.987138</td>
      <td>29.920301</td>
      <td>4.496861</td>
      <td>38.607251</td>
      <td>37.287446</td>
      <td>11.110566</td>
      <td>6.069157</td>
      <td>31.295519</td>
      <td>19.452688</td>
      <td>9.936030</td>
      <td>42.647778</td>
      <td>10.983031</td>
      <td>6.03810</td>
      <td>31.100656</td>
      <td>10.190608</td>
      <td>5.860192</td>
      <td>30.245893</td>
      <td>9.901814</td>
      <td>5.540768</td>
      <td>29.785104</td>
      <td>9.057602</td>
      <td>4.466488</td>
      <td>27.288346</td>
      <td>3.642427</td>
      <td>2.241031</td>
      <td>10.069850</td>
      <td>4.279418</td>
      <td>3.396436</td>
      <td>9.401860</td>
      <td>26.688646</td>
      <td>16.143527</td>
      <td>44.665690</td>
      <td>6.487460</td>
      <td>4.234549</td>
      <td>16.669741</td>
      <td>21.589572</td>
      <td>14.285778</td>
      <td>12.770954</td>
      <td>7.316309</td>
      <td>32.936054</td>
      <td>11.265591</td>
      <td>5.457742</td>
      <td>32.84983</td>
      <td>4.229902</td>
      <td>2.778719</td>
      <td>5.329305</td>
      <td>3.937687</td>
      <td>12.669933</td>
      <td>16.913808</td>
      <td>7.452541</td>
      <td>38.493138</td>
      <td>24.731514</td>
      <td>14.632022</td>
      <td>8.518522</td>
      <td>4.146838</td>
      <td>24.460398</td>
      <td>12.725242</td>
      <td>5.959634</td>
      <td>35.709163</td>
      <td>17.099002</td>
      <td>12.505011</td>
      <td>18.920968</td>
      <td>9.853506</td>
      <td>53.523543</td>
      <td>16.360580</td>
      <td>9.384914</td>
      <td>32.280942</td>
      <td>4.050482</td>
      <td>3.162768</td>
      <td>10.844561</td>
      <td>2.305094</td>
      <td>1.745898</td>
      <td>88.289366</td>
      <td>11.710634</td>
      <td>72.686530</td>
      <td>27.314041</td>
      <td>91.032366</td>
      <td>8.967742</td>
      <td>89.488391</td>
      <td>10.512069</td>
      <td>89.429162</td>
      <td>10.571273</td>
      <td>90.644517</td>
      <td>9.355809</td>
      <td>92.657096</td>
      <td>7.343229</td>
      <td>84.393784</td>
      <td>15.606434</td>
      <td>74.858171</td>
      <td>25.143659</td>
      <td>45.344147</td>
      <td>71.780581</td>
      <td>28.220814</td>
      <td>95.672252</td>
      <td>88.616993</td>
      <td>11.383115</td>
      <td>97.874756</td>
      <td>90.729686</td>
      <td>9.270964</td>
      <td>96.048643</td>
      <td>92.906645</td>
      <td>7.093900</td>
      <td>70.202439</td>
      <td>88.870833</td>
      <td>11.129505</td>
      <td>35.600543</td>
      <td>85.617707</td>
      <td>14.382747</td>
      <td>12.015501</td>
      <td>82.493918</td>
      <td>17.506667</td>
      <td>2.406937</td>
      <td>78.092719</td>
      <td>21.908305</td>
      <td>82.948534</td>
      <td>17.051792</td>
      <td>83.255307</td>
      <td>16.745475</td>
      <td>82.511344</td>
      <td>17.488877</td>
      <td>86.081737</td>
      <td>13.918597</td>
      <td>86.251061</td>
      <td>13.749528</td>
      <td>86.057572</td>
      <td>13.943006</td>
      <td>88.895945</td>
      <td>87.691889</td>
      <td>90.208751</td>
      <td>20.927001</td>
      <td>20.669050</td>
      <td>21.254749</td>
      <td>98.095197</td>
      <td>1.904803</td>
      <td>71.877533</td>
      <td>28.123348</td>
      <td>72.303634</td>
      <td>27.697186</td>
      <td>77.405093</td>
      <td>22.595833</td>
      <td>72.303634</td>
      <td>27.697186</td>
      <td>70.757707</td>
      <td>29.242787</td>
      <td>77.405093</td>
      <td>22.595833</td>
      <td>79.363892</td>
      <td>20.636842</td>
      <td>98.909285</td>
      <td>1.090715</td>
      <td>77.444789</td>
      <td>22.555876</td>
      <td>78.711957</td>
      <td>21.288768</td>
      <td>77.16287</td>
      <td>22.837472</td>
      <td>11.533938</td>
      <td>88.466275</td>
      <td>4.952688</td>
      <td>17.219017</td>
      <td>0.338141</td>
      <td>25.849242</td>
      <td>12.851014</td>
      <td>10.243116</td>
      <td>11.527321</td>
      <td>11.359658</td>
      <td>12.738025</td>
      <td>11.088687</td>
      <td>17.654965</td>
      <td>11.122732</td>
      <td>19.654864</td>
      <td>10.354079</td>
      <td>29.452593</td>
      <td>12.496478</td>
      <td>17.554516</td>
      <td>14.438634</td>
      <td>12.554545</td>
      <td>6.419979</td>
      <td>13.431377</td>
      <td>17.621261</td>
      <td>15.069444</td>
      <td>38.165767</td>
      <td>7.697329</td>
      <td>13.431377</td>
      <td>11.293904</td>
      <td>23.162888</td>
      <td>9.289423</td>
      <td>11.471185</td>
      <td>18.583476</td>
      <td>14.510064</td>
      <td>9.767987</td>
      <td>6.841042</td>
      <td>5.578603</td>
      <td>11.572785</td>
      <td>21.531364</td>
      <td>16.232866</td>
      <td>7.383778</td>
      <td>68.592423</td>
      <td>49.677695</td>
      <td>57.858164</td>
      <td>44.583458</td>
      <td>13.659552</td>
      <td>4.683031</td>
      <td>1.643757</td>
      <td>0.409498</td>
      <td>36.908431</td>
      <td>17.272145</td>
      <td>20.589434</td>
      <td>3.771825</td>
      <td>19.034365</td>
      <td>13.138527</td>
      <td>2.666596</td>
      <td>0.360299</td>
      <td>11.533938</td>
      <td>78.398826</td>
      <td>21.601387</td>
      <td>78.461046</td>
      <td>2.563074</td>
      <td>1.770864</td>
      <td>1.821025</td>
      <td>2.390502</td>
      <td>1.918356</td>
      <td>2.707898</td>
      <td>8.340448</td>
      <td>0.028815</td>
      <td>0.420598</td>
      <td>10.834792</td>
      <td>14.230843</td>
      <td>10.347279</td>
      <td>15.893170</td>
      <td>10.773426</td>
      <td>12.073959</td>
      <td>6.706724</td>
      <td>18.724226</td>
      <td>1.310459</td>
      <td>1.992316</td>
      <td>6.684632</td>
      <td>14.482711</td>
      <td>21.009605</td>
      <td>19.953895</td>
      <td>13.889648</td>
      <td>9.463180</td>
      <td>11.214194</td>
      <td>1.407257</td>
      <td>8.031804</td>
      <td>26.523372</td>
      <td>45.10555</td>
      <td>15.300640</td>
      <td>3.632657</td>
      <td>79.086126</td>
      <td>20.914088</td>
      <td>18.477588</td>
      <td>36.262860</td>
      <td>20.841729</td>
      <td>10.462753</td>
      <td>7.677801</td>
      <td>6.280470</td>
      <td>6.130630</td>
      <td>31.736073</td>
      <td>40.564248</td>
      <td>21.568943</td>
      <td>55.579402</td>
      <td>21.221772</td>
      <td>7.499466</td>
      <td>3.668517</td>
      <td>0.052508</td>
      <td>10.051547</td>
      <td>0.029989</td>
      <td>1.518783</td>
      <td>0.378762</td>
      <td>0.506937</td>
      <td>0.670224</td>
      <td>2.718143</td>
      <td>98.432337</td>
      <td>1.227321</td>
      <td>0.337780</td>
      <td>16.851547</td>
      <td>27.389221</td>
      <td>19.667769</td>
      <td>15.025827</td>
      <td>12.214088</td>
      <td>6.128388</td>
      <td>2.069797</td>
      <td>0.656670</td>
      <td>59.384739</td>
      <td>40.615368</td>
      <td>0.270557</td>
      <td>2.897430</td>
      <td>9.576445</td>
      <td>24.123126</td>
      <td>34.264561</td>
      <td>16.815846</td>
      <td>12.056210</td>
      <td>0.778396</td>
      <td>5.539465</td>
      <td>15.256471</td>
      <td>21.258610</td>
      <td>57.170267</td>
      <td>39.313169</td>
      <td>15.911028</td>
      <td>11.390043</td>
      <td>7.589079</td>
      <td>25.796360</td>
      <td>36.239893</td>
      <td>21.482246</td>
      <td>13.462888</td>
      <td>8.058610</td>
      <td>5.255294</td>
      <td>3.482246</td>
      <td>12.019465</td>
      <td>1.728603</td>
      <td>4.091266</td>
      <td>13.241376</td>
      <td>34.208843</td>
      <td>25.283297</td>
      <td>16.768231</td>
      <td>4.677511</td>
      <td>13.918668</td>
      <td>12.295306</td>
      <td>12.051965</td>
      <td>11.454258</td>
      <td>8.490066</td>
      <td>41.788646</td>
      <td>30.335009</td>
      <td>15.739654</td>
      <td>22.703454</td>
      <td>22.213271</td>
      <td>15.472340</td>
      <td>8.194638</td>
      <td>14.674270</td>
      <td>16.759893</td>
      <td>15.496429</td>
      <td>14.802420</td>
      <td>25.201214</td>
      <td>24.801507</td>
      <td>14.433305</td>
      <td>12.543817</td>
      <td>24.094845</td>
      <td>14.249829</td>
      <td>11.165464</td>
      <td>5.199909</td>
      <td>11.084899</td>
      <td>8.325800</td>
      <td>7.164451</td>
      <td>9.452537</td>
      <td>31.277406</td>
      <td>29.428284</td>
      <td>33.820254</td>
      <td>14.054307</td>
      <td>3.325917</td>
      <td>18.013950</td>
      <td>20.102084</td>
      <td>27.660517</td>
      <td>25.247154</td>
      <td>30.174264</td>
      <td>4.540757</td>
      <td>38.985698</td>
      <td>37.651135</td>
      <td>11.43944</td>
      <td>6.322425</td>
      <td>31.607228</td>
      <td>19.911416</td>
      <td>10.308278</td>
      <td>42.927456</td>
      <td>11.303502</td>
      <td>6.292223</td>
      <td>31.395988</td>
      <td>10.445490</td>
      <td>6.028145</td>
      <td>30.457393</td>
      <td>10.104499</td>
      <td>5.639659</td>
      <td>29.952399</td>
      <td>9.492228</td>
      <td>4.709568</td>
      <td>27.731219</td>
      <td>3.748017</td>
      <td>2.305341</td>
      <td>10.393134</td>
      <td>4.356471</td>
      <td>3.509851</td>
      <td>9.282921</td>
      <td>26.995583</td>
      <td>16.520057</td>
      <td>45.612123</td>
      <td>6.608635</td>
      <td>4.296397</td>
      <td>16.996759</td>
      <td>21.638467</td>
      <td>14.276425</td>
      <td>13.224829</td>
      <td>7.600959</td>
      <td>33.352017</td>
      <td>11.653806</td>
      <td>5.717159</td>
      <td>32.930210</td>
      <td>4.489227</td>
      <td>3.022521</td>
      <td>5.486525</td>
      <td>4.098918</td>
      <td>13.131933</td>
      <td>17.206741</td>
      <td>7.643579</td>
      <td>38.863317</td>
      <td>25.190677</td>
      <td>14.984129</td>
      <td>8.750624</td>
      <td>4.311935</td>
      <td>24.754383</td>
      <td>13.002292</td>
      <td>6.150076</td>
      <td>35.990334</td>
      <td>17.626160</td>
      <td>12.950458</td>
      <td>19.308698</td>
      <td>10.071075</td>
      <td>53.503484</td>
      <td>16.846727</td>
      <td>9.674168</td>
      <td>32.912226</td>
      <td>4.301525</td>
      <td>3.361487</td>
      <td>11.102608</td>
      <td>2.425144</td>
      <td>1.791862</td>
      <td>88.189716</td>
      <td>11.810284</td>
      <td>73.326089</td>
      <td>26.674476</td>
      <td>90.991466</td>
      <td>9.008683</td>
      <td>89.341876</td>
      <td>10.658555</td>
      <td>89.482886</td>
      <td>10.517546</td>
      <td>90.554184</td>
      <td>9.446093</td>
      <td>92.508673</td>
      <td>7.491818</td>
      <td>84.302539</td>
      <td>15.697695</td>
      <td>75.124401</td>
      <td>24.877235</td>
      <td>45.001956</td>
      <td>72.541173</td>
      <td>27.460128</td>
      <td>95.719465</td>
      <td>88.671233</td>
      <td>11.328852</td>
      <td>97.900322</td>
      <td>90.651923</td>
      <td>9.348653</td>
      <td>95.904517</td>
      <td>92.745156</td>
      <td>7.255382</td>
      <td>69.856914</td>
      <td>88.920613</td>
      <td>11.079680</td>
      <td>35.763262</td>
      <td>85.392905</td>
      <td>14.607575</td>
      <td>12.159936</td>
      <td>82.283837</td>
      <td>17.716764</td>
      <td>2.460698</td>
      <td>78.294135</td>
      <td>21.706988</td>
      <td>82.818157</td>
      <td>17.182146</td>
      <td>83.508943</td>
      <td>16.491771</td>
      <td>82.200604</td>
      <td>17.799604</td>
      <td>85.911764</td>
      <td>14.088555</td>
      <td>86.285515</td>
      <td>13.715059</td>
      <td>86.009163</td>
      <td>13.991493</td>
      <td>88.784163</td>
      <td>87.549989</td>
      <td>90.138833</td>
      <td>20.869835</td>
      <td>20.559808</td>
      <td>21.257655</td>
      <td>98.038603</td>
      <td>1.961397</td>
      <td>72.002923</td>
      <td>27.997994</td>
      <td>72.206557</td>
      <td>27.794355</td>
      <td>77.480762</td>
      <td>22.520171</td>
      <td>72.206557</td>
      <td>27.794355</td>
      <td>71.367480</td>
      <td>28.633104</td>
      <td>77.480762</td>
      <td>22.520171</td>
      <td>79.584437</td>
      <td>20.416384</td>
      <td>98.883529</td>
      <td>1.116471</td>
      <td>77.386907</td>
      <td>22.613792</td>
      <td>78.257656</td>
      <td>21.743172</td>
      <td>77.283927</td>
      <td>22.716409</td>
      <td>11.688497</td>
      <td>88.311738</td>
      <td>4.919705</td>
      <td>17.410805</td>
      <td>0.358609</td>
      <td>25.735126</td>
      <td>13.021125</td>
      <td>10.382377</td>
      <td>11.688001</td>
      <td>11.543273</td>
      <td>12.915409</td>
      <td>11.259035</td>
      <td>17.445263</td>
      <td>11.263993</td>
      <td>19.744588</td>
      <td>10.515730</td>
      <td>28.752790</td>
      <td>12.707132</td>
      <td>17.642612</td>
      <td>14.688113</td>
      <td>12.777756</td>
      <td>6.594078</td>
      <td>13.630698</td>
      <td>17.819627</td>
      <td>15.277687</td>
      <td>38.142644</td>
      <td>7.859051</td>
      <td>13.630698</td>
      <td>11.478358</td>
      <td>23.389179</td>
      <td>9.417159</td>
      <td>11.629824</td>
      <td>18.742857</td>
      <td>14.684334</td>
      <td>9.893950</td>
      <td>6.878840</td>
      <td>5.652902</td>
      <td>11.728523</td>
      <td>21.620197</td>
      <td>16.411676</td>
      <td>7.451668</td>
      <td>68.180666</td>
      <td>49.407308</td>
      <td>57.590176</td>
      <td>44.332468</td>
      <td>13.558673</td>
      <td>4.666210</td>
      <td>1.630368</td>
      <td>0.407297</td>
      <td>36.968044</td>
      <td>17.497532</td>
      <td>20.403801</td>
      <td>3.772942</td>
      <td>19.307719</td>
      <td>13.353172</td>
      <td>2.648881</td>
      <td>0.369584</td>
      <td>11.688497</td>
      <td>78.711407</td>
      <td>21.288785</td>
      <td>78.165704</td>
      <td>2.575426</td>
      <td>1.840101</td>
      <td>1.881679</td>
      <td>2.409616</td>
      <td>1.996546</td>
      <td>2.833065</td>
      <td>8.269563</td>
      <td>0.030597</td>
      <td>0.410384</td>
      <td>10.659670</td>
      <td>13.948262</td>
      <td>10.175752</td>
      <td>15.703214</td>
      <td>10.793204</td>
      <td>12.270576</td>
      <td>6.861583</td>
      <td>19.183209</td>
      <td>1.298204</td>
      <td>2.005293</td>
      <td>6.794979</td>
      <td>14.502628</td>
      <td>21.072186</td>
      <td>19.979781</td>
      <td>13.915730</td>
      <td>9.336567</td>
      <td>11.095885</td>
      <td>1.396999</td>
      <td>8.211946</td>
      <td>26.538699</td>
      <td>45.052628</td>
      <td>15.205959</td>
      <td>3.593955</td>
      <td>78.492655</td>
      <td>21.507601</td>
      <td>18.773273</td>
      <td>36.263449</td>
      <td>20.612836</td>
      <td>10.425672</td>
      <td>7.650272</td>
      <td>6.277569</td>
      <td>6.257836</td>
      <td>31.922910</td>
      <td>40.429542</td>
      <td>21.390139</td>
      <td>55.889590</td>
      <td>21.027708</td>
      <td>7.529259</td>
      <td>3.592777</td>
      <td>0.051151</td>
      <td>9.989733</td>
      <td>0.03177</td>
      <td>1.507281</td>
      <td>0.382249</td>
      <td>0.509083</td>
      <td>0.661066</td>
      <td>2.724835</td>
      <td>98.408625</td>
      <td>1.248481</td>
      <td>0.340016</td>
      <td>17.313811</td>
      <td>27.649723</td>
      <td>19.668038</td>
      <td>14.775794</td>
      <td>11.999765</td>
      <td>5.972340</td>
      <td>2.016663</td>
      <td>0.605171</td>
      <td>59.395453</td>
      <td>40.604675</td>
      <td>0.275970</td>
      <td>2.951439</td>
      <td>9.701620</td>
      <td>24.345085</td>
      <td>34.368033</td>
      <td>16.591077</td>
      <td>11.771199</td>
      <td>0.803108</td>
      <td>5.63718</td>
      <td>15.309083</td>
      <td>21.324238</td>
      <td>56.929611</td>
      <td>39.294462</td>
      <td>15.966402</td>
      <td>11.341279</td>
      <td>7.536530</td>
      <td>25.860890</td>
      <td>36.082148</td>
      <td>21.47783</td>
      <td>13.452761</td>
      <td>8.122889</td>
      <td>5.245304</td>
      <td>3.530043</td>
      <td>12.090666</td>
      <td>1.670483</td>
      <td>4.167976</td>
      <td>13.219311</td>
      <td>34.470462</td>
      <td>25.073134</td>
      <td>16.704970</td>
      <td>4.692630</td>
      <td>13.885503</td>
      <td>12.280450</td>
      <td>11.911038</td>
      <td>11.308982</td>
      <td>8.505842</td>
      <td>42.106700</td>
    </tr>
    <tr>
      <th>std</th>
      <td>607.251727</td>
      <td>607.238509</td>
      <td>16.369383</td>
      <td>959.805514</td>
      <td>264.309939</td>
      <td>13.730382</td>
      <td>249.663337</td>
      <td>14.793295</td>
      <td>0.893006</td>
      <td>2.488853</td>
      <td>0.656738</td>
      <td>0.116969</td>
      <td>18.224927</td>
      <td>3.097006</td>
      <td>4.380886</td>
      <td>0.655699</td>
      <td>9.526342</td>
      <td>16.075928</td>
      <td>16.108186</td>
      <td>9.289976</td>
      <td>6.367202</td>
      <td>9.560630</td>
      <td>9.938035</td>
      <td>9.486108</td>
      <td>9.404848</td>
      <td>24.692621</td>
      <td>24.115883</td>
      <td>9.238210</td>
      <td>7.484002</td>
      <td>13.456843</td>
      <td>8.604515</td>
      <td>7.390144</td>
      <td>5.432002</td>
      <td>7.412859</td>
      <td>6.246863</td>
      <td>6.388246</td>
      <td>7.039230</td>
      <td>16.992203</td>
      <td>18.063220</td>
      <td>21.278838</td>
      <td>8.389643</td>
      <td>3.895871</td>
      <td>11.128703</td>
      <td>10.850544</td>
      <td>11.175836</td>
      <td>12.773996</td>
      <td>13.289259</td>
      <td>6.411861</td>
      <td>17.043201</td>
      <td>14.540924</td>
      <td>8.337294</td>
      <td>5.990546</td>
      <td>17.956674</td>
      <td>14.293786</td>
      <td>9.985086</td>
      <td>23.163756</td>
      <td>8.299472</td>
      <td>6.00793</td>
      <td>18.154387</td>
      <td>8.174232</td>
      <td>6.154781</td>
      <td>18.948256</td>
      <td>7.923079</td>
      <td>5.591198</td>
      <td>19.057318</td>
      <td>8.301341</td>
      <td>5.554856</td>
      <td>19.933959</td>
      <td>5.324770</td>
      <td>4.876762</td>
      <td>14.832334</td>
      <td>4.641355</td>
      <td>5.260213</td>
      <td>15.585623</td>
      <td>19.582423</td>
      <td>17.672967</td>
      <td>31.033295</td>
      <td>6.290427</td>
      <td>5.669278</td>
      <td>18.184770</td>
      <td>18.229738</td>
      <td>15.976012</td>
      <td>10.779278</td>
      <td>8.577655</td>
      <td>23.963389</td>
      <td>9.540205</td>
      <td>6.537321</td>
      <td>23.31500</td>
      <td>7.637780</td>
      <td>6.788014</td>
      <td>4.827062</td>
      <td>4.819912</td>
      <td>15.397786</td>
      <td>13.269314</td>
      <td>9.254037</td>
      <td>23.282221</td>
      <td>21.105346</td>
      <td>17.036512</td>
      <td>6.922584</td>
      <td>4.976585</td>
      <td>19.429776</td>
      <td>11.319723</td>
      <td>7.534891</td>
      <td>23.904609</td>
      <td>17.142341</td>
      <td>14.653234</td>
      <td>15.040145</td>
      <td>10.660610</td>
      <td>29.623427</td>
      <td>11.427878</td>
      <td>9.276489</td>
      <td>20.959921</td>
      <td>5.264065</td>
      <td>4.792533</td>
      <td>16.949673</td>
      <td>5.760529</td>
      <td>5.054803</td>
      <td>7.428323</td>
      <td>7.428323</td>
      <td>23.456341</td>
      <td>23.456437</td>
      <td>7.174463</td>
      <td>7.174422</td>
      <td>14.960869</td>
      <td>14.961218</td>
      <td>11.281884</td>
      <td>11.281949</td>
      <td>9.835163</td>
      <td>9.835375</td>
      <td>7.996013</td>
      <td>7.995969</td>
      <td>14.047174</td>
      <td>14.047164</td>
      <td>24.460549</td>
      <td>24.460557</td>
      <td>21.157010</td>
      <td>25.257209</td>
      <td>25.257641</td>
      <td>6.598414</td>
      <td>11.132447</td>
      <td>11.132397</td>
      <td>4.094296</td>
      <td>9.652742</td>
      <td>9.653392</td>
      <td>8.458099</td>
      <td>8.603233</td>
      <td>8.603288</td>
      <td>20.430749</td>
      <td>15.956651</td>
      <td>15.956724</td>
      <td>19.103576</td>
      <td>16.980181</td>
      <td>16.980867</td>
      <td>8.044106</td>
      <td>19.380718</td>
      <td>19.381638</td>
      <td>1.483753</td>
      <td>19.104395</td>
      <td>19.104863</td>
      <td>13.378952</td>
      <td>13.379217</td>
      <td>17.629637</td>
      <td>17.629495</td>
      <td>15.343400</td>
      <td>15.343372</td>
      <td>15.864878</td>
      <td>15.864857</td>
      <td>18.634941</td>
      <td>18.635020</td>
      <td>17.037778</td>
      <td>17.038808</td>
      <td>6.245351</td>
      <td>7.488056</td>
      <td>5.975242</td>
      <td>13.271345</td>
      <td>14.701906</td>
      <td>12.460579</td>
      <td>3.146010</td>
      <td>3.146010</td>
      <td>17.455115</td>
      <td>17.455087</td>
      <td>22.536620</td>
      <td>22.536815</td>
      <td>21.107300</td>
      <td>21.107158</td>
      <td>22.536620</td>
      <td>22.536815</td>
      <td>25.909950</td>
      <td>25.909917</td>
      <td>21.107300</td>
      <td>21.107158</td>
      <td>22.530361</td>
      <td>22.530830</td>
      <td>2.027785</td>
      <td>2.027785</td>
      <td>17.474454</td>
      <td>17.474705</td>
      <td>22.797204</td>
      <td>22.797653</td>
      <td>20.52406</td>
      <td>20.524311</td>
      <td>4.945440</td>
      <td>4.945357</td>
      <td>5.736344</td>
      <td>7.095159</td>
      <td>0.876654</td>
      <td>16.206147</td>
      <td>5.791786</td>
      <td>4.959632</td>
      <td>4.950548</td>
      <td>5.277803</td>
      <td>17.437677</td>
      <td>5.123436</td>
      <td>18.903176</td>
      <td>4.817765</td>
      <td>20.952193</td>
      <td>16.713487</td>
      <td>29.817069</td>
      <td>5.593498</td>
      <td>11.313522</td>
      <td>6.578257</td>
      <td>6.860240</td>
      <td>6.309146</td>
      <td>5.687259</td>
      <td>7.988587</td>
      <td>7.285930</td>
      <td>15.769333</td>
      <td>5.203475</td>
      <td>5.687259</td>
      <td>6.885604</td>
      <td>10.522413</td>
      <td>5.465570</td>
      <td>4.925731</td>
      <td>7.942400</td>
      <td>6.855356</td>
      <td>6.578921</td>
      <td>6.187010</td>
      <td>6.914358</td>
      <td>4.973908</td>
      <td>8.999851</td>
      <td>8.755874</td>
      <td>4.219809</td>
      <td>12.803161</td>
      <td>13.650127</td>
      <td>12.194086</td>
      <td>12.806052</td>
      <td>5.469261</td>
      <td>2.503673</td>
      <td>1.808505</td>
      <td>0.671939</td>
      <td>11.178890</td>
      <td>8.516725</td>
      <td>7.939195</td>
      <td>2.190445</td>
      <td>9.539173</td>
      <td>8.055670</td>
      <td>2.276802</td>
      <td>0.536970</td>
      <td>4.945440</td>
      <td>18.766423</td>
      <td>18.766268</td>
      <td>13.841678</td>
      <td>3.819502</td>
      <td>2.625376</td>
      <td>2.623370</td>
      <td>3.111511</td>
      <td>3.182797</td>
      <td>6.062893</td>
      <td>7.555389</td>
      <td>0.131810</td>
      <td>0.633765</td>
      <td>6.677880</td>
      <td>7.442519</td>
      <td>5.123610</td>
      <td>7.209096</td>
      <td>5.904955</td>
      <td>9.050930</td>
      <td>5.582953</td>
      <td>13.809467</td>
      <td>1.973959</td>
      <td>2.427928</td>
      <td>4.944949</td>
      <td>6.528390</td>
      <td>6.190912</td>
      <td>5.506008</td>
      <td>4.908791</td>
      <td>5.097198</td>
      <td>6.811107</td>
      <td>2.028886</td>
      <td>6.418775</td>
      <td>9.149282</td>
      <td>10.64539</td>
      <td>7.457425</td>
      <td>2.444666</td>
      <td>13.345441</td>
      <td>13.345426</td>
      <td>8.180315</td>
      <td>6.383707</td>
      <td>6.549956</td>
      <td>3.590712</td>
      <td>3.774815</td>
      <td>4.321336</td>
      <td>6.177498</td>
      <td>9.279767</td>
      <td>8.421056</td>
      <td>8.114962</td>
      <td>29.413486</td>
      <td>18.215930</td>
      <td>6.011436</td>
      <td>5.376556</td>
      <td>0.226353</td>
      <td>10.558507</td>
      <td>0.180891</td>
      <td>1.923402</td>
      <td>0.525165</td>
      <td>1.477265</td>
      <td>1.085943</td>
      <td>1.719139</td>
      <td>1.906352</td>
      <td>1.742647</td>
      <td>0.597486</td>
      <td>13.244689</td>
      <td>13.058773</td>
      <td>8.301889</td>
      <td>8.202484</td>
      <td>8.241330</td>
      <td>6.801513</td>
      <td>3.756282</td>
      <td>1.766219</td>
      <td>11.272676</td>
      <td>11.272705</td>
      <td>1.707207</td>
      <td>3.834211</td>
      <td>7.388491</td>
      <td>11.889959</td>
      <td>10.011499</td>
      <td>9.417423</td>
      <td>12.323096</td>
      <td>1.727763</td>
      <td>5.513867</td>
      <td>9.164541</td>
      <td>9.138935</td>
      <td>17.097243</td>
      <td>9.550965</td>
      <td>5.642793</td>
      <td>5.496076</td>
      <td>3.655780</td>
      <td>9.679465</td>
      <td>9.213841</td>
      <td>6.940785</td>
      <td>5.933818</td>
      <td>5.001172</td>
      <td>3.277591</td>
      <td>2.703560</td>
      <td>7.119371</td>
      <td>3.327821</td>
      <td>6.741572</td>
      <td>12.587445</td>
      <td>16.330541</td>
      <td>14.201784</td>
      <td>14.033410</td>
      <td>9.205299</td>
      <td>11.290930</td>
      <td>9.461901</td>
      <td>9.532139</td>
      <td>8.681435</td>
      <td>8.620332</td>
      <td>16.240102</td>
      <td>12.614203</td>
      <td>6.896935</td>
      <td>10.588347</td>
      <td>10.611434</td>
      <td>6.756172</td>
      <td>4.050149</td>
      <td>6.951597</td>
      <td>6.978646</td>
      <td>6.872734</td>
      <td>6.657885</td>
      <td>13.134418</td>
      <td>14.478985</td>
      <td>6.596903</td>
      <td>5.462868</td>
      <td>7.920630</td>
      <td>5.901265</td>
      <td>5.145924</td>
      <td>3.149895</td>
      <td>5.254462</td>
      <td>4.125102</td>
      <td>4.167491</td>
      <td>4.353795</td>
      <td>9.823986</td>
      <td>9.777658</td>
      <td>12.093596</td>
      <td>6.118449</td>
      <td>2.300566</td>
      <td>7.401392</td>
      <td>7.589113</td>
      <td>7.099991</td>
      <td>7.821100</td>
      <td>7.739582</td>
      <td>3.312890</td>
      <td>9.197541</td>
      <td>8.706646</td>
      <td>6.07505</td>
      <td>3.984000</td>
      <td>10.752041</td>
      <td>9.490468</td>
      <td>6.278337</td>
      <td>13.690781</td>
      <td>6.067874</td>
      <td>3.961163</td>
      <td>10.822650</td>
      <td>5.360838</td>
      <td>3.803041</td>
      <td>11.297106</td>
      <td>5.072608</td>
      <td>3.124242</td>
      <td>11.256912</td>
      <td>5.865518</td>
      <td>3.607789</td>
      <td>11.623280</td>
      <td>3.257911</td>
      <td>2.702740</td>
      <td>8.431143</td>
      <td>2.871202</td>
      <td>2.813559</td>
      <td>8.046584</td>
      <td>10.365689</td>
      <td>9.504281</td>
      <td>17.396398</td>
      <td>3.797550</td>
      <td>2.941695</td>
      <td>9.806050</td>
      <td>10.934203</td>
      <td>8.386297</td>
      <td>6.958456</td>
      <td>4.891767</td>
      <td>13.113113</td>
      <td>6.733360</td>
      <td>4.098973</td>
      <td>12.410078</td>
      <td>4.306642</td>
      <td>3.732562</td>
      <td>3.330578</td>
      <td>3.014223</td>
      <td>8.239847</td>
      <td>8.849890</td>
      <td>5.359822</td>
      <td>13.402663</td>
      <td>12.847655</td>
      <td>9.588986</td>
      <td>4.923784</td>
      <td>3.057945</td>
      <td>10.612499</td>
      <td>7.452981</td>
      <td>4.391234</td>
      <td>13.519279</td>
      <td>10.079406</td>
      <td>8.195666</td>
      <td>9.973697</td>
      <td>5.930883</td>
      <td>16.226716</td>
      <td>7.451723</td>
      <td>5.493556</td>
      <td>12.122575</td>
      <td>3.319393</td>
      <td>2.953937</td>
      <td>9.679165</td>
      <td>3.024274</td>
      <td>2.521551</td>
      <td>4.340173</td>
      <td>4.340173</td>
      <td>13.791787</td>
      <td>13.791779</td>
      <td>3.823784</td>
      <td>3.823829</td>
      <td>8.070076</td>
      <td>8.070027</td>
      <td>5.705982</td>
      <td>5.705932</td>
      <td>4.960828</td>
      <td>4.960908</td>
      <td>4.450564</td>
      <td>4.450452</td>
      <td>7.529733</td>
      <td>7.529852</td>
      <td>13.552129</td>
      <td>13.552037</td>
      <td>11.040093</td>
      <td>14.859899</td>
      <td>14.860147</td>
      <td>3.003324</td>
      <td>5.742631</td>
      <td>5.742582</td>
      <td>1.798983</td>
      <td>4.736942</td>
      <td>4.737168</td>
      <td>4.471602</td>
      <td>4.724086</td>
      <td>4.724336</td>
      <td>10.624276</td>
      <td>8.502643</td>
      <td>8.502677</td>
      <td>11.204520</td>
      <td>9.152611</td>
      <td>9.152881</td>
      <td>4.960498</td>
      <td>10.574986</td>
      <td>10.575163</td>
      <td>0.949321</td>
      <td>10.127319</td>
      <td>10.127216</td>
      <td>7.079618</td>
      <td>7.079769</td>
      <td>8.943524</td>
      <td>8.943616</td>
      <td>8.567107</td>
      <td>8.567035</td>
      <td>9.160561</td>
      <td>9.160653</td>
      <td>10.028405</td>
      <td>10.028382</td>
      <td>9.951423</td>
      <td>9.951783</td>
      <td>4.185476</td>
      <td>4.855060</td>
      <td>3.893691</td>
      <td>10.487968</td>
      <td>11.431440</td>
      <td>9.789335</td>
      <td>2.192803</td>
      <td>2.192803</td>
      <td>9.141740</td>
      <td>9.141436</td>
      <td>12.232674</td>
      <td>12.232263</td>
      <td>11.013302</td>
      <td>11.013044</td>
      <td>12.232674</td>
      <td>12.232263</td>
      <td>14.299064</td>
      <td>14.299149</td>
      <td>11.013302</td>
      <td>11.013044</td>
      <td>11.949770</td>
      <td>11.950283</td>
      <td>1.368347</td>
      <td>1.368347</td>
      <td>8.839155</td>
      <td>8.839034</td>
      <td>12.206432</td>
      <td>12.206899</td>
      <td>10.359820</td>
      <td>10.359747</td>
      <td>3.057932</td>
      <td>3.057854</td>
      <td>3.011410</td>
      <td>4.594459</td>
      <td>0.456644</td>
      <td>8.732685</td>
      <td>3.534018</td>
      <td>2.975342</td>
      <td>3.078884</td>
      <td>3.283881</td>
      <td>8.748859</td>
      <td>3.140519</td>
      <td>9.501172</td>
      <td>3.002385</td>
      <td>11.723967</td>
      <td>8.509538</td>
      <td>17.208706</td>
      <td>3.368226</td>
      <td>6.014637</td>
      <td>4.002487</td>
      <td>3.502611</td>
      <td>3.247602</td>
      <td>3.509123</td>
      <td>5.216842</td>
      <td>4.753529</td>
      <td>7.996146</td>
      <td>2.900901</td>
      <td>3.509123</td>
      <td>4.285282</td>
      <td>5.986304</td>
      <td>3.231184</td>
      <td>3.019271</td>
      <td>3.786698</td>
      <td>3.613100</td>
      <td>3.495975</td>
      <td>3.116174</td>
      <td>3.749252</td>
      <td>3.076238</td>
      <td>4.425448</td>
      <td>4.278451</td>
      <td>2.474604</td>
      <td>9.152914</td>
      <td>10.548821</td>
      <td>8.923250</td>
      <td>9.907477</td>
      <td>3.661483</td>
      <td>1.562651</td>
      <td>1.135113</td>
      <td>0.354934</td>
      <td>8.565090</td>
      <td>5.894491</td>
      <td>5.985505</td>
      <td>1.370659</td>
      <td>6.693669</td>
      <td>5.559262</td>
      <td>1.639218</td>
      <td>0.331416</td>
      <td>3.057932</td>
      <td>15.334439</td>
      <td>15.334327</td>
      <td>9.524122</td>
      <td>2.852621</td>
      <td>1.816758</td>
      <td>1.565267</td>
      <td>2.248238</td>
      <td>2.165020</td>
      <td>4.025454</td>
      <td>5.449321</td>
      <td>0.060081</td>
      <td>0.318923</td>
      <td>4.593508</td>
      <td>5.744466</td>
      <td>3.598359</td>
      <td>4.499431</td>
      <td>3.825537</td>
      <td>6.921216</td>
      <td>4.269065</td>
      <td>10.578778</td>
      <td>1.257280</td>
      <td>1.702463</td>
      <td>3.288163</td>
      <td>4.567893</td>
      <td>3.884831</td>
      <td>3.400909</td>
      <td>3.359012</td>
      <td>2.972929</td>
      <td>4.787089</td>
      <td>1.287785</td>
      <td>4.168243</td>
      <td>6.528640</td>
      <td>7.430319</td>
      <td>5.076504</td>
      <td>1.391024</td>
      <td>10.050475</td>
      <td>10.050397</td>
      <td>5.705167</td>
      <td>3.499148</td>
      <td>3.866724</td>
      <td>1.936930</td>
      <td>2.354632</td>
      <td>2.663833</td>
      <td>4.619557</td>
      <td>6.043083</td>
      <td>5.184571</td>
      <td>5.833056</td>
      <td>24.371281</td>
      <td>14.784190</td>
      <td>4.119089</td>
      <td>3.328370</td>
      <td>0.127675</td>
      <td>8.658425</td>
      <td>0.08754</td>
      <td>1.207208</td>
      <td>0.307842</td>
      <td>0.769183</td>
      <td>0.558564</td>
      <td>0.969044</td>
      <td>0.973122</td>
      <td>0.858829</td>
      <td>0.317747</td>
      <td>9.982182</td>
      <td>9.380386</td>
      <td>5.134618</td>
      <td>5.118849</td>
      <td>5.786109</td>
      <td>4.902503</td>
      <td>2.392673</td>
      <td>0.860147</td>
      <td>8.432078</td>
      <td>8.432217</td>
      <td>0.784742</td>
      <td>2.435621</td>
      <td>5.479692</td>
      <td>8.241303</td>
      <td>6.280986</td>
      <td>6.512143</td>
      <td>9.631279</td>
      <td>0.865687</td>
      <td>3.59480</td>
      <td>6.481739</td>
      <td>5.710474</td>
      <td>13.419604</td>
      <td>5.374796</td>
      <td>2.635301</td>
      <td>2.450077</td>
      <td>1.665471</td>
      <td>5.442526</td>
      <td>5.127607</td>
      <td>3.44194</td>
      <td>2.721662</td>
      <td>2.724137</td>
      <td>1.470005</td>
      <td>1.260631</td>
      <td>3.786705</td>
      <td>1.607610</td>
      <td>3.583022</td>
      <td>7.820402</td>
      <td>9.497692</td>
      <td>7.541677</td>
      <td>9.162405</td>
      <td>6.274894</td>
      <td>5.537350</td>
      <td>4.548995</td>
      <td>4.411579</td>
      <td>3.728855</td>
      <td>4.252896</td>
      <td>8.470195</td>
    </tr>
    <tr>
      <th>min</th>
      <td>48001.000000</td>
      <td>48001.000000</td>
      <td>0.900000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>47.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>37.100000</td>
      <td>5.500000</td>
      <td>30.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>67.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>68.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>47.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.900000</td>
      <td>0.000000</td>
      <td>11.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.100000</td>
      <td>0.000000</td>
      <td>2.900000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.20000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>65.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.100000</td>
      <td>4.400000</td>
      <td>3.900000</td>
      <td>3.340000</td>
      <td>4.380000</td>
      <td>2.100000</td>
      <td>3.775000</td>
      <td>4.750000</td>
      <td>4.225000</td>
      <td>4.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.100000</td>
      <td>3.820000</td>
      <td>3.100000</td>
      <td>3.800000</td>
      <td>2.920000</td>
      <td>0.000000</td>
      <td>2.960000</td>
      <td>1.940000</td>
      <td>1.140000</td>
      <td>0.880000</td>
      <td>3.920000</td>
      <td>0.900000</td>
      <td>3.400000</td>
      <td>4.325000</td>
      <td>0.200000</td>
      <td>5.020000</td>
      <td>6.075000</td>
      <td>11.660000</td>
      <td>7.000000</td>
      <td>10.740000</td>
      <td>0.000000</td>
      <td>11.140000</td>
      <td>14.560000</td>
      <td>2.30000</td>
      <td>1.220000</td>
      <td>0.000000</td>
      <td>3.980000</td>
      <td>0.260000</td>
      <td>0.000000</td>
      <td>2.275000</td>
      <td>1.200000</td>
      <td>0.000000</td>
      <td>2.275000</td>
      <td>1.225000</td>
      <td>0.000000</td>
      <td>2.275000</td>
      <td>1.175000</td>
      <td>0.000000</td>
      <td>1.425000</td>
      <td>0.520000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.900000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.025000</td>
      <td>0.260000</td>
      <td>0.000000</td>
      <td>1.780000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.860000</td>
      <td>0.540000</td>
      <td>0.000000</td>
      <td>1.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.640000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.400000</td>
      <td>0.180000</td>
      <td>0.000000</td>
      <td>1.580000</td>
      <td>0.560000</td>
      <td>0.000000</td>
      <td>0.140000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>64.900000</td>
      <td>2.520000</td>
      <td>27.560000</td>
      <td>0.000000</td>
      <td>74.860000</td>
      <td>0.420000</td>
      <td>42.066667</td>
      <td>0.000000</td>
      <td>60.900000</td>
      <td>0.400000</td>
      <td>68.125000</td>
      <td>0.000000</td>
      <td>74.100000</td>
      <td>0.000000</td>
      <td>52.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.440000</td>
      <td>24.260000</td>
      <td>0.000000</td>
      <td>75.000000</td>
      <td>59.620000</td>
      <td>0.350000</td>
      <td>88.540000</td>
      <td>70.100000</td>
      <td>0.000000</td>
      <td>75.220000</td>
      <td>69.620000</td>
      <td>0.000000</td>
      <td>14.800000</td>
      <td>45.520000</td>
      <td>0.000000</td>
      <td>5.060000</td>
      <td>46.960000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.700000</td>
      <td>0.000000</td>
      <td>0.580000</td>
      <td>35.825000</td>
      <td>0.000000</td>
      <td>56.140000</td>
      <td>0.360000</td>
      <td>40.600000</td>
      <td>0.000000</td>
      <td>36.666667</td>
      <td>0.000000</td>
      <td>33.333333</td>
      <td>0.000000</td>
      <td>29.833333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>69.380000</td>
      <td>64.060000</td>
      <td>69.400000</td>
      <td>5.260000</td>
      <td>4.560000</td>
      <td>6.000000</td>
      <td>83.760000</td>
      <td>0.080000</td>
      <td>43.860000</td>
      <td>2.600000</td>
      <td>13.333333</td>
      <td>0.000000</td>
      <td>32.800000</td>
      <td>0.000000</td>
      <td>13.333333</td>
      <td>0.000000</td>
      <td>13.333333</td>
      <td>0.000000</td>
      <td>32.800000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>85.780000</td>
      <td>0.000000</td>
      <td>41.350000</td>
      <td>0.000000</td>
      <td>18.625000</td>
      <td>0.000000</td>
      <td>37.375000</td>
      <td>0.000000</td>
      <td>4.840000</td>
      <td>74.040000</td>
      <td>0.340000</td>
      <td>6.500000</td>
      <td>0.000000</td>
      <td>4.660000</td>
      <td>5.300000</td>
      <td>3.475000</td>
      <td>4.840000</td>
      <td>4.320000</td>
      <td>0.000000</td>
      <td>4.340000</td>
      <td>0.000000</td>
      <td>4.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.100000</td>
      <td>0.380000</td>
      <td>5.060000</td>
      <td>4.320000</td>
      <td>0.000000</td>
      <td>5.360000</td>
      <td>5.620000</td>
      <td>4.880000</td>
      <td>11.420000</td>
      <td>2.500000</td>
      <td>5.360000</td>
      <td>2.760000</td>
      <td>9.260000</td>
      <td>2.900000</td>
      <td>4.580000</td>
      <td>8.420000</td>
      <td>5.860000</td>
      <td>2.140000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.820000</td>
      <td>10.140000</td>
      <td>5.620000</td>
      <td>2.500000</td>
      <td>33.080000</td>
      <td>22.060000</td>
      <td>29.600000</td>
      <td>17.940000</td>
      <td>4.720000</td>
      <td>1.140000</td>
      <td>0.280000</td>
      <td>0.000000</td>
      <td>18.120000</td>
      <td>4.000000</td>
      <td>10.880000</td>
      <td>1.125000</td>
      <td>4.040000</td>
      <td>1.840000</td>
      <td>0.625000</td>
      <td>0.000000</td>
      <td>4.840000</td>
      <td>30.980000</td>
      <td>3.640000</td>
      <td>25.660000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.200000</td>
      <td>1.060000</td>
      <td>1.340000</td>
      <td>3.260000</td>
      <td>3.240000</td>
      <td>3.260000</td>
      <td>1.020000</td>
      <td>1.220000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.320000</td>
      <td>5.040000</td>
      <td>10.125000</td>
      <td>8.900000</td>
      <td>5.220000</td>
      <td>2.820000</td>
      <td>2.220000</td>
      <td>0.000000</td>
      <td>1.780000</td>
      <td>10.040000</td>
      <td>21.200000</td>
      <td>3.920000</td>
      <td>0.820000</td>
      <td>26.440000</td>
      <td>5.700000</td>
      <td>5.020000</td>
      <td>23.920000</td>
      <td>9.560000</td>
      <td>4.440000</td>
      <td>2.580000</td>
      <td>1.125000</td>
      <td>1.080000</td>
      <td>18.520000</td>
      <td>14.160000</td>
      <td>4.600000</td>
      <td>1.500000</td>
      <td>0.440000</td>
      <td>1.160000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.840000</td>
      <td>91.320000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.960000</td>
      <td>4.620000</td>
      <td>5.640000</td>
      <td>1.960000</td>
      <td>0.760000</td>
      <td>0.160000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>36.420000</td>
      <td>22.200000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>1.080000</td>
      <td>3.700000</td>
      <td>12.300000</td>
      <td>3.240000</td>
      <td>1.040000</td>
      <td>0.000000</td>
      <td>0.20000</td>
      <td>0.900000</td>
      <td>2.620000</td>
      <td>30.780000</td>
      <td>22.260000</td>
      <td>6.860000</td>
      <td>4.880000</td>
      <td>2.320000</td>
      <td>12.260000</td>
      <td>17.000000</td>
      <td>11.74000</td>
      <td>6.580000</td>
      <td>1.680000</td>
      <td>1.380000</td>
      <td>0.660000</td>
      <td>4.280000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>5.100000</td>
      <td>1.825000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.620000</td>
      <td>0.000000</td>
      <td>0.280000</td>
      <td>1.200000</td>
      <td>0.000000</td>
      <td>8.080000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48421.250000</td>
      <td>48421.250000</td>
      <td>18.375000</td>
      <td>113.000000</td>
      <td>19.000000</td>
      <td>12.800000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.900000</td>
      <td>10.825000</td>
      <td>10.225000</td>
      <td>8.700000</td>
      <td>4.300000</td>
      <td>7.900000</td>
      <td>9.600000</td>
      <td>8.700000</td>
      <td>8.200000</td>
      <td>1.975000</td>
      <td>4.800000</td>
      <td>8.000000</td>
      <td>7.400000</td>
      <td>14.900000</td>
      <td>8.600000</td>
      <td>6.200000</td>
      <td>1.600000</td>
      <td>5.700000</td>
      <td>3.975000</td>
      <td>2.800000</td>
      <td>4.400000</td>
      <td>19.125000</td>
      <td>15.825000</td>
      <td>18.700000</td>
      <td>8.200000</td>
      <td>1.100000</td>
      <td>9.800000</td>
      <td>12.775000</td>
      <td>19.900000</td>
      <td>16.700000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>28.175000</td>
      <td>28.000000</td>
      <td>5.800000</td>
      <td>2.600000</td>
      <td>19.400000</td>
      <td>9.700000</td>
      <td>3.100000</td>
      <td>27.300000</td>
      <td>5.800000</td>
      <td>2.60000</td>
      <td>19.300000</td>
      <td>5.300000</td>
      <td>2.300000</td>
      <td>17.600000</td>
      <td>5.100000</td>
      <td>2.300000</td>
      <td>17.400000</td>
      <td>3.800000</td>
      <td>1.400000</td>
      <td>13.300000</td>
      <td>0.550000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.350000</td>
      <td>0.000000</td>
      <td>20.450000</td>
      <td>2.400000</td>
      <td>1.100000</td>
      <td>0.000000</td>
      <td>7.400000</td>
      <td>0.000000</td>
      <td>6.200000</td>
      <td>2.600000</td>
      <td>14.300000</td>
      <td>5.400000</td>
      <td>1.525000</td>
      <td>16.10000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.300000</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>7.600000</td>
      <td>1.500000</td>
      <td>22.200000</td>
      <td>7.800000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>1.500000</td>
      <td>10.300000</td>
      <td>5.100000</td>
      <td>1.300000</td>
      <td>19.575000</td>
      <td>3.925000</td>
      <td>0.000000</td>
      <td>9.300000</td>
      <td>3.400000</td>
      <td>32.500000</td>
      <td>8.700000</td>
      <td>3.200000</td>
      <td>16.950000</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>85.200000</td>
      <td>7.100000</td>
      <td>57.400000</td>
      <td>7.800000</td>
      <td>87.600000</td>
      <td>3.925000</td>
      <td>84.525000</td>
      <td>0.000000</td>
      <td>84.850000</td>
      <td>3.100000</td>
      <td>87.000000</td>
      <td>2.600000</td>
      <td>90.050000</td>
      <td>1.450000</td>
      <td>79.500000</td>
      <td>7.300000</td>
      <td>63.200000</td>
      <td>0.000000</td>
      <td>33.300000</td>
      <td>56.475000</td>
      <td>4.800000</td>
      <td>94.500000</td>
      <td>83.900000</td>
      <td>4.100000</td>
      <td>97.350000</td>
      <td>87.100000</td>
      <td>2.900000</td>
      <td>95.000000</td>
      <td>90.300000</td>
      <td>0.425000</td>
      <td>59.450000</td>
      <td>84.875000</td>
      <td>0.000000</td>
      <td>22.800000</td>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>74.900000</td>
      <td>0.000000</td>
      <td>1.500000</td>
      <td>68.300000</td>
      <td>8.000000</td>
      <td>77.700000</td>
      <td>9.100000</td>
      <td>77.400000</td>
      <td>5.500000</td>
      <td>76.200000</td>
      <td>7.500000</td>
      <td>81.625000</td>
      <td>2.525000</td>
      <td>81.300000</td>
      <td>0.000000</td>
      <td>80.400000</td>
      <td>0.000000</td>
      <td>86.300000</td>
      <td>84.600000</td>
      <td>87.700000</td>
      <td>11.800000</td>
      <td>10.500000</td>
      <td>12.900000</td>
      <td>98.000000</td>
      <td>0.400000</td>
      <td>61.675000</td>
      <td>16.500000</td>
      <td>60.600000</td>
      <td>10.900000</td>
      <td>65.475000</td>
      <td>4.975000</td>
      <td>60.600000</td>
      <td>10.900000</td>
      <td>54.900000</td>
      <td>6.150000</td>
      <td>65.475000</td>
      <td>4.975000</td>
      <td>66.700000</td>
      <td>0.000000</td>
      <td>98.800000</td>
      <td>0.200000</td>
      <td>67.925000</td>
      <td>10.000000</td>
      <td>67.950000</td>
      <td>0.000000</td>
      <td>65.32500</td>
      <td>4.250000</td>
      <td>8.300000</td>
      <td>85.900000</td>
      <td>1.800000</td>
      <td>12.475000</td>
      <td>0.000000</td>
      <td>15.775000</td>
      <td>9.000000</td>
      <td>7.200000</td>
      <td>8.300000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>7.800000</td>
      <td>4.300000</td>
      <td>8.000000</td>
      <td>2.975000</td>
      <td>0.000000</td>
      <td>2.050000</td>
      <td>9.100000</td>
      <td>10.500000</td>
      <td>10.600000</td>
      <td>8.600000</td>
      <td>2.700000</td>
      <td>9.800000</td>
      <td>12.400000</td>
      <td>10.100000</td>
      <td>28.825000</td>
      <td>4.700000</td>
      <td>9.800000</td>
      <td>6.600000</td>
      <td>16.400000</td>
      <td>5.800000</td>
      <td>8.400000</td>
      <td>14.100000</td>
      <td>10.300000</td>
      <td>5.800000</td>
      <td>2.800000</td>
      <td>1.800000</td>
      <td>8.400000</td>
      <td>16.500000</td>
      <td>10.875000</td>
      <td>4.800000</td>
      <td>61.700000</td>
      <td>39.800000</td>
      <td>49.900000</td>
      <td>35.300000</td>
      <td>10.600000</td>
      <td>3.000000</td>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>28.700000</td>
      <td>11.400000</td>
      <td>15.700000</td>
      <td>2.400000</td>
      <td>12.600000</td>
      <td>7.800000</td>
      <td>1.500000</td>
      <td>0.100000</td>
      <td>8.300000</td>
      <td>68.200000</td>
      <td>7.600000</td>
      <td>73.100000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.500000</td>
      <td>9.200000</td>
      <td>7.100000</td>
      <td>11.900000</td>
      <td>7.200000</td>
      <td>6.700000</td>
      <td>3.400000</td>
      <td>8.700000</td>
      <td>0.000000</td>
      <td>0.400000</td>
      <td>3.300000</td>
      <td>10.300000</td>
      <td>17.100000</td>
      <td>16.900000</td>
      <td>10.900000</td>
      <td>6.500000</td>
      <td>6.700000</td>
      <td>0.100000</td>
      <td>4.000000</td>
      <td>20.400000</td>
      <td>39.40000</td>
      <td>10.600000</td>
      <td>2.000000</td>
      <td>74.300000</td>
      <td>12.300000</td>
      <td>13.500000</td>
      <td>33.100000</td>
      <td>17.300000</td>
      <td>8.300000</td>
      <td>5.400000</td>
      <td>3.600000</td>
      <td>2.800000</td>
      <td>25.900000</td>
      <td>37.200000</td>
      <td>16.200000</td>
      <td>33.000000</td>
      <td>2.100000</td>
      <td>4.300000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.700000</td>
      <td>97.900000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>8.300000</td>
      <td>17.800000</td>
      <td>14.900000</td>
      <td>10.000000</td>
      <td>6.800000</td>
      <td>1.800000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>53.300000</td>
      <td>32.900000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>4.300000</td>
      <td>15.700000</td>
      <td>29.025000</td>
      <td>10.500000</td>
      <td>4.100000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>8.550000</td>
      <td>16.400000</td>
      <td>45.300000</td>
      <td>34.200000</td>
      <td>13.200000</td>
      <td>9.025000</td>
      <td>5.600000</td>
      <td>20.100000</td>
      <td>31.250000</td>
      <td>17.800000</td>
      <td>10.500000</td>
      <td>5.650000</td>
      <td>3.300000</td>
      <td>1.900000</td>
      <td>8.550000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.575000</td>
      <td>23.875000</td>
      <td>16.900000</td>
      <td>6.400000</td>
      <td>0.000000</td>
      <td>7.375000</td>
      <td>6.775000</td>
      <td>6.600000</td>
      <td>6.575000</td>
      <td>3.900000</td>
      <td>32.600000</td>
      <td>21.705000</td>
      <td>11.165000</td>
      <td>15.120000</td>
      <td>14.760000</td>
      <td>10.725000</td>
      <td>5.720000</td>
      <td>10.040000</td>
      <td>11.980000</td>
      <td>11.020000</td>
      <td>10.325000</td>
      <td>15.385000</td>
      <td>14.450000</td>
      <td>9.985000</td>
      <td>8.925000</td>
      <td>18.660000</td>
      <td>10.360000</td>
      <td>7.565000</td>
      <td>3.060000</td>
      <td>7.520000</td>
      <td>5.380000</td>
      <td>4.180000</td>
      <td>6.225000</td>
      <td>24.460000</td>
      <td>22.805000</td>
      <td>25.365000</td>
      <td>10.000000</td>
      <td>1.760000</td>
      <td>12.925000</td>
      <td>15.020000</td>
      <td>22.860000</td>
      <td>19.665000</td>
      <td>25.110000</td>
      <td>2.360000</td>
      <td>32.365000</td>
      <td>31.635000</td>
      <td>7.50500</td>
      <td>3.900000</td>
      <td>24.030000</td>
      <td>13.125000</td>
      <td>5.725000</td>
      <td>33.805000</td>
      <td>7.400000</td>
      <td>3.865000</td>
      <td>24.045000</td>
      <td>6.900000</td>
      <td>3.700000</td>
      <td>22.310000</td>
      <td>6.745000</td>
      <td>3.420000</td>
      <td>21.630000</td>
      <td>5.680000</td>
      <td>2.520000</td>
      <td>19.925000</td>
      <td>1.860000</td>
      <td>0.800000</td>
      <td>4.905000</td>
      <td>2.580000</td>
      <td>1.820000</td>
      <td>3.435000</td>
      <td>19.140000</td>
      <td>9.320000</td>
      <td>33.925000</td>
      <td>4.060000</td>
      <td>2.520000</td>
      <td>9.725000</td>
      <td>13.670000</td>
      <td>8.400000</td>
      <td>8.600000</td>
      <td>4.585000</td>
      <td>24.130000</td>
      <td>7.385000</td>
      <td>3.220000</td>
      <td>24.325000</td>
      <td>1.840000</td>
      <td>1.020000</td>
      <td>3.485000</td>
      <td>2.480000</td>
      <td>7.290000</td>
      <td>11.135000</td>
      <td>3.945000</td>
      <td>29.605000</td>
      <td>15.630000</td>
      <td>7.660000</td>
      <td>5.645000</td>
      <td>2.640000</td>
      <td>17.320000</td>
      <td>7.965000</td>
      <td>3.125000</td>
      <td>26.606250</td>
      <td>9.880000</td>
      <td>6.805000</td>
      <td>12.490000</td>
      <td>6.220000</td>
      <td>43.063750</td>
      <td>11.840000</td>
      <td>5.800000</td>
      <td>24.525000</td>
      <td>2.220000</td>
      <td>1.580000</td>
      <td>4.825000</td>
      <td>0.340000</td>
      <td>0.000000</td>
      <td>86.040000</td>
      <td>8.890000</td>
      <td>64.065000</td>
      <td>16.242500</td>
      <td>88.665000</td>
      <td>6.325000</td>
      <td>84.465000</td>
      <td>4.527500</td>
      <td>86.660000</td>
      <td>6.505000</td>
      <td>87.960000</td>
      <td>5.980000</td>
      <td>90.565000</td>
      <td>4.385000</td>
      <td>80.565000</td>
      <td>10.325000</td>
      <td>67.300000</td>
      <td>16.230000</td>
      <td>37.580000</td>
      <td>62.590000</td>
      <td>16.285000</td>
      <td>94.980000</td>
      <td>85.550000</td>
      <td>7.428750</td>
      <td>97.150000</td>
      <td>88.205000</td>
      <td>6.020000</td>
      <td>95.365000</td>
      <td>90.785000</td>
      <td>4.060000</td>
      <td>63.635000</td>
      <td>84.905000</td>
      <td>5.290000</td>
      <td>27.200000</td>
      <td>81.115000</td>
      <td>8.130000</td>
      <td>8.965000</td>
      <td>76.240000</td>
      <td>10.140000</td>
      <td>1.725000</td>
      <td>72.585000</td>
      <td>14.850000</td>
      <td>79.030000</td>
      <td>12.270000</td>
      <td>78.805000</td>
      <td>10.120000</td>
      <td>77.665000</td>
      <td>11.760000</td>
      <td>81.730000</td>
      <td>7.700000</td>
      <td>80.885000</td>
      <td>6.757500</td>
      <td>81.845000</td>
      <td>7.195000</td>
      <td>86.760000</td>
      <td>85.205000</td>
      <td>88.200000</td>
      <td>13.580000</td>
      <td>12.465000</td>
      <td>14.150000</td>
      <td>97.585000</td>
      <td>0.680000</td>
      <td>65.865000</td>
      <td>21.880000</td>
      <td>64.490000</td>
      <td>19.815000</td>
      <td>70.485000</td>
      <td>14.625000</td>
      <td>64.490000</td>
      <td>19.815000</td>
      <td>62.250000</td>
      <td>19.015000</td>
      <td>70.485000</td>
      <td>14.625000</td>
      <td>72.005000</td>
      <td>11.562500</td>
      <td>98.780000</td>
      <td>0.400000</td>
      <td>71.575000</td>
      <td>15.980000</td>
      <td>71.970000</td>
      <td>13.927083</td>
      <td>70.663750</td>
      <td>15.330000</td>
      <td>9.540000</td>
      <td>86.440000</td>
      <td>2.960000</td>
      <td>14.065000</td>
      <td>0.020000</td>
      <td>19.560000</td>
      <td>10.440000</td>
      <td>8.180000</td>
      <td>9.560000</td>
      <td>9.280000</td>
      <td>7.045000</td>
      <td>9.085000</td>
      <td>10.980000</td>
      <td>9.130000</td>
      <td>11.185000</td>
      <td>4.000000</td>
      <td>16.236250</td>
      <td>10.470000</td>
      <td>13.555000</td>
      <td>11.960000</td>
      <td>10.465000</td>
      <td>4.320000</td>
      <td>11.260000</td>
      <td>14.040000</td>
      <td>11.830000</td>
      <td>32.880000</td>
      <td>5.850000</td>
      <td>11.260000</td>
      <td>8.245000</td>
      <td>19.260000</td>
      <td>7.245000</td>
      <td>9.520000</td>
      <td>16.325000</td>
      <td>12.360000</td>
      <td>7.590000</td>
      <td>4.745000</td>
      <td>3.160000</td>
      <td>9.560000</td>
      <td>18.565000</td>
      <td>13.440000</td>
      <td>5.820000</td>
      <td>62.875000</td>
      <td>41.860000</td>
      <td>51.240000</td>
      <td>37.285000</td>
      <td>11.260000</td>
      <td>3.560000</td>
      <td>0.860000</td>
      <td>0.180000</td>
      <td>31.030000</td>
      <td>13.720000</td>
      <td>16.260000</td>
      <td>2.720000</td>
      <td>14.960000</td>
      <td>9.780000</td>
      <td>1.620000</td>
      <td>0.200000</td>
      <td>9.540000</td>
      <td>67.605000</td>
      <td>8.520000</td>
      <td>74.110000</td>
      <td>0.640000</td>
      <td>0.840000</td>
      <td>0.780000</td>
      <td>0.860000</td>
      <td>0.520000</td>
      <td>0.565000</td>
      <td>4.225000</td>
      <td>0.000000</td>
      <td>0.180000</td>
      <td>7.685000</td>
      <td>10.020000</td>
      <td>8.020000</td>
      <td>13.340000</td>
      <td>8.380000</td>
      <td>8.045000</td>
      <td>4.380000</td>
      <td>11.660000</td>
      <td>0.480000</td>
      <td>0.820000</td>
      <td>4.325000</td>
      <td>11.380000</td>
      <td>18.725000</td>
      <td>17.860000</td>
      <td>11.480000</td>
      <td>7.140000</td>
      <td>7.540000</td>
      <td>0.560000</td>
      <td>5.165000</td>
      <td>22.210000</td>
      <td>40.045000</td>
      <td>11.645000</td>
      <td>2.560000</td>
      <td>74.975000</td>
      <td>15.000000</td>
      <td>14.760000</td>
      <td>34.000000</td>
      <td>18.105000</td>
      <td>9.180000</td>
      <td>6.120000</td>
      <td>4.245000</td>
      <td>3.800000</td>
      <td>27.885000</td>
      <td>38.380000</td>
      <td>17.720000</td>
      <td>36.520000</td>
      <td>5.740000</td>
      <td>5.221250</td>
      <td>1.005000</td>
      <td>0.000000</td>
      <td>1.530000</td>
      <td>0.00000</td>
      <td>0.585000</td>
      <td>0.160000</td>
      <td>0.180000</td>
      <td>0.320000</td>
      <td>2.120000</td>
      <td>98.040000</td>
      <td>0.780000</td>
      <td>0.140000</td>
      <td>10.855000</td>
      <td>21.665000</td>
      <td>16.520000</td>
      <td>11.790000</td>
      <td>8.100000</td>
      <td>2.965000</td>
      <td>0.640000</td>
      <td>0.180000</td>
      <td>53.130000</td>
      <td>34.255000</td>
      <td>0.040000</td>
      <td>1.500000</td>
      <td>5.640000</td>
      <td>18.735000</td>
      <td>30.340000</td>
      <td>11.505000</td>
      <td>5.360000</td>
      <td>0.300000</td>
      <td>2.98000</td>
      <td>10.470000</td>
      <td>18.380000</td>
      <td>46.665000</td>
      <td>35.685000</td>
      <td>14.525000</td>
      <td>9.920000</td>
      <td>6.445000</td>
      <td>21.980000</td>
      <td>33.045000</td>
      <td>19.14000</td>
      <td>11.840000</td>
      <td>6.600000</td>
      <td>4.260000</td>
      <td>2.680000</td>
      <td>9.880000</td>
      <td>0.560000</td>
      <td>1.805000</td>
      <td>7.705000</td>
      <td>28.830000</td>
      <td>20.660000</td>
      <td>9.762500</td>
      <td>0.780000</td>
      <td>10.390000</td>
      <td>9.185000</td>
      <td>9.040000</td>
      <td>9.050000</td>
      <td>6.115000</td>
      <td>37.276250</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49003.000000</td>
      <td>49003.000000</td>
      <td>27.800000</td>
      <td>335.500000</td>
      <td>52.500000</td>
      <td>17.050000</td>
      <td>51.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.300000</td>
      <td>0.000000</td>
      <td>13.600000</td>
      <td>18.950000</td>
      <td>18.500000</td>
      <td>13.400000</td>
      <td>6.800000</td>
      <td>12.200000</td>
      <td>14.500000</td>
      <td>13.300000</td>
      <td>12.700000</td>
      <td>19.250000</td>
      <td>17.400000</td>
      <td>12.400000</td>
      <td>10.800000</td>
      <td>22.350000</td>
      <td>12.100000</td>
      <td>9.300000</td>
      <td>3.600000</td>
      <td>9.400000</td>
      <td>6.600000</td>
      <td>5.300000</td>
      <td>7.800000</td>
      <td>29.500000</td>
      <td>27.350000</td>
      <td>31.700000</td>
      <td>11.900000</td>
      <td>2.300000</td>
      <td>15.300000</td>
      <td>17.700000</td>
      <td>26.400000</td>
      <td>23.800000</td>
      <td>28.700000</td>
      <td>2.800000</td>
      <td>38.500000</td>
      <td>36.300000</td>
      <td>9.200000</td>
      <td>4.700000</td>
      <td>30.400000</td>
      <td>16.700000</td>
      <td>7.500000</td>
      <td>42.050000</td>
      <td>9.000000</td>
      <td>4.70000</td>
      <td>29.900000</td>
      <td>8.500000</td>
      <td>4.500000</td>
      <td>28.400000</td>
      <td>8.300000</td>
      <td>4.300000</td>
      <td>27.800000</td>
      <td>7.300000</td>
      <td>3.200000</td>
      <td>25.000000</td>
      <td>2.300000</td>
      <td>0.900000</td>
      <td>5.400000</td>
      <td>3.100000</td>
      <td>2.400000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>12.400000</td>
      <td>45.900000</td>
      <td>5.100000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>18.800000</td>
      <td>10.600000</td>
      <td>11.000000</td>
      <td>5.500000</td>
      <td>31.750000</td>
      <td>9.250000</td>
      <td>4.000000</td>
      <td>31.00000</td>
      <td>2.000000</td>
      <td>0.900000</td>
      <td>4.200000</td>
      <td>3.100000</td>
      <td>9.000000</td>
      <td>14.300000</td>
      <td>5.100000</td>
      <td>36.900000</td>
      <td>19.800000</td>
      <td>9.000000</td>
      <td>7.200000</td>
      <td>3.200000</td>
      <td>22.100000</td>
      <td>10.200000</td>
      <td>4.200000</td>
      <td>34.900000</td>
      <td>12.900000</td>
      <td>8.000000</td>
      <td>16.100000</td>
      <td>7.500000</td>
      <td>57.500000</td>
      <td>14.900000</td>
      <td>7.400000</td>
      <td>30.800000</td>
      <td>2.700000</td>
      <td>1.800000</td>
      <td>1.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>89.300000</td>
      <td>10.700000</td>
      <td>76.050000</td>
      <td>23.950000</td>
      <td>92.450000</td>
      <td>7.550000</td>
      <td>94.700000</td>
      <td>5.300000</td>
      <td>91.900000</td>
      <td>8.100000</td>
      <td>93.100000</td>
      <td>6.900000</td>
      <td>94.500000</td>
      <td>5.500000</td>
      <td>86.700000</td>
      <td>13.300000</td>
      <td>77.900000</td>
      <td>22.100000</td>
      <td>45.000000</td>
      <td>75.500000</td>
      <td>24.500000</td>
      <td>97.100000</td>
      <td>91.200000</td>
      <td>8.800000</td>
      <td>99.300000</td>
      <td>92.700000</td>
      <td>7.300000</td>
      <td>98.500000</td>
      <td>94.950000</td>
      <td>5.050000</td>
      <td>72.000000</td>
      <td>93.450000</td>
      <td>6.550000</td>
      <td>34.500000</td>
      <td>89.500000</td>
      <td>10.500000</td>
      <td>11.600000</td>
      <td>86.400000</td>
      <td>13.600000</td>
      <td>2.300000</td>
      <td>81.300000</td>
      <td>18.700000</td>
      <td>84.600000</td>
      <td>15.400000</td>
      <td>86.500000</td>
      <td>13.500000</td>
      <td>84.400000</td>
      <td>15.600000</td>
      <td>88.900000</td>
      <td>11.100000</td>
      <td>91.200000</td>
      <td>8.800000</td>
      <td>90.200000</td>
      <td>9.800000</td>
      <td>89.800000</td>
      <td>88.700000</td>
      <td>91.200000</td>
      <td>17.000000</td>
      <td>16.200000</td>
      <td>18.100000</td>
      <td>99.100000</td>
      <td>0.900000</td>
      <td>72.650000</td>
      <td>27.350000</td>
      <td>74.700000</td>
      <td>25.300000</td>
      <td>80.400000</td>
      <td>19.600000</td>
      <td>74.700000</td>
      <td>25.300000</td>
      <td>74.300000</td>
      <td>25.700000</td>
      <td>80.400000</td>
      <td>19.600000</td>
      <td>83.900000</td>
      <td>16.100000</td>
      <td>99.400000</td>
      <td>0.600000</td>
      <td>78.850000</td>
      <td>21.150000</td>
      <td>84.000000</td>
      <td>16.000000</td>
      <td>79.45000</td>
      <td>20.550000</td>
      <td>11.000000</td>
      <td>89.000000</td>
      <td>3.500000</td>
      <td>16.600000</td>
      <td>0.000000</td>
      <td>23.450000</td>
      <td>12.200000</td>
      <td>9.700000</td>
      <td>11.000000</td>
      <td>10.700000</td>
      <td>7.900000</td>
      <td>10.400000</td>
      <td>12.950000</td>
      <td>10.700000</td>
      <td>13.750000</td>
      <td>3.450000</td>
      <td>21.800000</td>
      <td>11.900000</td>
      <td>16.000000</td>
      <td>13.900000</td>
      <td>11.800000</td>
      <td>4.800000</td>
      <td>12.800000</td>
      <td>16.600000</td>
      <td>14.000000</td>
      <td>38.300000</td>
      <td>6.800000</td>
      <td>12.800000</td>
      <td>9.900000</td>
      <td>22.000000</td>
      <td>8.600000</td>
      <td>10.900000</td>
      <td>17.800000</td>
      <td>14.150000</td>
      <td>8.800000</td>
      <td>5.600000</td>
      <td>3.600000</td>
      <td>11.000000</td>
      <td>20.400000</td>
      <td>15.700000</td>
      <td>6.600000</td>
      <td>69.800000</td>
      <td>50.000000</td>
      <td>58.000000</td>
      <td>44.900000</td>
      <td>13.100000</td>
      <td>4.400000</td>
      <td>1.100000</td>
      <td>0.200000</td>
      <td>36.300000</td>
      <td>16.100000</td>
      <td>18.900000</td>
      <td>3.300000</td>
      <td>17.700000</td>
      <td>11.800000</td>
      <td>2.100000</td>
      <td>0.200000</td>
      <td>11.000000</td>
      <td>86.600000</td>
      <td>13.400000</td>
      <td>80.600000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.100000</td>
      <td>1.400000</td>
      <td>0.700000</td>
      <td>0.600000</td>
      <td>6.900000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>10.300000</td>
      <td>14.200000</td>
      <td>10.200000</td>
      <td>15.600000</td>
      <td>10.000000</td>
      <td>10.100000</td>
      <td>5.500000</td>
      <td>16.200000</td>
      <td>0.700000</td>
      <td>1.200000</td>
      <td>5.800000</td>
      <td>14.000000</td>
      <td>20.900000</td>
      <td>19.800000</td>
      <td>13.700000</td>
      <td>9.200000</td>
      <td>10.300000</td>
      <td>0.800000</td>
      <td>6.800000</td>
      <td>26.300000</td>
      <td>45.40000</td>
      <td>14.300000</td>
      <td>3.300000</td>
      <td>82.300000</td>
      <td>17.700000</td>
      <td>17.500000</td>
      <td>36.200000</td>
      <td>20.600000</td>
      <td>10.200000</td>
      <td>7.200000</td>
      <td>5.600000</td>
      <td>4.700000</td>
      <td>31.500000</td>
      <td>40.900000</td>
      <td>21.300000</td>
      <td>57.300000</td>
      <td>19.300000</td>
      <td>6.400000</td>
      <td>2.300000</td>
      <td>0.000000</td>
      <td>7.200000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>2.500000</td>
      <td>98.800000</td>
      <td>0.900000</td>
      <td>0.100000</td>
      <td>13.900000</td>
      <td>28.400000</td>
      <td>19.800000</td>
      <td>14.600000</td>
      <td>10.500000</td>
      <td>4.200000</td>
      <td>0.900000</td>
      <td>0.200000</td>
      <td>60.200000</td>
      <td>39.800000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>8.100000</td>
      <td>24.000000</td>
      <td>34.900000</td>
      <td>15.900000</td>
      <td>8.100000</td>
      <td>0.100000</td>
      <td>4.500000</td>
      <td>15.000000</td>
      <td>21.500000</td>
      <td>55.100000</td>
      <td>39.700000</td>
      <td>15.900000</td>
      <td>11.200000</td>
      <td>7.300000</td>
      <td>24.400000</td>
      <td>36.300000</td>
      <td>20.800000</td>
      <td>13.100000</td>
      <td>7.700000</td>
      <td>5.000000</td>
      <td>3.200000</td>
      <td>11.300000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>10.550000</td>
      <td>34.600000</td>
      <td>25.200000</td>
      <td>13.800000</td>
      <td>0.800000</td>
      <td>11.900000</td>
      <td>11.100000</td>
      <td>11.350000</td>
      <td>10.800000</td>
      <td>7.700000</td>
      <td>41.400000</td>
      <td>29.110000</td>
      <td>14.720000</td>
      <td>20.950000</td>
      <td>20.390000</td>
      <td>14.590000</td>
      <td>7.320000</td>
      <td>13.360000</td>
      <td>15.710000</td>
      <td>14.350000</td>
      <td>13.870000</td>
      <td>24.540000</td>
      <td>23.170000</td>
      <td>13.510000</td>
      <td>11.700000</td>
      <td>23.240000</td>
      <td>13.140000</td>
      <td>10.110000</td>
      <td>4.480000</td>
      <td>9.980000</td>
      <td>7.420000</td>
      <td>6.080000</td>
      <td>8.780000</td>
      <td>30.210000</td>
      <td>28.550000</td>
      <td>33.120000</td>
      <td>13.060000</td>
      <td>2.710000</td>
      <td>16.440000</td>
      <td>18.620000</td>
      <td>27.240000</td>
      <td>24.620000</td>
      <td>29.650000</td>
      <td>3.660000</td>
      <td>38.490000</td>
      <td>36.960000</td>
      <td>10.33000</td>
      <td>5.360000</td>
      <td>30.960000</td>
      <td>18.430000</td>
      <td>8.820000</td>
      <td>42.410000</td>
      <td>10.190000</td>
      <td>5.300000</td>
      <td>30.460000</td>
      <td>9.610000</td>
      <td>5.130000</td>
      <td>29.080000</td>
      <td>9.220000</td>
      <td>4.950000</td>
      <td>28.550000</td>
      <td>8.100000</td>
      <td>3.800000</td>
      <td>25.910000</td>
      <td>2.840000</td>
      <td>1.570000</td>
      <td>8.367500</td>
      <td>3.820000</td>
      <td>2.955000</td>
      <td>7.635000</td>
      <td>26.530000</td>
      <td>15.632500</td>
      <td>45.737500</td>
      <td>5.760000</td>
      <td>3.640000</td>
      <td>15.780000</td>
      <td>20.335000</td>
      <td>13.010000</td>
      <td>11.600000</td>
      <td>6.310000</td>
      <td>32.660000</td>
      <td>10.000000</td>
      <td>4.900000</td>
      <td>31.560000</td>
      <td>3.240000</td>
      <td>1.777500</td>
      <td>4.710000</td>
      <td>3.460000</td>
      <td>11.360000</td>
      <td>15.540000</td>
      <td>6.210000</td>
      <td>37.100000</td>
      <td>23.760000</td>
      <td>13.305000</td>
      <td>7.780000</td>
      <td>3.680000</td>
      <td>23.432500</td>
      <td>11.332500</td>
      <td>5.055000</td>
      <td>35.160000</td>
      <td>15.920000</td>
      <td>11.200000</td>
      <td>17.480000</td>
      <td>8.900000</td>
      <td>54.283333</td>
      <td>15.660000</td>
      <td>8.560000</td>
      <td>32.007500</td>
      <td>3.440000</td>
      <td>2.590000</td>
      <td>9.000000</td>
      <td>1.467500</td>
      <td>0.940000</td>
      <td>88.660000</td>
      <td>11.340000</td>
      <td>74.367500</td>
      <td>25.632500</td>
      <td>91.360000</td>
      <td>8.640000</td>
      <td>90.910000</td>
      <td>9.090000</td>
      <td>90.300000</td>
      <td>9.700000</td>
      <td>91.200000</td>
      <td>8.800000</td>
      <td>93.210000</td>
      <td>6.790000</td>
      <td>84.857500</td>
      <td>15.142500</td>
      <td>75.350000</td>
      <td>24.660000</td>
      <td>45.010000</td>
      <td>74.045000</td>
      <td>25.970000</td>
      <td>96.380000</td>
      <td>89.460000</td>
      <td>10.540000</td>
      <td>98.360000</td>
      <td>91.230000</td>
      <td>8.770000</td>
      <td>97.220000</td>
      <td>93.500000</td>
      <td>6.500000</td>
      <td>70.590000</td>
      <td>90.902500</td>
      <td>9.097500</td>
      <td>35.695000</td>
      <td>86.770000</td>
      <td>13.230000</td>
      <td>11.850000</td>
      <td>82.660000</td>
      <td>17.360000</td>
      <td>2.400000</td>
      <td>78.410000</td>
      <td>21.590000</td>
      <td>82.980000</td>
      <td>17.020000</td>
      <td>84.120000</td>
      <td>15.880000</td>
      <td>82.770000</td>
      <td>17.230000</td>
      <td>87.170000</td>
      <td>12.830000</td>
      <td>87.970000</td>
      <td>12.030000</td>
      <td>87.442500</td>
      <td>12.557500</td>
      <td>89.260000</td>
      <td>88.240000</td>
      <td>90.420000</td>
      <td>18.200000</td>
      <td>17.340000</td>
      <td>19.110000</td>
      <td>98.880000</td>
      <td>1.120000</td>
      <td>71.860000</td>
      <td>28.140000</td>
      <td>72.740000</td>
      <td>27.260000</td>
      <td>78.060000</td>
      <td>21.940000</td>
      <td>72.740000</td>
      <td>27.260000</td>
      <td>72.210000</td>
      <td>27.790000</td>
      <td>78.060000</td>
      <td>21.940000</td>
      <td>80.460000</td>
      <td>19.540000</td>
      <td>99.320000</td>
      <td>0.680000</td>
      <td>77.800000</td>
      <td>22.200000</td>
      <td>79.217500</td>
      <td>20.782500</td>
      <td>77.700000</td>
      <td>22.300000</td>
      <td>11.440000</td>
      <td>88.560000</td>
      <td>4.120000</td>
      <td>17.240000</td>
      <td>0.180000</td>
      <td>24.400000</td>
      <td>12.760000</td>
      <td>10.130000</td>
      <td>11.460000</td>
      <td>11.240000</td>
      <td>10.920000</td>
      <td>10.940000</td>
      <td>16.075000</td>
      <td>11.140000</td>
      <td>18.030000</td>
      <td>9.220000</td>
      <td>27.263333</td>
      <td>12.360000</td>
      <td>16.830000</td>
      <td>14.010000</td>
      <td>12.200000</td>
      <td>6.010000</td>
      <td>13.380000</td>
      <td>17.250000</td>
      <td>14.710000</td>
      <td>38.760000</td>
      <td>7.350000</td>
      <td>13.380000</td>
      <td>11.020000</td>
      <td>22.950000</td>
      <td>9.000000</td>
      <td>11.380000</td>
      <td>18.520000</td>
      <td>14.270000</td>
      <td>9.440000</td>
      <td>6.440000</td>
      <td>4.630000</td>
      <td>11.450000</td>
      <td>21.300000</td>
      <td>16.270000</td>
      <td>7.130000</td>
      <td>68.350000</td>
      <td>48.840000</td>
      <td>57.080000</td>
      <td>43.900000</td>
      <td>13.080000</td>
      <td>4.490000</td>
      <td>1.290000</td>
      <td>0.300000</td>
      <td>36.680000</td>
      <td>17.020000</td>
      <td>18.770000</td>
      <td>3.480000</td>
      <td>18.720000</td>
      <td>12.560000</td>
      <td>2.120000</td>
      <td>0.280000</td>
      <td>11.440000</td>
      <td>84.350000</td>
      <td>15.650000</td>
      <td>80.570000</td>
      <td>1.340000</td>
      <td>1.320000</td>
      <td>1.440000</td>
      <td>1.740000</td>
      <td>1.200000</td>
      <td>1.340000</td>
      <td>7.780000</td>
      <td>0.000000</td>
      <td>0.320000</td>
      <td>10.450000</td>
      <td>13.980000</td>
      <td>10.380000</td>
      <td>15.660000</td>
      <td>10.230000</td>
      <td>10.360000</td>
      <td>5.850000</td>
      <td>17.980000</td>
      <td>0.880000</td>
      <td>1.360000</td>
      <td>6.120000</td>
      <td>13.630000</td>
      <td>20.960000</td>
      <td>20.040000</td>
      <td>13.840000</td>
      <td>9.330000</td>
      <td>10.820000</td>
      <td>0.980000</td>
      <td>7.300000</td>
      <td>25.950000</td>
      <td>45.520000</td>
      <td>14.920000</td>
      <td>3.540000</td>
      <td>81.910000</td>
      <td>18.090000</td>
      <td>17.350000</td>
      <td>36.140000</td>
      <td>20.600000</td>
      <td>10.430000</td>
      <td>7.310000</td>
      <td>5.960000</td>
      <td>4.980000</td>
      <td>31.300000</td>
      <td>40.900000</td>
      <td>21.400000</td>
      <td>53.240000</td>
      <td>22.440000</td>
      <td>6.510000</td>
      <td>3.020000</td>
      <td>0.000000</td>
      <td>9.040000</td>
      <td>0.00000</td>
      <td>1.190000</td>
      <td>0.320000</td>
      <td>0.320000</td>
      <td>0.530000</td>
      <td>2.560000</td>
      <td>98.640000</td>
      <td>1.040000</td>
      <td>0.240000</td>
      <td>15.020000</td>
      <td>29.000000</td>
      <td>19.800000</td>
      <td>14.640000</td>
      <td>11.010000</td>
      <td>4.610000</td>
      <td>1.220000</td>
      <td>0.380000</td>
      <td>60.310000</td>
      <td>39.690000</td>
      <td>0.160000</td>
      <td>2.240000</td>
      <td>8.640000</td>
      <td>25.380000</td>
      <td>34.900000</td>
      <td>15.460000</td>
      <td>8.550000</td>
      <td>0.600000</td>
      <td>4.81000</td>
      <td>15.360000</td>
      <td>21.870000</td>
      <td>55.540000</td>
      <td>39.520000</td>
      <td>16.000000</td>
      <td>11.120000</td>
      <td>7.460000</td>
      <td>25.190000</td>
      <td>36.170000</td>
      <td>21.30000</td>
      <td>13.190000</td>
      <td>7.880000</td>
      <td>5.120000</td>
      <td>3.360000</td>
      <td>11.630000</td>
      <td>1.220000</td>
      <td>3.277500</td>
      <td>11.730000</td>
      <td>34.690000</td>
      <td>25.710000</td>
      <td>15.770000</td>
      <td>2.210000</td>
      <td>12.980000</td>
      <td>11.780000</td>
      <td>11.560000</td>
      <td>10.900000</td>
      <td>8.000000</td>
      <td>41.930000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49506.750000</td>
      <td>49506.750000</td>
      <td>38.350000</td>
      <td>1118.750000</td>
      <td>172.750000</td>
      <td>22.600000</td>
      <td>168.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.500000</td>
      <td>0.000000</td>
      <td>19.100000</td>
      <td>29.875000</td>
      <td>29.300000</td>
      <td>18.900000</td>
      <td>10.000000</td>
      <td>18.300000</td>
      <td>20.500000</td>
      <td>18.700000</td>
      <td>18.000000</td>
      <td>39.300000</td>
      <td>37.075000</td>
      <td>17.700000</td>
      <td>14.900000</td>
      <td>30.850000</td>
      <td>16.900000</td>
      <td>13.700000</td>
      <td>6.700000</td>
      <td>13.525000</td>
      <td>10.300000</td>
      <td>8.800000</td>
      <td>12.200000</td>
      <td>40.875000</td>
      <td>40.000000</td>
      <td>46.500000</td>
      <td>16.900000</td>
      <td>4.200000</td>
      <td>22.000000</td>
      <td>24.100000</td>
      <td>33.700000</td>
      <td>31.500000</td>
      <td>37.500000</td>
      <td>6.225000</td>
      <td>48.125000</td>
      <td>46.100000</td>
      <td>14.200000</td>
      <td>7.600000</td>
      <td>42.100000</td>
      <td>26.000000</td>
      <td>13.650000</td>
      <td>56.900000</td>
      <td>14.100000</td>
      <td>7.50000</td>
      <td>42.000000</td>
      <td>13.100000</td>
      <td>7.300000</td>
      <td>40.500000</td>
      <td>12.800000</td>
      <td>7.000000</td>
      <td>40.100000</td>
      <td>11.800000</td>
      <td>5.800000</td>
      <td>37.500000</td>
      <td>4.800000</td>
      <td>2.800000</td>
      <td>14.700000</td>
      <td>6.000000</td>
      <td>4.700000</td>
      <td>14.325000</td>
      <td>38.500000</td>
      <td>25.000000</td>
      <td>67.700000</td>
      <td>8.800000</td>
      <td>5.600000</td>
      <td>26.500000</td>
      <td>31.900000</td>
      <td>21.225000</td>
      <td>16.000000</td>
      <td>9.625000</td>
      <td>47.575000</td>
      <td>14.200000</td>
      <td>7.400000</td>
      <td>47.15000</td>
      <td>4.800000</td>
      <td>2.900000</td>
      <td>6.800000</td>
      <td>5.100000</td>
      <td>18.800000</td>
      <td>22.800000</td>
      <td>10.400000</td>
      <td>52.600000</td>
      <td>35.900000</td>
      <td>22.200000</td>
      <td>10.900000</td>
      <td>5.400000</td>
      <td>34.825000</td>
      <td>17.150000</td>
      <td>7.900000</td>
      <td>50.000000</td>
      <td>24.675000</td>
      <td>18.175000</td>
      <td>25.000000</td>
      <td>13.300000</td>
      <td>75.000000</td>
      <td>22.000000</td>
      <td>12.925000</td>
      <td>45.025000</td>
      <td>5.600000</td>
      <td>4.400000</td>
      <td>16.600000</td>
      <td>2.050000</td>
      <td>1.000000</td>
      <td>92.900000</td>
      <td>14.800000</td>
      <td>92.200000</td>
      <td>42.600000</td>
      <td>96.075000</td>
      <td>12.400000</td>
      <td>100.000000</td>
      <td>15.475000</td>
      <td>96.900000</td>
      <td>15.150000</td>
      <td>97.400000</td>
      <td>13.000000</td>
      <td>98.550000</td>
      <td>9.950000</td>
      <td>92.700000</td>
      <td>20.500000</td>
      <td>100.000000</td>
      <td>36.800000</td>
      <td>57.200000</td>
      <td>95.200000</td>
      <td>43.525000</td>
      <td>100.000000</td>
      <td>95.900000</td>
      <td>16.100000</td>
      <td>100.000000</td>
      <td>97.100000</td>
      <td>12.900000</td>
      <td>100.000000</td>
      <td>99.575000</td>
      <td>9.700000</td>
      <td>84.700000</td>
      <td>100.000000</td>
      <td>15.125000</td>
      <td>46.200000</td>
      <td>100.000000</td>
      <td>20.000000</td>
      <td>16.200000</td>
      <td>100.000000</td>
      <td>25.100000</td>
      <td>3.100000</td>
      <td>92.000000</td>
      <td>31.700000</td>
      <td>90.900000</td>
      <td>22.300000</td>
      <td>94.500000</td>
      <td>22.600000</td>
      <td>92.500000</td>
      <td>23.800000</td>
      <td>97.475000</td>
      <td>18.375000</td>
      <td>100.000000</td>
      <td>18.800000</td>
      <td>100.000000</td>
      <td>19.600000</td>
      <td>93.000000</td>
      <td>92.500000</td>
      <td>94.000000</td>
      <td>25.800000</td>
      <td>26.300000</td>
      <td>25.900000</td>
      <td>99.600000</td>
      <td>2.000000</td>
      <td>83.500000</td>
      <td>38.325000</td>
      <td>89.100000</td>
      <td>39.400000</td>
      <td>95.025000</td>
      <td>34.525000</td>
      <td>89.100000</td>
      <td>39.400000</td>
      <td>93.850000</td>
      <td>45.100000</td>
      <td>95.025000</td>
      <td>34.525000</td>
      <td>100.000000</td>
      <td>33.300000</td>
      <td>99.800000</td>
      <td>1.200000</td>
      <td>90.000000</td>
      <td>32.075000</td>
      <td>100.000000</td>
      <td>32.050000</td>
      <td>95.75000</td>
      <td>34.675000</td>
      <td>14.100000</td>
      <td>91.700000</td>
      <td>6.000000</td>
      <td>21.100000</td>
      <td>0.300000</td>
      <td>33.200000</td>
      <td>15.800000</td>
      <td>12.600000</td>
      <td>14.200000</td>
      <td>13.700000</td>
      <td>17.100000</td>
      <td>13.500000</td>
      <td>24.175000</td>
      <td>13.600000</td>
      <td>28.600000</td>
      <td>14.675000</td>
      <td>46.550000</td>
      <td>15.200000</td>
      <td>22.975000</td>
      <td>17.500000</td>
      <td>15.300000</td>
      <td>8.400000</td>
      <td>16.300000</td>
      <td>21.825000</td>
      <td>18.700000</td>
      <td>46.500000</td>
      <td>9.500000</td>
      <td>16.300000</td>
      <td>14.450000</td>
      <td>28.400000</td>
      <td>11.900000</td>
      <td>14.000000</td>
      <td>22.625000</td>
      <td>17.900000</td>
      <td>12.400000</td>
      <td>9.400000</td>
      <td>6.825000</td>
      <td>14.200000</td>
      <td>25.600000</td>
      <td>20.300000</td>
      <td>9.100000</td>
      <td>77.700000</td>
      <td>59.300000</td>
      <td>67.700000</td>
      <td>54.000000</td>
      <td>16.100000</td>
      <td>5.900000</td>
      <td>2.000000</td>
      <td>0.500000</td>
      <td>43.900000</td>
      <td>21.000000</td>
      <td>23.400000</td>
      <td>4.500000</td>
      <td>23.200000</td>
      <td>16.300000</td>
      <td>3.200000</td>
      <td>0.500000</td>
      <td>14.100000</td>
      <td>92.400000</td>
      <td>31.800000</td>
      <td>87.600000</td>
      <td>3.300000</td>
      <td>2.200000</td>
      <td>2.600000</td>
      <td>3.400000</td>
      <td>2.500000</td>
      <td>3.300000</td>
      <td>12.500000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>14.000000</td>
      <td>19.100000</td>
      <td>13.400000</td>
      <td>19.500000</td>
      <td>13.100000</td>
      <td>14.200000</td>
      <td>8.200000</td>
      <td>24.900000</td>
      <td>1.600000</td>
      <td>2.800000</td>
      <td>9.000000</td>
      <td>18.000000</td>
      <td>24.200000</td>
      <td>22.900000</td>
      <td>16.600000</td>
      <td>11.900000</td>
      <td>14.000000</td>
      <td>1.800000</td>
      <td>10.400000</td>
      <td>31.800000</td>
      <td>51.70000</td>
      <td>18.800000</td>
      <td>4.800000</td>
      <td>87.700000</td>
      <td>25.700000</td>
      <td>21.900000</td>
      <td>39.600000</td>
      <td>24.000000</td>
      <td>12.400000</td>
      <td>9.400000</td>
      <td>8.100000</td>
      <td>7.300000</td>
      <td>37.200000</td>
      <td>44.700000</td>
      <td>26.400000</td>
      <td>84.900000</td>
      <td>35.500000</td>
      <td>8.900000</td>
      <td>5.100000</td>
      <td>0.000000</td>
      <td>15.800000</td>
      <td>0.000000</td>
      <td>2.100000</td>
      <td>0.500000</td>
      <td>0.600000</td>
      <td>0.900000</td>
      <td>3.500000</td>
      <td>99.500000</td>
      <td>1.600000</td>
      <td>0.400000</td>
      <td>20.400000</td>
      <td>36.600000</td>
      <td>24.200000</td>
      <td>19.700000</td>
      <td>16.900000</td>
      <td>7.800000</td>
      <td>2.200000</td>
      <td>0.700000</td>
      <td>67.100000</td>
      <td>46.700000</td>
      <td>0.200000</td>
      <td>3.700000</td>
      <td>12.975000</td>
      <td>31.700000</td>
      <td>40.100000</td>
      <td>23.000000</td>
      <td>15.500000</td>
      <td>1.000000</td>
      <td>7.700000</td>
      <td>20.900000</td>
      <td>26.300000</td>
      <td>68.800000</td>
      <td>44.700000</td>
      <td>18.300000</td>
      <td>13.300000</td>
      <td>9.300000</td>
      <td>30.075000</td>
      <td>41.100000</td>
      <td>24.600000</td>
      <td>15.800000</td>
      <td>9.800000</td>
      <td>6.700000</td>
      <td>4.800000</td>
      <td>14.400000</td>
      <td>2.200000</td>
      <td>5.800000</td>
      <td>18.100000</td>
      <td>42.700000</td>
      <td>32.400000</td>
      <td>24.025000</td>
      <td>4.900000</td>
      <td>17.700000</td>
      <td>15.700000</td>
      <td>15.500000</td>
      <td>14.700000</td>
      <td>11.000000</td>
      <td>50.000000</td>
      <td>37.785000</td>
      <td>18.570000</td>
      <td>28.890000</td>
      <td>28.415000</td>
      <td>18.615000</td>
      <td>9.440000</td>
      <td>17.470000</td>
      <td>20.205000</td>
      <td>18.280000</td>
      <td>17.535000</td>
      <td>33.457500</td>
      <td>33.270000</td>
      <td>16.920000</td>
      <td>14.635000</td>
      <td>28.360000</td>
      <td>16.275000</td>
      <td>13.205000</td>
      <td>6.371250</td>
      <td>13.300000</td>
      <td>10.275000</td>
      <td>9.020000</td>
      <td>11.700000</td>
      <td>36.920000</td>
      <td>35.030000</td>
      <td>41.535000</td>
      <td>16.390000</td>
      <td>4.220000</td>
      <td>21.725000</td>
      <td>23.320000</td>
      <td>31.780000</td>
      <td>29.620000</td>
      <td>35.110000</td>
      <td>5.960000</td>
      <td>45.115000</td>
      <td>42.915000</td>
      <td>13.47500</td>
      <td>7.560000</td>
      <td>38.000000</td>
      <td>25.350000</td>
      <td>13.290000</td>
      <td>51.715000</td>
      <td>13.180000</td>
      <td>7.535000</td>
      <td>37.677500</td>
      <td>12.440000</td>
      <td>7.280000</td>
      <td>36.975000</td>
      <td>12.200000</td>
      <td>7.055000</td>
      <td>36.435000</td>
      <td>11.515000</td>
      <td>5.695000</td>
      <td>33.750000</td>
      <td>4.600000</td>
      <td>2.820000</td>
      <td>13.193750</td>
      <td>5.220000</td>
      <td>4.300000</td>
      <td>12.825000</td>
      <td>33.750000</td>
      <td>21.935000</td>
      <td>57.861250</td>
      <td>8.155000</td>
      <td>5.223750</td>
      <td>22.981250</td>
      <td>28.150000</td>
      <td>18.500000</td>
      <td>15.395000</td>
      <td>9.095000</td>
      <td>41.500000</td>
      <td>13.720000</td>
      <td>6.850000</td>
      <td>40.615000</td>
      <td>5.495000</td>
      <td>3.460000</td>
      <td>6.690000</td>
      <td>4.875000</td>
      <td>17.678750</td>
      <td>21.310000</td>
      <td>9.375000</td>
      <td>48.087500</td>
      <td>32.590000</td>
      <td>20.395000</td>
      <td>10.315000</td>
      <td>5.120000</td>
      <td>31.180000</td>
      <td>16.240000</td>
      <td>7.678750</td>
      <td>43.946250</td>
      <td>23.415000</td>
      <td>17.395000</td>
      <td>22.715000</td>
      <td>12.040000</td>
      <td>64.630000</td>
      <td>20.295000</td>
      <td>12.015000</td>
      <td>39.595000</td>
      <td>5.440000</td>
      <td>4.220000</td>
      <td>15.360000</td>
      <td>3.200000</td>
      <td>2.260000</td>
      <td>91.110000</td>
      <td>13.960000</td>
      <td>83.757500</td>
      <td>35.935000</td>
      <td>93.675000</td>
      <td>11.335000</td>
      <td>95.472500</td>
      <td>15.535000</td>
      <td>93.495000</td>
      <td>13.340000</td>
      <td>94.020000</td>
      <td>12.055000</td>
      <td>95.615000</td>
      <td>9.435000</td>
      <td>89.675000</td>
      <td>19.435000</td>
      <td>83.775000</td>
      <td>32.737500</td>
      <td>52.080000</td>
      <td>83.715000</td>
      <td>37.425000</td>
      <td>97.500000</td>
      <td>92.571250</td>
      <td>14.450000</td>
      <td>99.100000</td>
      <td>93.995000</td>
      <td>11.795000</td>
      <td>98.460000</td>
      <td>95.955000</td>
      <td>9.215000</td>
      <td>77.423750</td>
      <td>94.710000</td>
      <td>15.095000</td>
      <td>43.755000</td>
      <td>91.870000</td>
      <td>18.885000</td>
      <td>15.160000</td>
      <td>89.860000</td>
      <td>23.760000</td>
      <td>3.040000</td>
      <td>85.150000</td>
      <td>27.415000</td>
      <td>87.735000</td>
      <td>20.970000</td>
      <td>89.880000</td>
      <td>21.195000</td>
      <td>88.240000</td>
      <td>22.335000</td>
      <td>92.300000</td>
      <td>18.270000</td>
      <td>93.242500</td>
      <td>19.115000</td>
      <td>92.805000</td>
      <td>18.155000</td>
      <td>91.580000</td>
      <td>90.755000</td>
      <td>92.855000</td>
      <td>25.080000</td>
      <td>25.500000</td>
      <td>25.120000</td>
      <td>99.320000</td>
      <td>2.415000</td>
      <td>78.120000</td>
      <td>34.135000</td>
      <td>80.185000</td>
      <td>35.515000</td>
      <td>85.375000</td>
      <td>29.515000</td>
      <td>80.185000</td>
      <td>35.515000</td>
      <td>80.985000</td>
      <td>37.750000</td>
      <td>85.375000</td>
      <td>29.515000</td>
      <td>88.437500</td>
      <td>27.995000</td>
      <td>99.600000</td>
      <td>1.220000</td>
      <td>84.045000</td>
      <td>28.425000</td>
      <td>86.072917</td>
      <td>28.055000</td>
      <td>84.670000</td>
      <td>29.336250</td>
      <td>13.560000</td>
      <td>90.460000</td>
      <td>5.860000</td>
      <td>20.515000</td>
      <td>0.520000</td>
      <td>31.195000</td>
      <td>15.220000</td>
      <td>12.180000</td>
      <td>13.500000</td>
      <td>13.280000</td>
      <td>16.935000</td>
      <td>13.055000</td>
      <td>22.395000</td>
      <td>13.120000</td>
      <td>26.061250</td>
      <td>14.710000</td>
      <td>39.285000</td>
      <td>14.360000</td>
      <td>20.815000</td>
      <td>16.665000</td>
      <td>14.680000</td>
      <td>8.355000</td>
      <td>15.440000</td>
      <td>21.135000</td>
      <td>18.275000</td>
      <td>43.475000</td>
      <td>9.135000</td>
      <td>15.440000</td>
      <td>14.135000</td>
      <td>27.055000</td>
      <td>11.015000</td>
      <td>13.475000</td>
      <td>21.150000</td>
      <td>16.480000</td>
      <td>11.680000</td>
      <td>8.300000</td>
      <td>7.071250</td>
      <td>13.615000</td>
      <td>24.075000</td>
      <td>18.930000</td>
      <td>8.640000</td>
      <td>74.315000</td>
      <td>57.245000</td>
      <td>64.315000</td>
      <td>51.905000</td>
      <td>15.295000</td>
      <td>5.660000</td>
      <td>2.060000</td>
      <td>0.500000</td>
      <td>42.695000</td>
      <td>20.635000</td>
      <td>22.710000</td>
      <td>4.520000</td>
      <td>22.660000</td>
      <td>15.855000</td>
      <td>3.060000</td>
      <td>0.440000</td>
      <td>13.560000</td>
      <td>91.480000</td>
      <td>32.395000</td>
      <td>85.035000</td>
      <td>3.655000</td>
      <td>2.155000</td>
      <td>2.500000</td>
      <td>3.155000</td>
      <td>2.715000</td>
      <td>3.615000</td>
      <td>11.335000</td>
      <td>0.040000</td>
      <td>0.580000</td>
      <td>13.155000</td>
      <td>18.015000</td>
      <td>12.395000</td>
      <td>18.435000</td>
      <td>12.595000</td>
      <td>13.790000</td>
      <td>7.880000</td>
      <td>24.700000</td>
      <td>1.775000</td>
      <td>2.800000</td>
      <td>8.815000</td>
      <td>17.210000</td>
      <td>23.440000</td>
      <td>22.120000</td>
      <td>16.180000</td>
      <td>11.357500</td>
      <td>13.440000</td>
      <td>1.860000</td>
      <td>10.155000</td>
      <td>30.140000</td>
      <td>50.255000</td>
      <td>17.795000</td>
      <td>4.447500</td>
      <td>85.000000</td>
      <td>25.025000</td>
      <td>22.080000</td>
      <td>38.475000</td>
      <td>22.780000</td>
      <td>11.740000</td>
      <td>8.840000</td>
      <td>7.855000</td>
      <td>6.995000</td>
      <td>35.660000</td>
      <td>43.335000</td>
      <td>25.315000</td>
      <td>79.525000</td>
      <td>32.660000</td>
      <td>8.315000</td>
      <td>5.240000</td>
      <td>0.040000</td>
      <td>15.510000</td>
      <td>0.02000</td>
      <td>2.140000</td>
      <td>0.520000</td>
      <td>0.560000</td>
      <td>0.820000</td>
      <td>3.100000</td>
      <td>98.980000</td>
      <td>1.540000</td>
      <td>0.440000</td>
      <td>20.410000</td>
      <td>34.930000</td>
      <td>22.960000</td>
      <td>17.975000</td>
      <td>15.080000</td>
      <td>7.210000</td>
      <td>2.220000</td>
      <td>0.680000</td>
      <td>65.745000</td>
      <td>46.870000</td>
      <td>0.300000</td>
      <td>3.480000</td>
      <td>13.075000</td>
      <td>30.515000</td>
      <td>38.675000</td>
      <td>20.975000</td>
      <td>14.505000</td>
      <td>1.000000</td>
      <td>7.68000</td>
      <td>19.760000</td>
      <td>25.010000</td>
      <td>65.735000</td>
      <td>42.875000</td>
      <td>17.315000</td>
      <td>12.455000</td>
      <td>8.495000</td>
      <td>28.855000</td>
      <td>38.910000</td>
      <td>23.43500</td>
      <td>14.740000</td>
      <td>9.160000</td>
      <td>6.120000</td>
      <td>4.220000</td>
      <td>13.500000</td>
      <td>2.360000</td>
      <td>5.471250</td>
      <td>17.230000</td>
      <td>40.100000</td>
      <td>29.657500</td>
      <td>22.785000</td>
      <td>6.100000</td>
      <td>16.195000</td>
      <td>15.171250</td>
      <td>14.110000</td>
      <td>13.135000</td>
      <td>9.971250</td>
      <td>47.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>49971.000000</td>
      <td>49971.000000</td>
      <td>100.000000</td>
      <td>5545.000000</td>
      <td>2074.000000</td>
      <td>240.000000</td>
      <td>1985.000000</td>
      <td>136.000000</td>
      <td>9.000000</td>
      <td>26.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>157.000000</td>
      <td>33.000000</td>
      <td>50.000000</td>
      <td>9.100000</td>
      <td>60.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>55.300000</td>
      <td>77.800000</td>
      <td>72.600000</td>
      <td>58.000000</td>
      <td>60.500000</td>
      <td>78.300000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>75.600000</td>
      <td>49.400000</td>
      <td>100.000000</td>
      <td>66.700000</td>
      <td>59.500000</td>
      <td>50.800000</td>
      <td>45.300000</td>
      <td>58.100000</td>
      <td>47.200000</td>
      <td>65.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>52.100000</td>
      <td>57.700000</td>
      <td>100.000000</td>
      <td>76.900000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>83.300000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>50.000000</td>
      <td>92.300000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>52.200000</td>
      <td>92.30000</td>
      <td>100.000000</td>
      <td>72.500000</td>
      <td>55.800000</td>
      <td>100.000000</td>
      <td>69.900000</td>
      <td>53.700000</td>
      <td>100.000000</td>
      <td>85.700000</td>
      <td>85.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>41.900000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>64.800000</td>
      <td>85.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>85.700000</td>
      <td>85.700000</td>
      <td>34.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>87.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>48.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>85.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>85.700000</td>
      <td>85.700000</td>
      <td>100.000000</td>
      <td>60.000000</td>
      <td>57.100000</td>
      <td>100.000000</td>
      <td>64.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>52.200000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>66.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>70.600000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>15.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>79.400000</td>
      <td>82.900000</td>
      <td>78.900000</td>
      <td>100.000000</td>
      <td>32.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>31.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>52.600000</td>
      <td>100.000000</td>
      <td>44.100000</td>
      <td>77.900000</td>
      <td>9.700000</td>
      <td>100.000000</td>
      <td>52.300000</td>
      <td>54.100000</td>
      <td>46.500000</td>
      <td>47.200000</td>
      <td>100.000000</td>
      <td>47.200000</td>
      <td>100.000000</td>
      <td>52.600000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>67.400000</td>
      <td>100.000000</td>
      <td>63.900000</td>
      <td>100.000000</td>
      <td>65.600000</td>
      <td>62.500000</td>
      <td>82.200000</td>
      <td>58.100000</td>
      <td>100.000000</td>
      <td>75.500000</td>
      <td>62.500000</td>
      <td>53.500000</td>
      <td>100.000000</td>
      <td>75.500000</td>
      <td>52.600000</td>
      <td>69.000000</td>
      <td>72.200000</td>
      <td>70.000000</td>
      <td>59.100000</td>
      <td>100.000000</td>
      <td>52.600000</td>
      <td>100.000000</td>
      <td>86.400000</td>
      <td>49.800000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>55.600000</td>
      <td>16.100000</td>
      <td>23.400000</td>
      <td>6.600000</td>
      <td>100.000000</td>
      <td>63.600000</td>
      <td>100.000000</td>
      <td>27.800000</td>
      <td>72.400000</td>
      <td>59.200000</td>
      <td>40.700000</td>
      <td>7.000000</td>
      <td>52.600000</td>
      <td>100.000000</td>
      <td>88.900000</td>
      <td>100.000000</td>
      <td>38.100000</td>
      <td>22.000000</td>
      <td>35.100000</td>
      <td>22.600000</td>
      <td>37.600000</td>
      <td>91.300000</td>
      <td>73.200000</td>
      <td>2.100000</td>
      <td>4.400000</td>
      <td>43.300000</td>
      <td>50.000000</td>
      <td>37.800000</td>
      <td>100.000000</td>
      <td>68.500000</td>
      <td>78.900000</td>
      <td>46.000000</td>
      <td>81.800000</td>
      <td>22.800000</td>
      <td>22.500000</td>
      <td>37.100000</td>
      <td>47.100000</td>
      <td>55.600000</td>
      <td>47.400000</td>
      <td>41.100000</td>
      <td>100.000000</td>
      <td>48.100000</td>
      <td>22.800000</td>
      <td>62.900000</td>
      <td>73.700000</td>
      <td>100.00000</td>
      <td>55.400000</td>
      <td>22.700000</td>
      <td>100.000000</td>
      <td>93.000000</td>
      <td>71.100000</td>
      <td>81.300000</td>
      <td>100.000000</td>
      <td>30.800000</td>
      <td>35.300000</td>
      <td>55.600000</td>
      <td>82.400000</td>
      <td>97.300000</td>
      <td>100.000000</td>
      <td>47.100000</td>
      <td>100.000000</td>
      <td>82.400000</td>
      <td>90.200000</td>
      <td>100.000000</td>
      <td>2.900000</td>
      <td>58.200000</td>
      <td>3.200000</td>
      <td>21.100000</td>
      <td>4.300000</td>
      <td>36.500000</td>
      <td>17.600000</td>
      <td>15.700000</td>
      <td>100.000000</td>
      <td>34.300000</td>
      <td>6.100000</td>
      <td>100.000000</td>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>48.300000</td>
      <td>56.300000</td>
      <td>36.300000</td>
      <td>28.600000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>57.900000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>76.300000</td>
      <td>25.400000</td>
      <td>78.600000</td>
      <td>73.900000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>83.300000</td>
      <td>100.000000</td>
      <td>31.100000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>68.900000</td>
      <td>76.900000</td>
      <td>100.000000</td>
      <td>28.800000</td>
      <td>33.300000</td>
      <td>100.000000</td>
      <td>28.300000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>84.300000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>63.500000</td>
      <td>100.000000</td>
      <td>66.700000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>78.180000</td>
      <td>49.300000</td>
      <td>62.980000</td>
      <td>62.960000</td>
      <td>47.160000</td>
      <td>33.860000</td>
      <td>49.420000</td>
      <td>49.100000</td>
      <td>49.620000</td>
      <td>53.580000</td>
      <td>100.000000</td>
      <td>91.266667</td>
      <td>54.040000</td>
      <td>42.380000</td>
      <td>60.120000</td>
      <td>44.760000</td>
      <td>37.580000</td>
      <td>20.320000</td>
      <td>35.420000</td>
      <td>24.800000</td>
      <td>23.320000</td>
      <td>26.920000</td>
      <td>63.740000</td>
      <td>62.800000</td>
      <td>78.825000</td>
      <td>44.960000</td>
      <td>15.520000</td>
      <td>46.280000</td>
      <td>57.000000</td>
      <td>53.120000</td>
      <td>53.720000</td>
      <td>58.520000</td>
      <td>23.660000</td>
      <td>77.240000</td>
      <td>71.540000</td>
      <td>44.04000</td>
      <td>28.840000</td>
      <td>68.220000</td>
      <td>56.980000</td>
      <td>37.420000</td>
      <td>86.660000</td>
      <td>43.940000</td>
      <td>28.620000</td>
      <td>68.380000</td>
      <td>43.100000</td>
      <td>32.300000</td>
      <td>72.480000</td>
      <td>41.320000</td>
      <td>24.880000</td>
      <td>75.100000</td>
      <td>38.600000</td>
      <td>27.080000</td>
      <td>68.380000</td>
      <td>26.740000</td>
      <td>24.840000</td>
      <td>60.050000</td>
      <td>20.800000</td>
      <td>22.900000</td>
      <td>59.500000</td>
      <td>60.160000</td>
      <td>66.650000</td>
      <td>100.000000</td>
      <td>24.620000</td>
      <td>20.340000</td>
      <td>61.700000</td>
      <td>67.900000</td>
      <td>59.950000</td>
      <td>45.120000</td>
      <td>34.160000</td>
      <td>78.525000</td>
      <td>45.880000</td>
      <td>30.840000</td>
      <td>79.460000</td>
      <td>29.200000</td>
      <td>23.420000</td>
      <td>26.700000</td>
      <td>24.920000</td>
      <td>52.200000</td>
      <td>52.660000</td>
      <td>32.300000</td>
      <td>83.333333</td>
      <td>79.960000</td>
      <td>51.840000</td>
      <td>37.820000</td>
      <td>25.020000</td>
      <td>66.000000</td>
      <td>46.040000</td>
      <td>26.380000</td>
      <td>87.850000</td>
      <td>56.860000</td>
      <td>46.920000</td>
      <td>67.400000</td>
      <td>41.800000</td>
      <td>100.000000</td>
      <td>47.180000</td>
      <td>34.120000</td>
      <td>74.525000</td>
      <td>24.780000</td>
      <td>23.760000</td>
      <td>81.300000</td>
      <td>17.925000</td>
      <td>15.850000</td>
      <td>97.480000</td>
      <td>35.100000</td>
      <td>100.000000</td>
      <td>72.440000</td>
      <td>99.580000</td>
      <td>25.140000</td>
      <td>100.000000</td>
      <td>57.933333</td>
      <td>99.600000</td>
      <td>39.100000</td>
      <td>100.000000</td>
      <td>31.875000</td>
      <td>100.000000</td>
      <td>25.900000</td>
      <td>100.000000</td>
      <td>47.920000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>82.150000</td>
      <td>100.000000</td>
      <td>75.740000</td>
      <td>100.000000</td>
      <td>99.650000</td>
      <td>40.380000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>29.900000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>30.380000</td>
      <td>95.320000</td>
      <td>100.000000</td>
      <td>54.480000</td>
      <td>70.575000</td>
      <td>100.000000</td>
      <td>53.040000</td>
      <td>34.140000</td>
      <td>100.000000</td>
      <td>70.300000</td>
      <td>5.520000</td>
      <td>100.000000</td>
      <td>64.175000</td>
      <td>99.640000</td>
      <td>43.860000</td>
      <td>100.000000</td>
      <td>59.400000</td>
      <td>100.000000</td>
      <td>63.333333</td>
      <td>100.000000</td>
      <td>66.666667</td>
      <td>100.000000</td>
      <td>70.166667</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>97.950000</td>
      <td>97.700000</td>
      <td>98.150000</td>
      <td>70.740000</td>
      <td>75.660000</td>
      <td>66.300000</td>
      <td>99.920000</td>
      <td>16.240000</td>
      <td>97.400000</td>
      <td>56.140000</td>
      <td>100.000000</td>
      <td>86.666667</td>
      <td>100.000000</td>
      <td>67.200000</td>
      <td>100.000000</td>
      <td>86.666667</td>
      <td>100.000000</td>
      <td>86.666667</td>
      <td>100.000000</td>
      <td>67.200000</td>
      <td>100.000000</td>
      <td>75.000000</td>
      <td>100.000000</td>
      <td>14.220000</td>
      <td>100.000000</td>
      <td>58.650000</td>
      <td>100.000000</td>
      <td>81.375000</td>
      <td>100.000000</td>
      <td>62.625000</td>
      <td>25.960000</td>
      <td>95.160000</td>
      <td>21.260000</td>
      <td>33.900000</td>
      <td>2.600000</td>
      <td>66.020000</td>
      <td>27.940000</td>
      <td>24.960000</td>
      <td>25.640000</td>
      <td>25.680000</td>
      <td>50.620000</td>
      <td>25.660000</td>
      <td>62.033333</td>
      <td>26.460000</td>
      <td>62.820000</td>
      <td>55.625000</td>
      <td>98.150000</td>
      <td>29.640000</td>
      <td>40.460000</td>
      <td>35.080000</td>
      <td>27.040000</td>
      <td>21.080000</td>
      <td>28.120000</td>
      <td>34.880000</td>
      <td>33.740000</td>
      <td>74.620000</td>
      <td>21.600000</td>
      <td>28.120000</td>
      <td>27.200000</td>
      <td>48.760000</td>
      <td>22.940000</td>
      <td>25.980000</td>
      <td>32.340000</td>
      <td>28.920000</td>
      <td>31.520000</td>
      <td>21.120000</td>
      <td>35.250000</td>
      <td>26.020000</td>
      <td>42.440000</td>
      <td>32.200000</td>
      <td>22.480000</td>
      <td>89.460000</td>
      <td>74.580000</td>
      <td>78.380000</td>
      <td>68.080000</td>
      <td>26.700000</td>
      <td>10.020000</td>
      <td>8.440000</td>
      <td>2.500000</td>
      <td>66.160000</td>
      <td>41.780000</td>
      <td>48.820000</td>
      <td>9.620000</td>
      <td>46.260000</td>
      <td>37.440000</td>
      <td>13.420000</td>
      <td>3.260000</td>
      <td>25.960000</td>
      <td>96.360000</td>
      <td>69.020000</td>
      <td>94.960000</td>
      <td>15.080000</td>
      <td>14.560000</td>
      <td>11.300000</td>
      <td>13.320000</td>
      <td>13.720000</td>
      <td>39.600000</td>
      <td>29.340000</td>
      <td>0.440000</td>
      <td>2.000000</td>
      <td>31.340000</td>
      <td>31.225000</td>
      <td>24.820000</td>
      <td>38.000000</td>
      <td>38.680000</td>
      <td>45.140000</td>
      <td>30.800000</td>
      <td>65.120000</td>
      <td>8.520000</td>
      <td>11.840000</td>
      <td>20.720000</td>
      <td>38.180000</td>
      <td>33.220000</td>
      <td>30.220000</td>
      <td>26.960000</td>
      <td>26.880000</td>
      <td>31.120000</td>
      <td>9.020000</td>
      <td>29.420000</td>
      <td>52.520000</td>
      <td>65.540000</td>
      <td>37.520000</td>
      <td>8.960000</td>
      <td>94.300000</td>
      <td>73.560000</td>
      <td>37.560000</td>
      <td>49.200000</td>
      <td>43.080000</td>
      <td>17.780000</td>
      <td>20.000000</td>
      <td>17.360000</td>
      <td>37.480000</td>
      <td>48.960000</td>
      <td>55.240000</td>
      <td>37.060000</td>
      <td>94.600000</td>
      <td>55.420000</td>
      <td>35.520000</td>
      <td>24.700000</td>
      <td>0.960000</td>
      <td>42.440000</td>
      <td>0.64000</td>
      <td>8.180000</td>
      <td>2.300000</td>
      <td>9.400000</td>
      <td>4.640000</td>
      <td>7.160000</td>
      <td>99.860000</td>
      <td>8.220000</td>
      <td>1.700000</td>
      <td>60.620000</td>
      <td>49.280000</td>
      <td>37.720000</td>
      <td>35.860000</td>
      <td>33.020000</td>
      <td>32.300000</td>
      <td>17.800000</td>
      <td>7.220000</td>
      <td>77.800000</td>
      <td>63.580000</td>
      <td>10.620000</td>
      <td>18.700000</td>
      <td>34.320000</td>
      <td>52.540000</td>
      <td>54.340000</td>
      <td>47.075000</td>
      <td>64.580000</td>
      <td>7.300000</td>
      <td>23.88000</td>
      <td>36.760000</td>
      <td>42.360000</td>
      <td>95.840000</td>
      <td>61.880000</td>
      <td>32.980000</td>
      <td>30.500000</td>
      <td>14.180000</td>
      <td>49.460000</td>
      <td>53.440000</td>
      <td>37.16000</td>
      <td>26.000000</td>
      <td>27.860000</td>
      <td>12.480000</td>
      <td>10.640000</td>
      <td>35.540000</td>
      <td>10.180000</td>
      <td>23.625000</td>
      <td>40.766667</td>
      <td>65.475000</td>
      <td>54.640000</td>
      <td>49.080000</td>
      <td>35.340000</td>
      <td>55.060000</td>
      <td>28.900000</td>
      <td>34.640000</td>
      <td>30.320000</td>
      <td>46.460000</td>
      <td>77.575000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfl.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip</th>
      <th>zcta</th>
      <th>perc_pre1950_housing__CLPPP</th>
      <th>children_under6__CLPPP</th>
      <th>children_tested__CLPPP</th>
      <th>perc_tested__CLPPP</th>
      <th>bll_lt5__CLPPP</th>
      <th>bll_5to9__CLPPP</th>
      <th>capillary_ge10__CLPPP</th>
      <th>venous_10to19__CLPPP</th>
      <th>venous_20to44__CLPPP</th>
      <th>venous_ge45__CLPPP</th>
      <th>blltot_ge5__CLPPP</th>
      <th>blltot_ge10__CLPPP</th>
      <th>perc_bll_ge5__CLPPP</th>
      <th>perc_bll_ge10__CLPPP</th>
      <th>HC03_EST_VC01__S1701</th>
      <th>HC03_EST_VC03__S1701</th>
      <th>HC03_EST_VC04__S1701</th>
      <th>HC03_EST_VC05__S1701</th>
      <th>HC03_EST_VC06__S1701</th>
      <th>HC03_EST_VC09__S1701</th>
      <th>HC03_EST_VC10__S1701</th>
      <th>HC03_EST_VC13__S1701</th>
      <th>HC03_EST_VC14__S1701</th>
      <th>HC03_EST_VC20__S1701</th>
      <th>HC03_EST_VC22__S1701</th>
      <th>HC03_EST_VC23__S1701</th>
      <th>HC03_EST_VC26__S1701</th>
      <th>HC03_EST_VC27__S1701</th>
      <th>HC03_EST_VC28__S1701</th>
      <th>HC03_EST_VC29__S1701</th>
      <th>HC03_EST_VC30__S1701</th>
      <th>HC03_EST_VC33__S1701</th>
      <th>HC03_EST_VC34__S1701</th>
      <th>HC03_EST_VC35__S1701</th>
      <th>HC03_EST_VC36__S1701</th>
      <th>HC03_EST_VC37__S1701</th>
      <th>HC03_EST_VC38__S1701</th>
      <th>HC03_EST_VC39__S1701</th>
      <th>HC03_EST_VC42__S1701</th>
      <th>HC03_EST_VC43__S1701</th>
      <th>HC03_EST_VC44__S1701</th>
      <th>HC03_EST_VC45__S1701</th>
      <th>HC03_EST_VC54__S1701</th>
      <th>HC03_EST_VC55__S1701</th>
      <th>HC03_EST_VC56__S1701</th>
      <th>HC03_EST_VC60__S1701</th>
      <th>HC03_EST_VC61__S1701</th>
      <th>HC03_EST_VC62__S1701</th>
      <th>HC02_EST_VC01__S1702</th>
      <th>HC04_EST_VC01__S1702</th>
      <th>HC06_EST_VC01__S1702</th>
      <th>HC02_EST_VC02__S1702</th>
      <th>HC04_EST_VC02__S1702</th>
      <th>HC06_EST_VC02__S1702</th>
      <th>HC02_EST_VC06__S1702</th>
      <th>HC04_EST_VC06__S1702</th>
      <th>HC06_EST_VC06__S1702</th>
      <th>HC02_EST_VC07__S1702</th>
      <th>HC04_EST_VC07__S1702</th>
      <th>HC06_EST_VC07__S1702</th>
      <th>HC02_EST_VC16__S1702</th>
      <th>HC04_EST_VC16__S1702</th>
      <th>HC06_EST_VC16__S1702</th>
      <th>HC02_EST_VC18__S1702</th>
      <th>HC04_EST_VC18__S1702</th>
      <th>HC06_EST_VC18__S1702</th>
      <th>HC02_EST_VC19__S1702</th>
      <th>HC04_EST_VC19__S1702</th>
      <th>HC06_EST_VC19__S1702</th>
      <th>HC02_EST_VC21__S1702</th>
      <th>HC04_EST_VC21__S1702</th>
      <th>HC06_EST_VC21__S1702</th>
      <th>HC02_EST_VC23__S1702</th>
      <th>HC04_EST_VC23__S1702</th>
      <th>HC06_EST_VC23__S1702</th>
      <th>HC02_EST_VC24__S1702</th>
      <th>HC04_EST_VC24__S1702</th>
      <th>HC06_EST_VC24__S1702</th>
      <th>HC02_EST_VC27__S1702</th>
      <th>HC04_EST_VC27__S1702</th>
      <th>HC02_EST_VC28__S1702</th>
      <th>HC04_EST_VC28__S1702</th>
      <th>HC06_EST_VC28__S1702</th>
      <th>HC02_EST_VC29__S1702</th>
      <th>HC04_EST_VC29__S1702</th>
      <th>HC06_EST_VC29__S1702</th>
      <th>HC02_EST_VC30__S1702</th>
      <th>HC04_EST_VC30__S1702</th>
      <th>HC02_EST_VC33__S1702</th>
      <th>HC04_EST_VC33__S1702</th>
      <th>HC06_EST_VC33__S1702</th>
      <th>HC02_EST_VC34__S1702</th>
      <th>HC04_EST_VC34__S1702</th>
      <th>HC06_EST_VC34__S1702</th>
      <th>HC02_EST_VC35__S1702</th>
      <th>HC04_EST_VC35__S1702</th>
      <th>HC02_EST_VC39__S1702</th>
      <th>HC04_EST_VC39__S1702</th>
      <th>HC06_EST_VC39__S1702</th>
      <th>HC02_EST_VC40__S1702</th>
      <th>HC04_EST_VC40__S1702</th>
      <th>HC06_EST_VC40__S1702</th>
      <th>HC02_EST_VC41__S1702</th>
      <th>HC04_EST_VC41__S1702</th>
      <th>HC02_EST_VC45__S1702</th>
      <th>HC04_EST_VC45__S1702</th>
      <th>HC06_EST_VC45__S1702</th>
      <th>HC02_EST_VC46__S1702</th>
      <th>HC04_EST_VC46__S1702</th>
      <th>HC06_EST_VC46__S1702</th>
      <th>HC02_EST_VC47__S1702</th>
      <th>HC04_EST_VC47__S1702</th>
      <th>HC06_EST_VC47__S1702</th>
      <th>HC02_EST_VC48__S1702</th>
      <th>HC04_EST_VC48__S1702</th>
      <th>HC02_EST_VC01__S1401</th>
      <th>HC03_EST_VC01__S1401</th>
      <th>HC02_EST_VC02__S1401</th>
      <th>HC03_EST_VC02__S1401</th>
      <th>HC02_EST_VC03__S1401</th>
      <th>HC03_EST_VC03__S1401</th>
      <th>HC02_EST_VC04__S1401</th>
      <th>HC03_EST_VC04__S1401</th>
      <th>HC02_EST_VC05__S1401</th>
      <th>HC03_EST_VC05__S1401</th>
      <th>HC02_EST_VC06__S1401</th>
      <th>HC03_EST_VC06__S1401</th>
      <th>HC02_EST_VC07__S1401</th>
      <th>HC03_EST_VC07__S1401</th>
      <th>HC02_EST_VC08__S1401</th>
      <th>HC03_EST_VC08__S1401</th>
      <th>HC02_EST_VC09__S1401</th>
      <th>HC03_EST_VC09__S1401</th>
      <th>HC01_EST_VC12__S1401</th>
      <th>HC02_EST_VC12__S1401</th>
      <th>HC03_EST_VC12__S1401</th>
      <th>HC01_EST_VC13__S1401</th>
      <th>HC02_EST_VC13__S1401</th>
      <th>HC03_EST_VC13__S1401</th>
      <th>HC01_EST_VC14__S1401</th>
      <th>HC02_EST_VC14__S1401</th>
      <th>HC03_EST_VC14__S1401</th>
      <th>HC01_EST_VC15__S1401</th>
      <th>HC02_EST_VC15__S1401</th>
      <th>HC03_EST_VC15__S1401</th>
      <th>HC01_EST_VC16__S1401</th>
      <th>HC02_EST_VC16__S1401</th>
      <th>HC03_EST_VC16__S1401</th>
      <th>HC01_EST_VC17__S1401</th>
      <th>HC02_EST_VC17__S1401</th>
      <th>HC03_EST_VC17__S1401</th>
      <th>HC01_EST_VC18__S1401</th>
      <th>HC02_EST_VC18__S1401</th>
      <th>HC03_EST_VC18__S1401</th>
      <th>HC01_EST_VC19__S1401</th>
      <th>HC02_EST_VC19__S1401</th>
      <th>HC03_EST_VC19__S1401</th>
      <th>HC02_EST_VC22__S1401</th>
      <th>HC03_EST_VC22__S1401</th>
      <th>HC02_EST_VC24__S1401</th>
      <th>HC03_EST_VC24__S1401</th>
      <th>HC02_EST_VC26__S1401</th>
      <th>HC03_EST_VC26__S1401</th>
      <th>HC02_EST_VC29__S1401</th>
      <th>HC03_EST_VC29__S1401</th>
      <th>HC02_EST_VC31__S1401</th>
      <th>HC03_EST_VC31__S1401</th>
      <th>HC02_EST_VC33__S1401</th>
      <th>HC03_EST_VC33__S1401</th>
      <th>HC01_EST_VC16__S1501</th>
      <th>HC02_EST_VC16__S1501</th>
      <th>HC03_EST_VC16__S1501</th>
      <th>HC01_EST_VC17__S1501</th>
      <th>HC02_EST_VC17__S1501</th>
      <th>HC03_EST_VC17__S1501</th>
      <th>HC02_EST_VC01__S1601</th>
      <th>HC03_EST_VC01__S1601</th>
      <th>HC02_EST_VC03__S1601</th>
      <th>HC03_EST_VC03__S1601</th>
      <th>HC02_EST_VC04__S1601</th>
      <th>HC03_EST_VC04__S1601</th>
      <th>HC02_EST_VC05__S1601</th>
      <th>HC03_EST_VC05__S1601</th>
      <th>HC02_EST_VC10__S1601</th>
      <th>HC03_EST_VC10__S1601</th>
      <th>HC02_EST_VC12__S1601</th>
      <th>HC03_EST_VC12__S1601</th>
      <th>HC02_EST_VC14__S1601</th>
      <th>HC03_EST_VC14__S1601</th>
      <th>HC02_EST_VC16__S1601</th>
      <th>HC03_EST_VC16__S1601</th>
      <th>HC02_EST_VC28__S1601</th>
      <th>HC03_EST_VC28__S1601</th>
      <th>HC02_EST_VC30__S1601</th>
      <th>HC03_EST_VC30__S1601</th>
      <th>HC02_EST_VC31__S1601</th>
      <th>HC03_EST_VC31__S1601</th>
      <th>HC02_EST_VC32__S1601</th>
      <th>HC03_EST_VC32__S1601</th>
      <th>HC03_EST_VC01__S2701</th>
      <th>HC05_EST_VC01__S2701</th>
      <th>HC03_EST_VC04__S2701</th>
      <th>HC03_EST_VC05__S2701</th>
      <th>HC03_EST_VC06__S2701</th>
      <th>HC03_EST_VC08__S2701</th>
      <th>HC03_EST_VC11__S2701</th>
      <th>HC03_EST_VC12__S2701</th>
      <th>HC03_EST_VC15__S2701</th>
      <th>HC03_EST_VC16__S2701</th>
      <th>HC03_EST_VC22__S2701</th>
      <th>HC03_EST_VC24__S2701</th>
      <th>HC03_EST_VC25__S2701</th>
      <th>HC03_EST_VC28__S2701</th>
      <th>HC03_EST_VC29__S2701</th>
      <th>HC03_EST_VC30__S2701</th>
      <th>HC03_EST_VC31__S2701</th>
      <th>HC03_EST_VC34__S2701</th>
      <th>HC03_EST_VC35__S2701</th>
      <th>HC03_EST_VC36__S2701</th>
      <th>HC03_EST_VC37__S2701</th>
      <th>HC03_EST_VC38__S2701</th>
      <th>HC03_EST_VC41__S2701</th>
      <th>HC03_EST_VC42__S2701</th>
      <th>HC03_EST_VC43__S2701</th>
      <th>HC03_EST_VC44__S2701</th>
      <th>HC03_EST_VC45__S2701</th>
      <th>HC03_EST_VC48__S2701</th>
      <th>HC03_EST_VC49__S2701</th>
      <th>HC03_EST_VC50__S2701</th>
      <th>HC03_EST_VC51__S2701</th>
      <th>HC03_EST_VC54__S2701</th>
      <th>HC03_EST_VC55__S2701</th>
      <th>HC03_EST_VC56__S2701</th>
      <th>HC03_EST_VC57__S2701</th>
      <th>HC03_EST_VC58__S2701</th>
      <th>HC03_EST_VC59__S2701</th>
      <th>HC03_EST_VC62__S2701</th>
      <th>HC03_EST_VC63__S2701</th>
      <th>HC03_EST_VC64__S2701</th>
      <th>HC03_EST_VC65__S2701</th>
      <th>HC05_EST_VC68__S2701</th>
      <th>HC05_EST_VC69__S2701</th>
      <th>HC05_EST_VC70__S2701</th>
      <th>HC05_EST_VC71__S2701</th>
      <th>HC05_EST_VC72__S2701</th>
      <th>HC05_EST_VC73__S2701</th>
      <th>HC05_EST_VC74__S2701</th>
      <th>HC05_EST_VC75__S2701</th>
      <th>HC05_EST_VC76__S2701</th>
      <th>HC05_EST_VC77__S2701</th>
      <th>HC05_EST_VC78__S2701</th>
      <th>HC05_EST_VC79__S2701</th>
      <th>HC05_EST_VC80__S2701</th>
      <th>HC05_EST_VC81__S2701</th>
      <th>HC05_EST_VC82__S2701</th>
      <th>HC05_EST_VC83__S2701</th>
      <th>HC05_EST_VC84__S2701</th>
      <th>HC03_VC04__DP04</th>
      <th>HC03_VC05__DP04</th>
      <th>HC03_VC14__DP04</th>
      <th>HC03_VC15__DP04</th>
      <th>HC03_VC16__DP04</th>
      <th>HC03_VC17__DP04</th>
      <th>HC03_VC18__DP04</th>
      <th>HC03_VC19__DP04</th>
      <th>HC03_VC20__DP04</th>
      <th>HC03_VC21__DP04</th>
      <th>HC03_VC22__DP04</th>
      <th>HC03_VC27__DP04</th>
      <th>HC03_VC28__DP04</th>
      <th>HC03_VC29__DP04</th>
      <th>HC03_VC30__DP04</th>
      <th>HC03_VC31__DP04</th>
      <th>HC03_VC32__DP04</th>
      <th>HC03_VC33__DP04</th>
      <th>HC03_VC34__DP04</th>
      <th>HC03_VC35__DP04</th>
      <th>HC03_VC40__DP04</th>
      <th>HC03_VC41__DP04</th>
      <th>HC03_VC42__DP04</th>
      <th>HC03_VC43__DP04</th>
      <th>HC03_VC44__DP04</th>
      <th>HC03_VC45__DP04</th>
      <th>HC03_VC46__DP04</th>
      <th>HC03_VC47__DP04</th>
      <th>HC03_VC48__DP04</th>
      <th>HC03_VC54__DP04</th>
      <th>HC03_VC55__DP04</th>
      <th>HC03_VC56__DP04</th>
      <th>HC03_VC57__DP04</th>
      <th>HC03_VC58__DP04</th>
      <th>HC03_VC59__DP04</th>
      <th>HC03_VC64__DP04</th>
      <th>HC03_VC65__DP04</th>
      <th>HC03_VC74__DP04</th>
      <th>HC03_VC75__DP04</th>
      <th>HC03_VC76__DP04</th>
      <th>HC03_VC77__DP04</th>
      <th>HC03_VC78__DP04</th>
      <th>HC03_VC79__DP04</th>
      <th>HC03_VC84__DP04</th>
      <th>HC03_VC85__DP04</th>
      <th>HC03_VC86__DP04</th>
      <th>HC03_VC87__DP04</th>
      <th>HC03_VC92__DP04</th>
      <th>HC03_VC93__DP04</th>
      <th>HC03_VC94__DP04</th>
      <th>HC03_VC95__DP04</th>
      <th>HC03_VC96__DP04</th>
      <th>HC03_VC97__DP04</th>
      <th>HC03_VC98__DP04</th>
      <th>HC03_VC99__DP04</th>
      <th>HC03_VC100__DP04</th>
      <th>HC03_VC105__DP04</th>
      <th>HC03_VC106__DP04</th>
      <th>HC03_VC107__DP04</th>
      <th>HC03_VC112__DP04</th>
      <th>HC03_VC113__DP04</th>
      <th>HC03_VC114__DP04</th>
      <th>HC03_VC119__DP04</th>
      <th>HC03_VC120__DP04</th>
      <th>HC03_VC121__DP04</th>
      <th>HC03_VC122__DP04</th>
      <th>HC03_VC123__DP04</th>
      <th>HC03_VC124__DP04</th>
      <th>HC03_VC125__DP04</th>
      <th>HC03_VC126__DP04</th>
      <th>HC03_VC132__DP04</th>
      <th>HC03_VC133__DP04</th>
      <th>HC03_VC138__DP04</th>
      <th>HC03_VC139__DP04</th>
      <th>HC03_VC140__DP04</th>
      <th>HC03_VC141__DP04</th>
      <th>HC03_VC142__DP04</th>
      <th>HC03_VC143__DP04</th>
      <th>HC03_VC144__DP04</th>
      <th>HC03_VC148__DP04</th>
      <th>HC03_VC149__DP04</th>
      <th>HC03_VC150__DP04</th>
      <th>HC03_VC151__DP04</th>
      <th>HC03_VC152__DP04</th>
      <th>HC03_VC158__DP04</th>
      <th>HC03_VC159__DP04</th>
      <th>HC03_VC160__DP04</th>
      <th>HC03_VC161__DP04</th>
      <th>HC03_VC162__DP04</th>
      <th>HC03_VC168__DP04</th>
      <th>HC03_VC169__DP04</th>
      <th>HC03_VC170__DP04</th>
      <th>HC03_VC171__DP04</th>
      <th>HC03_VC172__DP04</th>
      <th>HC03_VC173__DP04</th>
      <th>HC03_VC174__DP04</th>
      <th>HC03_VC182__DP04</th>
      <th>HC03_VC183__DP04</th>
      <th>HC03_VC184__DP04</th>
      <th>HC03_VC185__DP04</th>
      <th>HC03_VC186__DP04</th>
      <th>HC03_VC187__DP04</th>
      <th>HC03_VC188__DP04</th>
      <th>HC03_VC197__DP04</th>
      <th>HC03_VC198__DP04</th>
      <th>HC03_VC199__DP04</th>
      <th>HC03_VC200__DP04</th>
      <th>HC03_VC201__DP04</th>
      <th>HC03_VC202__DP04</th>
      <th>perc_pre1950_housing__CLPPP_close</th>
      <th>HC03_EST_VC01__S1701_close</th>
      <th>HC03_EST_VC03__S1701_close</th>
      <th>HC03_EST_VC04__S1701_close</th>
      <th>HC03_EST_VC05__S1701_close</th>
      <th>HC03_EST_VC06__S1701_close</th>
      <th>HC03_EST_VC09__S1701_close</th>
      <th>HC03_EST_VC10__S1701_close</th>
      <th>HC03_EST_VC13__S1701_close</th>
      <th>HC03_EST_VC14__S1701_close</th>
      <th>HC03_EST_VC20__S1701_close</th>
      <th>HC03_EST_VC22__S1701_close</th>
      <th>HC03_EST_VC23__S1701_close</th>
      <th>HC03_EST_VC26__S1701_close</th>
      <th>HC03_EST_VC27__S1701_close</th>
      <th>HC03_EST_VC28__S1701_close</th>
      <th>HC03_EST_VC29__S1701_close</th>
      <th>HC03_EST_VC30__S1701_close</th>
      <th>HC03_EST_VC33__S1701_close</th>
      <th>HC03_EST_VC34__S1701_close</th>
      <th>HC03_EST_VC35__S1701_close</th>
      <th>HC03_EST_VC36__S1701_close</th>
      <th>HC03_EST_VC37__S1701_close</th>
      <th>HC03_EST_VC38__S1701_close</th>
      <th>HC03_EST_VC39__S1701_close</th>
      <th>HC03_EST_VC42__S1701_close</th>
      <th>HC03_EST_VC43__S1701_close</th>
      <th>HC03_EST_VC44__S1701_close</th>
      <th>HC03_EST_VC45__S1701_close</th>
      <th>HC03_EST_VC54__S1701_close</th>
      <th>HC03_EST_VC55__S1701_close</th>
      <th>HC03_EST_VC56__S1701_close</th>
      <th>HC03_EST_VC60__S1701_close</th>
      <th>HC03_EST_VC61__S1701_close</th>
      <th>HC03_EST_VC62__S1701_close</th>
      <th>HC02_EST_VC01__S1702_close</th>
      <th>HC04_EST_VC01__S1702_close</th>
      <th>HC06_EST_VC01__S1702_close</th>
      <th>HC02_EST_VC02__S1702_close</th>
      <th>HC04_EST_VC02__S1702_close</th>
      <th>HC06_EST_VC02__S1702_close</th>
      <th>HC02_EST_VC06__S1702_close</th>
      <th>HC04_EST_VC06__S1702_close</th>
      <th>HC06_EST_VC06__S1702_close</th>
      <th>HC02_EST_VC07__S1702_close</th>
      <th>HC04_EST_VC07__S1702_close</th>
      <th>HC06_EST_VC07__S1702_close</th>
      <th>HC02_EST_VC16__S1702_close</th>
      <th>HC04_EST_VC16__S1702_close</th>
      <th>HC06_EST_VC16__S1702_close</th>
      <th>HC02_EST_VC18__S1702_close</th>
      <th>HC04_EST_VC18__S1702_close</th>
      <th>HC06_EST_VC18__S1702_close</th>
      <th>HC02_EST_VC19__S1702_close</th>
      <th>HC04_EST_VC19__S1702_close</th>
      <th>HC06_EST_VC19__S1702_close</th>
      <th>HC02_EST_VC21__S1702_close</th>
      <th>HC04_EST_VC21__S1702_close</th>
      <th>HC06_EST_VC21__S1702_close</th>
      <th>HC02_EST_VC23__S1702_close</th>
      <th>HC04_EST_VC23__S1702_close</th>
      <th>HC06_EST_VC23__S1702_close</th>
      <th>HC02_EST_VC24__S1702_close</th>
      <th>HC04_EST_VC24__S1702_close</th>
      <th>HC06_EST_VC24__S1702_close</th>
      <th>HC02_EST_VC27__S1702_close</th>
      <th>HC04_EST_VC27__S1702_close</th>
      <th>HC02_EST_VC28__S1702_close</th>
      <th>HC04_EST_VC28__S1702_close</th>
      <th>HC06_EST_VC28__S1702_close</th>
      <th>HC02_EST_VC29__S1702_close</th>
      <th>HC04_EST_VC29__S1702_close</th>
      <th>HC06_EST_VC29__S1702_close</th>
      <th>HC02_EST_VC30__S1702_close</th>
      <th>HC04_EST_VC30__S1702_close</th>
      <th>HC02_EST_VC33__S1702_close</th>
      <th>HC04_EST_VC33__S1702_close</th>
      <th>HC06_EST_VC33__S1702_close</th>
      <th>HC02_EST_VC34__S1702_close</th>
      <th>HC04_EST_VC34__S1702_close</th>
      <th>HC06_EST_VC34__S1702_close</th>
      <th>HC02_EST_VC35__S1702_close</th>
      <th>HC04_EST_VC35__S1702_close</th>
      <th>HC02_EST_VC39__S1702_close</th>
      <th>HC04_EST_VC39__S1702_close</th>
      <th>HC06_EST_VC39__S1702_close</th>
      <th>HC02_EST_VC40__S1702_close</th>
      <th>HC04_EST_VC40__S1702_close</th>
      <th>HC06_EST_VC40__S1702_close</th>
      <th>HC02_EST_VC41__S1702_close</th>
      <th>HC04_EST_VC41__S1702_close</th>
      <th>HC02_EST_VC45__S1702_close</th>
      <th>HC04_EST_VC45__S1702_close</th>
      <th>HC06_EST_VC45__S1702_close</th>
      <th>HC02_EST_VC46__S1702_close</th>
      <th>HC04_EST_VC46__S1702_close</th>
      <th>HC06_EST_VC46__S1702_close</th>
      <th>HC02_EST_VC47__S1702_close</th>
      <th>HC04_EST_VC47__S1702_close</th>
      <th>HC06_EST_VC47__S1702_close</th>
      <th>HC02_EST_VC48__S1702_close</th>
      <th>HC04_EST_VC48__S1702_close</th>
      <th>HC02_EST_VC01__S1401_close</th>
      <th>HC03_EST_VC01__S1401_close</th>
      <th>HC02_EST_VC02__S1401_close</th>
      <th>HC03_EST_VC02__S1401_close</th>
      <th>HC02_EST_VC03__S1401_close</th>
      <th>HC03_EST_VC03__S1401_close</th>
      <th>HC02_EST_VC04__S1401_close</th>
      <th>HC03_EST_VC04__S1401_close</th>
      <th>HC02_EST_VC05__S1401_close</th>
      <th>HC03_EST_VC05__S1401_close</th>
      <th>HC02_EST_VC06__S1401_close</th>
      <th>HC03_EST_VC06__S1401_close</th>
      <th>HC02_EST_VC07__S1401_close</th>
      <th>HC03_EST_VC07__S1401_close</th>
      <th>HC02_EST_VC08__S1401_close</th>
      <th>HC03_EST_VC08__S1401_close</th>
      <th>HC02_EST_VC09__S1401_close</th>
      <th>HC03_EST_VC09__S1401_close</th>
      <th>HC01_EST_VC12__S1401_close</th>
      <th>HC02_EST_VC12__S1401_close</th>
      <th>HC03_EST_VC12__S1401_close</th>
      <th>HC01_EST_VC13__S1401_close</th>
      <th>HC02_EST_VC13__S1401_close</th>
      <th>HC03_EST_VC13__S1401_close</th>
      <th>HC01_EST_VC14__S1401_close</th>
      <th>HC02_EST_VC14__S1401_close</th>
      <th>HC03_EST_VC14__S1401_close</th>
      <th>HC01_EST_VC15__S1401_close</th>
      <th>HC02_EST_VC15__S1401_close</th>
      <th>HC03_EST_VC15__S1401_close</th>
      <th>HC01_EST_VC16__S1401_close</th>
      <th>HC02_EST_VC16__S1401_close</th>
      <th>HC03_EST_VC16__S1401_close</th>
      <th>HC01_EST_VC17__S1401_close</th>
      <th>HC02_EST_VC17__S1401_close</th>
      <th>HC03_EST_VC17__S1401_close</th>
      <th>HC01_EST_VC18__S1401_close</th>
      <th>HC02_EST_VC18__S1401_close</th>
      <th>HC03_EST_VC18__S1401_close</th>
      <th>HC01_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC19__S1401_close</th>
      <th>HC03_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC22__S1401_close</th>
      <th>HC03_EST_VC22__S1401_close</th>
      <th>HC02_EST_VC24__S1401_close</th>
      <th>HC03_EST_VC24__S1401_close</th>
      <th>HC02_EST_VC26__S1401_close</th>
      <th>HC03_EST_VC26__S1401_close</th>
      <th>HC02_EST_VC29__S1401_close</th>
      <th>HC03_EST_VC29__S1401_close</th>
      <th>HC02_EST_VC31__S1401_close</th>
      <th>HC03_EST_VC31__S1401_close</th>
      <th>HC02_EST_VC33__S1401_close</th>
      <th>HC03_EST_VC33__S1401_close</th>
      <th>HC01_EST_VC16__S1501_close</th>
      <th>HC02_EST_VC16__S1501_close</th>
      <th>HC03_EST_VC16__S1501_close</th>
      <th>HC01_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC17__S1501_close</th>
      <th>HC03_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC01__S1601_close</th>
      <th>HC03_EST_VC01__S1601_close</th>
      <th>HC02_EST_VC03__S1601_close</th>
      <th>HC03_EST_VC03__S1601_close</th>
      <th>HC02_EST_VC04__S1601_close</th>
      <th>HC03_EST_VC04__S1601_close</th>
      <th>HC02_EST_VC05__S1601_close</th>
      <th>HC03_EST_VC05__S1601_close</th>
      <th>HC02_EST_VC10__S1601_close</th>
      <th>HC03_EST_VC10__S1601_close</th>
      <th>HC02_EST_VC12__S1601_close</th>
      <th>HC03_EST_VC12__S1601_close</th>
      <th>HC02_EST_VC14__S1601_close</th>
      <th>HC03_EST_VC14__S1601_close</th>
      <th>HC02_EST_VC16__S1601_close</th>
      <th>HC03_EST_VC16__S1601_close</th>
      <th>HC02_EST_VC28__S1601_close</th>
      <th>HC03_EST_VC28__S1601_close</th>
      <th>HC02_EST_VC30__S1601_close</th>
      <th>HC03_EST_VC30__S1601_close</th>
      <th>HC02_EST_VC31__S1601_close</th>
      <th>HC03_EST_VC31__S1601_close</th>
      <th>HC02_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC01__S2701_close</th>
      <th>HC05_EST_VC01__S2701_close</th>
      <th>HC03_EST_VC04__S2701_close</th>
      <th>HC03_EST_VC05__S2701_close</th>
      <th>HC03_EST_VC06__S2701_close</th>
      <th>HC03_EST_VC08__S2701_close</th>
      <th>HC03_EST_VC11__S2701_close</th>
      <th>HC03_EST_VC12__S2701_close</th>
      <th>HC03_EST_VC15__S2701_close</th>
      <th>HC03_EST_VC16__S2701_close</th>
      <th>HC03_EST_VC22__S2701_close</th>
      <th>HC03_EST_VC24__S2701_close</th>
      <th>HC03_EST_VC25__S2701_close</th>
      <th>HC03_EST_VC28__S2701_close</th>
      <th>HC03_EST_VC29__S2701_close</th>
      <th>HC03_EST_VC30__S2701_close</th>
      <th>HC03_EST_VC31__S2701_close</th>
      <th>HC03_EST_VC34__S2701_close</th>
      <th>HC03_EST_VC35__S2701_close</th>
      <th>HC03_EST_VC36__S2701_close</th>
      <th>HC03_EST_VC37__S2701_close</th>
      <th>HC03_EST_VC38__S2701_close</th>
      <th>HC03_EST_VC41__S2701_close</th>
      <th>HC03_EST_VC42__S2701_close</th>
      <th>HC03_EST_VC43__S2701_close</th>
      <th>HC03_EST_VC44__S2701_close</th>
      <th>HC03_EST_VC45__S2701_close</th>
      <th>HC03_EST_VC48__S2701_close</th>
      <th>HC03_EST_VC49__S2701_close</th>
      <th>HC03_EST_VC50__S2701_close</th>
      <th>HC03_EST_VC51__S2701_close</th>
      <th>HC03_EST_VC54__S2701_close</th>
      <th>HC03_EST_VC55__S2701_close</th>
      <th>HC03_EST_VC56__S2701_close</th>
      <th>HC03_EST_VC57__S2701_close</th>
      <th>HC03_EST_VC58__S2701_close</th>
      <th>HC03_EST_VC59__S2701_close</th>
      <th>HC03_EST_VC62__S2701_close</th>
      <th>HC03_EST_VC63__S2701_close</th>
      <th>HC03_EST_VC64__S2701_close</th>
      <th>HC03_EST_VC65__S2701_close</th>
      <th>HC05_EST_VC68__S2701_close</th>
      <th>HC05_EST_VC69__S2701_close</th>
      <th>HC05_EST_VC70__S2701_close</th>
      <th>HC05_EST_VC71__S2701_close</th>
      <th>HC05_EST_VC72__S2701_close</th>
      <th>HC05_EST_VC73__S2701_close</th>
      <th>HC05_EST_VC74__S2701_close</th>
      <th>HC05_EST_VC75__S2701_close</th>
      <th>HC05_EST_VC76__S2701_close</th>
      <th>HC05_EST_VC77__S2701_close</th>
      <th>HC05_EST_VC78__S2701_close</th>
      <th>HC05_EST_VC79__S2701_close</th>
      <th>HC05_EST_VC80__S2701_close</th>
      <th>HC05_EST_VC81__S2701_close</th>
      <th>HC05_EST_VC82__S2701_close</th>
      <th>HC05_EST_VC83__S2701_close</th>
      <th>HC05_EST_VC84__S2701_close</th>
      <th>HC03_VC04__DP04_close</th>
      <th>HC03_VC05__DP04_close</th>
      <th>HC03_VC14__DP04_close</th>
      <th>HC03_VC15__DP04_close</th>
      <th>HC03_VC16__DP04_close</th>
      <th>HC03_VC17__DP04_close</th>
      <th>HC03_VC18__DP04_close</th>
      <th>HC03_VC19__DP04_close</th>
      <th>HC03_VC20__DP04_close</th>
      <th>HC03_VC21__DP04_close</th>
      <th>HC03_VC22__DP04_close</th>
      <th>HC03_VC27__DP04_close</th>
      <th>HC03_VC28__DP04_close</th>
      <th>HC03_VC29__DP04_close</th>
      <th>HC03_VC30__DP04_close</th>
      <th>HC03_VC31__DP04_close</th>
      <th>HC03_VC32__DP04_close</th>
      <th>HC03_VC33__DP04_close</th>
      <th>HC03_VC34__DP04_close</th>
      <th>HC03_VC35__DP04_close</th>
      <th>HC03_VC40__DP04_close</th>
      <th>HC03_VC41__DP04_close</th>
      <th>HC03_VC42__DP04_close</th>
      <th>HC03_VC43__DP04_close</th>
      <th>HC03_VC44__DP04_close</th>
      <th>HC03_VC45__DP04_close</th>
      <th>HC03_VC46__DP04_close</th>
      <th>HC03_VC47__DP04_close</th>
      <th>HC03_VC48__DP04_close</th>
      <th>HC03_VC54__DP04_close</th>
      <th>HC03_VC55__DP04_close</th>
      <th>HC03_VC56__DP04_close</th>
      <th>HC03_VC57__DP04_close</th>
      <th>HC03_VC58__DP04_close</th>
      <th>HC03_VC59__DP04_close</th>
      <th>HC03_VC64__DP04_close</th>
      <th>HC03_VC65__DP04_close</th>
      <th>HC03_VC74__DP04_close</th>
      <th>HC03_VC75__DP04_close</th>
      <th>HC03_VC76__DP04_close</th>
      <th>HC03_VC77__DP04_close</th>
      <th>HC03_VC78__DP04_close</th>
      <th>HC03_VC79__DP04_close</th>
      <th>HC03_VC84__DP04_close</th>
      <th>HC03_VC85__DP04_close</th>
      <th>HC03_VC86__DP04_close</th>
      <th>HC03_VC87__DP04_close</th>
      <th>HC03_VC92__DP04_close</th>
      <th>HC03_VC93__DP04_close</th>
      <th>HC03_VC94__DP04_close</th>
      <th>HC03_VC95__DP04_close</th>
      <th>HC03_VC96__DP04_close</th>
      <th>HC03_VC97__DP04_close</th>
      <th>HC03_VC98__DP04_close</th>
      <th>HC03_VC99__DP04_close</th>
      <th>HC03_VC100__DP04_close</th>
      <th>HC03_VC105__DP04_close</th>
      <th>HC03_VC106__DP04_close</th>
      <th>HC03_VC107__DP04_close</th>
      <th>HC03_VC112__DP04_close</th>
      <th>HC03_VC113__DP04_close</th>
      <th>HC03_VC114__DP04_close</th>
      <th>HC03_VC119__DP04_close</th>
      <th>HC03_VC120__DP04_close</th>
      <th>HC03_VC121__DP04_close</th>
      <th>HC03_VC122__DP04_close</th>
      <th>HC03_VC123__DP04_close</th>
      <th>HC03_VC124__DP04_close</th>
      <th>HC03_VC125__DP04_close</th>
      <th>HC03_VC126__DP04_close</th>
      <th>HC03_VC132__DP04_close</th>
      <th>HC03_VC133__DP04_close</th>
      <th>HC03_VC138__DP04_close</th>
      <th>HC03_VC139__DP04_close</th>
      <th>HC03_VC140__DP04_close</th>
      <th>HC03_VC141__DP04_close</th>
      <th>HC03_VC142__DP04_close</th>
      <th>HC03_VC143__DP04_close</th>
      <th>HC03_VC144__DP04_close</th>
      <th>HC03_VC148__DP04_close</th>
      <th>HC03_VC149__DP04_close</th>
      <th>HC03_VC150__DP04_close</th>
      <th>HC03_VC151__DP04_close</th>
      <th>HC03_VC152__DP04_close</th>
      <th>HC03_VC158__DP04_close</th>
      <th>HC03_VC159__DP04_close</th>
      <th>HC03_VC160__DP04_close</th>
      <th>HC03_VC161__DP04_close</th>
      <th>HC03_VC162__DP04_close</th>
      <th>HC03_VC168__DP04_close</th>
      <th>HC03_VC169__DP04_close</th>
      <th>HC03_VC170__DP04_close</th>
      <th>HC03_VC171__DP04_close</th>
      <th>HC03_VC172__DP04_close</th>
      <th>HC03_VC173__DP04_close</th>
      <th>HC03_VC174__DP04_close</th>
      <th>HC03_VC182__DP04_close</th>
      <th>HC03_VC183__DP04_close</th>
      <th>HC03_VC184__DP04_close</th>
      <th>HC03_VC185__DP04_close</th>
      <th>HC03_VC186__DP04_close</th>
      <th>HC03_VC187__DP04_close</th>
      <th>HC03_VC188__DP04_close</th>
      <th>HC03_VC197__DP04_close</th>
      <th>HC03_VC198__DP04_close</th>
      <th>HC03_VC199__DP04_close</th>
      <th>HC03_VC200__DP04_close</th>
      <th>HC03_VC201__DP04_close</th>
      <th>HC03_VC202__DP04_close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48001</td>
      <td>48001</td>
      <td>25.7</td>
      <td>660</td>
      <td>168</td>
      <td>25.5</td>
      <td>165</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.8</td>
      <td>0</td>
      <td>11.8</td>
      <td>14.6</td>
      <td>13.9</td>
      <td>11.7</td>
      <td>9.2</td>
      <td>10.6</td>
      <td>12.9</td>
      <td>11.8</td>
      <td>11.6</td>
      <td>7.7</td>
      <td>36.2</td>
      <td>11.4</td>
      <td>10.2</td>
      <td>15.9</td>
      <td>12.0</td>
      <td>8.6</td>
      <td>5.3</td>
      <td>7.2</td>
      <td>4.7</td>
      <td>3.4</td>
      <td>6.2</td>
      <td>24.8</td>
      <td>22.8</td>
      <td>27.6</td>
      <td>11.0</td>
      <td>1.7</td>
      <td>10.3</td>
      <td>18.1</td>
      <td>21.4</td>
      <td>18.8</td>
      <td>23.9</td>
      <td>1.9</td>
      <td>24.8</td>
      <td>31.8</td>
      <td>7.9</td>
      <td>5.5</td>
      <td>19.8</td>
      <td>9.1</td>
      <td>7.4</td>
      <td>20.8</td>
      <td>8.0</td>
      <td>5.6</td>
      <td>19.8</td>
      <td>7.9</td>
      <td>5.6</td>
      <td>19.1</td>
      <td>7.7</td>
      <td>5.3</td>
      <td>19.3</td>
      <td>3.8</td>
      <td>2.3</td>
      <td>12.0</td>
      <td>2.1</td>
      <td>2.4</td>
      <td>1.6</td>
      <td>4.8</td>
      <td>3.9</td>
      <td>18.2</td>
      <td>11.0</td>
      <td>3.2</td>
      <td>0.0</td>
      <td>5.5</td>
      <td>5.1</td>
      <td>10.5</td>
      <td>12.7</td>
      <td>16.1</td>
      <td>7.5</td>
      <td>6.2</td>
      <td>20</td>
      <td>9.1</td>
      <td>4.7</td>
      <td>25.4</td>
      <td>4.1</td>
      <td>0.0</td>
      <td>7.2</td>
      <td>4.6</td>
      <td>18.6</td>
      <td>4.8</td>
      <td>5.4</td>
      <td>5.8</td>
      <td>34.2</td>
      <td>18.1</td>
      <td>6.6</td>
      <td>3.6</td>
      <td>17.0</td>
      <td>7.7</td>
      <td>6.8</td>
      <td>15.3</td>
      <td>22.3</td>
      <td>14.0</td>
      <td>20.9</td>
      <td>14</td>
      <td>62.5</td>
      <td>4.6</td>
      <td>4.7</td>
      <td>7.6</td>
      <td>3.9</td>
      <td>2.0</td>
      <td>17.7</td>
      <td>0</td>
      <td>0</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>74.5</td>
      <td>25.5</td>
      <td>91.3</td>
      <td>8.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>77.3</td>
      <td>22.7</td>
      <td>93.6</td>
      <td>6.4</td>
      <td>95.3</td>
      <td>4.7</td>
      <td>82.3</td>
      <td>17.7</td>
      <td>42.0</td>
      <td>58.0</td>
      <td>64.6</td>
      <td>72.9</td>
      <td>27.1</td>
      <td>97.5</td>
      <td>80.4</td>
      <td>19.6</td>
      <td>97.8</td>
      <td>94.1</td>
      <td>5.9</td>
      <td>87.5</td>
      <td>94.4</td>
      <td>5.6</td>
      <td>46.8</td>
      <td>100</td>
      <td>0</td>
      <td>25.1</td>
      <td>74.2</td>
      <td>25.8</td>
      <td>8.4</td>
      <td>83.5</td>
      <td>16.5</td>
      <td>2.8</td>
      <td>71.6</td>
      <td>28.4</td>
      <td>76.5</td>
      <td>23.5</td>
      <td>62.1</td>
      <td>37.9</td>
      <td>82.0</td>
      <td>18.0</td>
      <td>78.0</td>
      <td>22.0</td>
      <td>51.8</td>
      <td>48.2</td>
      <td>89.7</td>
      <td>10.3</td>
      <td>88.2</td>
      <td>84.9</td>
      <td>91.4</td>
      <td>16.0</td>
      <td>16.3</td>
      <td>15.7</td>
      <td>98.7</td>
      <td>1.3</td>
      <td>60.2</td>
      <td>39.8</td>
      <td>69.7</td>
      <td>30.3</td>
      <td>63.6</td>
      <td>36.4</td>
      <td>69.7</td>
      <td>30.3</td>
      <td>54.4</td>
      <td>45.6</td>
      <td>63.6</td>
      <td>36.4</td>
      <td>68.9</td>
      <td>31.1</td>
      <td>98.7</td>
      <td>1.3</td>
      <td>59.4</td>
      <td>40.6</td>
      <td>77.6</td>
      <td>22.4</td>
      <td>54.0</td>
      <td>46.0</td>
      <td>11.8</td>
      <td>88.2</td>
      <td>3.3</td>
      <td>17.8</td>
      <td>0.0</td>
      <td>25.9</td>
      <td>12.2</td>
      <td>11.3</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>0.0</td>
      <td>11.6</td>
      <td>32.6</td>
      <td>11.9</td>
      <td>7.3</td>
      <td>0.0</td>
      <td>18.9</td>
      <td>12.8</td>
      <td>22.7</td>
      <td>12.4</td>
      <td>13.2</td>
      <td>5.6</td>
      <td>13.7</td>
      <td>17.6</td>
      <td>13.9</td>
      <td>43.7</td>
      <td>8.4</td>
      <td>13.7</td>
      <td>11.1</td>
      <td>20.6</td>
      <td>11.0</td>
      <td>11.7</td>
      <td>26.9</td>
      <td>10.2</td>
      <td>9.1</td>
      <td>4.9</td>
      <td>7.8</td>
      <td>11.8</td>
      <td>29.3</td>
      <td>11.7</td>
      <td>6.9</td>
      <td>71.8</td>
      <td>52.7</td>
      <td>60.4</td>
      <td>47.3</td>
      <td>14.5</td>
      <td>5.4</td>
      <td>1.0</td>
      <td>0</td>
      <td>34.3</td>
      <td>13.7</td>
      <td>22.7</td>
      <td>3.7</td>
      <td>16.2</td>
      <td>10.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>11.8</td>
      <td>84.5</td>
      <td>15.5</td>
      <td>82.0</td>
      <td>3.5</td>
      <td>1.2</td>
      <td>1.4</td>
      <td>3.8</td>
      <td>0.6</td>
      <td>1.4</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.2</td>
      <td>7.4</td>
      <td>14.0</td>
      <td>9.4</td>
      <td>17.5</td>
      <td>12.0</td>
      <td>14.8</td>
      <td>9.5</td>
      <td>14.9</td>
      <td>0.5</td>
      <td>1.4</td>
      <td>5.2</td>
      <td>12.9</td>
      <td>23.1</td>
      <td>23.8</td>
      <td>13.8</td>
      <td>11.3</td>
      <td>7.9</td>
      <td>0.7</td>
      <td>8.7</td>
      <td>25.6</td>
      <td>49.4</td>
      <td>13.6</td>
      <td>2.0</td>
      <td>78.2</td>
      <td>21.8</td>
      <td>21.5</td>
      <td>34.7</td>
      <td>19.7</td>
      <td>11.6</td>
      <td>6.9</td>
      <td>5.6</td>
      <td>7.7</td>
      <td>35.5</td>
      <td>38.3</td>
      <td>18.5</td>
      <td>87.1</td>
      <td>1.3</td>
      <td>7.5</td>
      <td>0.6</td>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.2</td>
      <td>2.6</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.5</td>
      <td>24.8</td>
      <td>18.5</td>
      <td>18.0</td>
      <td>14.8</td>
      <td>8.0</td>
      <td>1.6</td>
      <td>0.8</td>
      <td>60.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>2.5</td>
      <td>9.7</td>
      <td>19.5</td>
      <td>33.6</td>
      <td>25.9</td>
      <td>8.8</td>
      <td>1.3</td>
      <td>2.3</td>
      <td>7.0</td>
      <td>16.8</td>
      <td>72.6</td>
      <td>34.7</td>
      <td>17.8</td>
      <td>10.3</td>
      <td>9.8</td>
      <td>27.3</td>
      <td>26.4</td>
      <td>19.6</td>
      <td>17.0</td>
      <td>13.6</td>
      <td>4.8</td>
      <td>2.4</td>
      <td>16.2</td>
      <td>1.1</td>
      <td>8.9</td>
      <td>4.7</td>
      <td>33.4</td>
      <td>32.7</td>
      <td>14.5</td>
      <td>4.6</td>
      <td>9.4</td>
      <td>7.3</td>
      <td>5.7</td>
      <td>20.7</td>
      <td>14.2</td>
      <td>42.7</td>
      <td>24.66</td>
      <td>11.66</td>
      <td>17.26</td>
      <td>16.56</td>
      <td>11.38</td>
      <td>6.06</td>
      <td>10.58</td>
      <td>12.72</td>
      <td>11.48</td>
      <td>10.98</td>
      <td>27.225</td>
      <td>53.54</td>
      <td>10.18</td>
      <td>9.18</td>
      <td>27.68</td>
      <td>8.50</td>
      <td>7.22</td>
      <td>5.14</td>
      <td>8.72</td>
      <td>5.46</td>
      <td>4.32</td>
      <td>6.52</td>
      <td>31.80</td>
      <td>34.02</td>
      <td>29.62</td>
      <td>10.46</td>
      <td>2.26</td>
      <td>14.06</td>
      <td>15.76</td>
      <td>24.44</td>
      <td>22.02</td>
      <td>26.98</td>
      <td>4.72</td>
      <td>36.66</td>
      <td>34.04</td>
      <td>8.32</td>
      <td>4.82</td>
      <td>25.44</td>
      <td>13.54</td>
      <td>6.68</td>
      <td>35.28</td>
      <td>8.22</td>
      <td>4.74</td>
      <td>25.32</td>
      <td>7.98</td>
      <td>4.78</td>
      <td>23.76</td>
      <td>7.48</td>
      <td>4.30</td>
      <td>22.86</td>
      <td>7.34</td>
      <td>4.16</td>
      <td>24.78</td>
      <td>2.78</td>
      <td>2.74</td>
      <td>4.10</td>
      <td>1.74</td>
      <td>1.38</td>
      <td>2.94</td>
      <td>16.00</td>
      <td>12.14</td>
      <td>29.800</td>
      <td>3.48</td>
      <td>1.96</td>
      <td>11.54</td>
      <td>37.04</td>
      <td>38.00</td>
      <td>7.18</td>
      <td>2.58</td>
      <td>39.90</td>
      <td>6.56</td>
      <td>3.54</td>
      <td>19.78</td>
      <td>5.20</td>
      <td>1.00</td>
      <td>4.66</td>
      <td>4.04</td>
      <td>7.1</td>
      <td>12.22</td>
      <td>4.62</td>
      <td>33.80</td>
      <td>17.18</td>
      <td>10.60</td>
      <td>6.34</td>
      <td>4.76</td>
      <td>16.68</td>
      <td>10.04</td>
      <td>3.18</td>
      <td>38.18</td>
      <td>8.60</td>
      <td>8.14</td>
      <td>14.28</td>
      <td>6.60</td>
      <td>42.26</td>
      <td>13.74</td>
      <td>5.90</td>
      <td>30.36</td>
      <td>5.00</td>
      <td>5.20</td>
      <td>5.76</td>
      <td>0.24</td>
      <td>0.0</td>
      <td>89.76</td>
      <td>10.24</td>
      <td>87.36</td>
      <td>12.64</td>
      <td>90.06</td>
      <td>9.94</td>
      <td>96.20</td>
      <td>3.80</td>
      <td>85.32</td>
      <td>14.68</td>
      <td>87.28</td>
      <td>12.72</td>
      <td>92.88</td>
      <td>7.12</td>
      <td>84.56</td>
      <td>15.44</td>
      <td>66.825</td>
      <td>33.175</td>
      <td>49.28</td>
      <td>93.12</td>
      <td>6.88</td>
      <td>86.04</td>
      <td>84.42</td>
      <td>15.58</td>
      <td>92.78</td>
      <td>87.94</td>
      <td>12.06</td>
      <td>99.28</td>
      <td>91.22</td>
      <td>8.78</td>
      <td>80.84</td>
      <td>97.26</td>
      <td>2.74</td>
      <td>40.52</td>
      <td>90.52</td>
      <td>9.48</td>
      <td>9.98</td>
      <td>75.925</td>
      <td>24.075</td>
      <td>3.48</td>
      <td>69.72</td>
      <td>30.28</td>
      <td>82.96</td>
      <td>17.04</td>
      <td>82.80</td>
      <td>17.20</td>
      <td>82.78</td>
      <td>17.22</td>
      <td>92.44</td>
      <td>7.56</td>
      <td>92.06</td>
      <td>7.94</td>
      <td>91.46</td>
      <td>8.54</td>
      <td>90.56</td>
      <td>90.00</td>
      <td>91.18</td>
      <td>20.08</td>
      <td>19.4</td>
      <td>20.68</td>
      <td>99.42</td>
      <td>0.58</td>
      <td>89.42</td>
      <td>10.58</td>
      <td>83.96</td>
      <td>16.04</td>
      <td>92.28</td>
      <td>7.72</td>
      <td>83.96</td>
      <td>16.04</td>
      <td>81.30</td>
      <td>18.70</td>
      <td>92.28</td>
      <td>7.72</td>
      <td>95.90</td>
      <td>4.10</td>
      <td>99.56</td>
      <td>0.44</td>
      <td>90.16</td>
      <td>9.84</td>
      <td>88.20</td>
      <td>11.80</td>
      <td>87.44</td>
      <td>12.56</td>
      <td>11.50</td>
      <td>88.50</td>
      <td>4.76</td>
      <td>16.76</td>
      <td>0.16</td>
      <td>27.96</td>
      <td>12.74</td>
      <td>10.28</td>
      <td>11.44</td>
      <td>11.18</td>
      <td>22.90</td>
      <td>11.18</td>
      <td>9.92</td>
      <td>11.48</td>
      <td>11.760</td>
      <td>9.460</td>
      <td>18.05</td>
      <td>12.14</td>
      <td>17.54</td>
      <td>13.80</td>
      <td>11.34</td>
      <td>7.04</td>
      <td>13.48</td>
      <td>17.24</td>
      <td>13.96</td>
      <td>43.12</td>
      <td>6.74</td>
      <td>13.48</td>
      <td>12.22</td>
      <td>20.10</td>
      <td>9.38</td>
      <td>11.50</td>
      <td>22.28</td>
      <td>18.44</td>
      <td>12.76</td>
      <td>3.70</td>
      <td>3.44</td>
      <td>11.52</td>
      <td>28.14</td>
      <td>17.92</td>
      <td>6.16</td>
      <td>74.74</td>
      <td>54.22</td>
      <td>62.60</td>
      <td>48.02</td>
      <td>15.36</td>
      <td>5.54</td>
      <td>1.32</td>
      <td>0.66</td>
      <td>32.76</td>
      <td>12.16</td>
      <td>20.64</td>
      <td>2.48</td>
      <td>14.18</td>
      <td>9.40</td>
      <td>1.66</td>
      <td>0.26</td>
      <td>11.50</td>
      <td>81.08</td>
      <td>18.92</td>
      <td>78.24</td>
      <td>4.58</td>
      <td>1.96</td>
      <td>1.16</td>
      <td>1.66</td>
      <td>1.74</td>
      <td>0.56</td>
      <td>10.08</td>
      <td>0.00</td>
      <td>0.30</td>
      <td>12.48</td>
      <td>17.72</td>
      <td>13.96</td>
      <td>14.52</td>
      <td>7.86</td>
      <td>10.78</td>
      <td>5.30</td>
      <td>17.12</td>
      <td>0.50</td>
      <td>0.86</td>
      <td>4.62</td>
      <td>11.90</td>
      <td>21.40</td>
      <td>23.78</td>
      <td>16.78</td>
      <td>10.10</td>
      <td>10.04</td>
      <td>0.68</td>
      <td>4.20</td>
      <td>24.18</td>
      <td>53.42</td>
      <td>14.34</td>
      <td>3.14</td>
      <td>84.58</td>
      <td>15.42</td>
      <td>17.08</td>
      <td>35.90</td>
      <td>23.66</td>
      <td>12.16</td>
      <td>7.18</td>
      <td>4.04</td>
      <td>3.94</td>
      <td>29.0</td>
      <td>42.74</td>
      <td>24.32</td>
      <td>84.44</td>
      <td>6.26</td>
      <td>4.40</td>
      <td>1.32</td>
      <td>0</td>
      <td>2.84</td>
      <td>0.0</td>
      <td>0.58</td>
      <td>0.18</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>2.50</td>
      <td>99.14</td>
      <td>0.74</td>
      <td>0.10</td>
      <td>13.58</td>
      <td>16.26</td>
      <td>16.40</td>
      <td>18.54</td>
      <td>18.86</td>
      <td>10.82</td>
      <td>4.60</td>
      <td>0.92</td>
      <td>59.86</td>
      <td>40.14</td>
      <td>0.28</td>
      <td>1.42</td>
      <td>6.32</td>
      <td>14.58</td>
      <td>30.58</td>
      <td>24.54</td>
      <td>22.28</td>
      <td>0.36</td>
      <td>2.26</td>
      <td>8.16</td>
      <td>19.60</td>
      <td>69.60</td>
      <td>40.08</td>
      <td>13.68</td>
      <td>12.22</td>
      <td>6.82</td>
      <td>27.20</td>
      <td>35.98</td>
      <td>13.94</td>
      <td>16.16</td>
      <td>11.28</td>
      <td>4.26</td>
      <td>3.14</td>
      <td>15.26</td>
      <td>0.00</td>
      <td>1.30</td>
      <td>4.96</td>
      <td>31.72</td>
      <td>23.58</td>
      <td>32.24</td>
      <td>6.20</td>
      <td>20.60</td>
      <td>8.68</td>
      <td>6.80</td>
      <td>14.24</td>
      <td>12.00</td>
      <td>37.64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48002</td>
      <td>48002</td>
      <td>25.8</td>
      <td>219</td>
      <td>24</td>
      <td>11.0</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4.2</td>
      <td>0</td>
      <td>5.0</td>
      <td>2.8</td>
      <td>2.2</td>
      <td>4.1</td>
      <td>10.8</td>
      <td>4.3</td>
      <td>5.6</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>32.1</td>
      <td>3.1</td>
      <td>4.5</td>
      <td>15.6</td>
      <td>5.8</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>1.2</td>
      <td>1.8</td>
      <td>0.6</td>
      <td>17.4</td>
      <td>14.8</td>
      <td>21.3</td>
      <td>5.7</td>
      <td>0.0</td>
      <td>6.1</td>
      <td>11.1</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>15.7</td>
      <td>4.2</td>
      <td>4.2</td>
      <td>5.5</td>
      <td>4.1</td>
      <td>3.7</td>
      <td>7.0</td>
      <td>4.2</td>
      <td>4.2</td>
      <td>5.5</td>
      <td>3.0</td>
      <td>2.8</td>
      <td>6.2</td>
      <td>3.0</td>
      <td>2.9</td>
      <td>6.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>11.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.1</td>
      <td>12.8</td>
      <td>0.0</td>
      <td>19.4</td>
      <td>21.7</td>
      <td>5.4</td>
      <td>5.7</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>4.9</td>
      <td>4.2</td>
      <td>12.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.5</td>
      <td>5.0</td>
      <td>16.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>12.8</td>
      <td>18.5</td>
      <td>20</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>23.5</td>
      <td>2.5</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>93.5</td>
      <td>6.5</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>95.3</td>
      <td>4.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>90.7</td>
      <td>9.3</td>
      <td>94.7</td>
      <td>5.3</td>
      <td>89.9</td>
      <td>10.1</td>
      <td>61.3</td>
      <td>38.7</td>
      <td>58.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>91.9</td>
      <td>8.1</td>
      <td>97.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>69.2</td>
      <td>75</td>
      <td>25</td>
      <td>27.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>19.7</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>3.3</td>
      <td>88.2</td>
      <td>11.8</td>
      <td>84.4</td>
      <td>15.6</td>
      <td>82.4</td>
      <td>17.6</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>88.9</td>
      <td>11.1</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>25.0</td>
      <td>90.9</td>
      <td>90.7</td>
      <td>91.0</td>
      <td>13.3</td>
      <td>12.4</td>
      <td>14.2</td>
      <td>97.2</td>
      <td>2.8</td>
      <td>62.0</td>
      <td>38.0</td>
      <td>56.7</td>
      <td>43.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>56.7</td>
      <td>43.3</td>
      <td>51.0</td>
      <td>49.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>99.6</td>
      <td>0.4</td>
      <td>90.2</td>
      <td>9.8</td>
      <td>86.9</td>
      <td>13.1</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>8.9</td>
      <td>91.1</td>
      <td>0.0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>25.6</td>
      <td>9.9</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>7.8</td>
      <td>27.0</td>
      <td>7.4</td>
      <td>49.6</td>
      <td>25.6</td>
      <td>63.5</td>
      <td>10.1</td>
      <td>16.6</td>
      <td>13.7</td>
      <td>8.6</td>
      <td>0.0</td>
      <td>11.3</td>
      <td>13.0</td>
      <td>7.9</td>
      <td>68.3</td>
      <td>7.9</td>
      <td>11.3</td>
      <td>3.3</td>
      <td>21.9</td>
      <td>10.0</td>
      <td>8.9</td>
      <td>10.7</td>
      <td>13.7</td>
      <td>12.9</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>8.9</td>
      <td>22.6</td>
      <td>19.2</td>
      <td>5.5</td>
      <td>79.2</td>
      <td>63.8</td>
      <td>64.2</td>
      <td>56.0</td>
      <td>16.1</td>
      <td>7.8</td>
      <td>0.8</td>
      <td>0</td>
      <td>26.8</td>
      <td>10.7</td>
      <td>17.6</td>
      <td>2.3</td>
      <td>9.6</td>
      <td>8.3</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>8.9</td>
      <td>91.1</td>
      <td>8.9</td>
      <td>98.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>0</td>
      <td>0.0</td>
      <td>12.6</td>
      <td>21.9</td>
      <td>7.9</td>
      <td>24.3</td>
      <td>5.9</td>
      <td>10.0</td>
      <td>1.7</td>
      <td>15.6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>15.6</td>
      <td>31.0</td>
      <td>20.2</td>
      <td>14.8</td>
      <td>9.8</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>10.0</td>
      <td>63.2</td>
      <td>18.7</td>
      <td>5.6</td>
      <td>93.5</td>
      <td>6.5</td>
      <td>4.8</td>
      <td>27.2</td>
      <td>37.8</td>
      <td>13.6</td>
      <td>11.6</td>
      <td>5.0</td>
      <td>1.1</td>
      <td>18.4</td>
      <td>39.6</td>
      <td>40.8</td>
      <td>37.6</td>
      <td>40.8</td>
      <td>3.8</td>
      <td>5.1</td>
      <td>0</td>
      <td>11.8</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>97.4</td>
      <td>0.9</td>
      <td>1.7</td>
      <td>1.0</td>
      <td>19.9</td>
      <td>17.9</td>
      <td>29.7</td>
      <td>24.1</td>
      <td>6.3</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>67.7</td>
      <td>32.3</td>
      <td>0</td>
      <td>0.9</td>
      <td>5.3</td>
      <td>4.7</td>
      <td>33.4</td>
      <td>41.0</td>
      <td>14.6</td>
      <td>1.7</td>
      <td>1.4</td>
      <td>4.4</td>
      <td>9.4</td>
      <td>83.1</td>
      <td>32.8</td>
      <td>16.6</td>
      <td>11.9</td>
      <td>20.1</td>
      <td>18.6</td>
      <td>36.2</td>
      <td>19.6</td>
      <td>14.6</td>
      <td>19.1</td>
      <td>1.7</td>
      <td>7.2</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31.5</td>
      <td>0.0</td>
      <td>44.4</td>
      <td>24.1</td>
      <td>20.4</td>
      <td>0.0</td>
      <td>9.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.4</td>
      <td>28.64</td>
      <td>10.42</td>
      <td>15.02</td>
      <td>14.82</td>
      <td>9.46</td>
      <td>4.54</td>
      <td>8.88</td>
      <td>11.88</td>
      <td>10.14</td>
      <td>10.06</td>
      <td>14.260</td>
      <td>13.98</td>
      <td>9.80</td>
      <td>8.34</td>
      <td>19.18</td>
      <td>7.04</td>
      <td>8.48</td>
      <td>4.96</td>
      <td>7.06</td>
      <td>4.62</td>
      <td>2.60</td>
      <td>6.86</td>
      <td>26.42</td>
      <td>27.56</td>
      <td>22.66</td>
      <td>8.92</td>
      <td>2.22</td>
      <td>11.40</td>
      <td>14.72</td>
      <td>26.40</td>
      <td>24.10</td>
      <td>29.00</td>
      <td>7.04</td>
      <td>46.56</td>
      <td>30.88</td>
      <td>6.66</td>
      <td>3.26</td>
      <td>31.52</td>
      <td>11.92</td>
      <td>5.38</td>
      <td>42.58</td>
      <td>6.32</td>
      <td>2.78</td>
      <td>31.52</td>
      <td>5.98</td>
      <td>2.66</td>
      <td>31.60</td>
      <td>6.02</td>
      <td>2.68</td>
      <td>31.38</td>
      <td>4.30</td>
      <td>1.76</td>
      <td>20.10</td>
      <td>2.22</td>
      <td>1.78</td>
      <td>6.22</td>
      <td>3.80</td>
      <td>3.50</td>
      <td>12.50</td>
      <td>26.58</td>
      <td>13.56</td>
      <td>57.025</td>
      <td>6.74</td>
      <td>5.80</td>
      <td>22.68</td>
      <td>12.68</td>
      <td>9.52</td>
      <td>5.64</td>
      <td>3.80</td>
      <td>14.34</td>
      <td>7.28</td>
      <td>1.46</td>
      <td>42.40</td>
      <td>3.50</td>
      <td>3.68</td>
      <td>2.02</td>
      <td>1.82</td>
      <td>3.6</td>
      <td>10.62</td>
      <td>3.50</td>
      <td>39.18</td>
      <td>10.18</td>
      <td>3.56</td>
      <td>5.74</td>
      <td>2.36</td>
      <td>24.10</td>
      <td>6.26</td>
      <td>2.20</td>
      <td>41.90</td>
      <td>5.06</td>
      <td>3.74</td>
      <td>18.08</td>
      <td>9.90</td>
      <td>65.68</td>
      <td>11.68</td>
      <td>4.66</td>
      <td>30.06</td>
      <td>0.80</td>
      <td>0.68</td>
      <td>3.34</td>
      <td>1.60</td>
      <td>1.6</td>
      <td>94.52</td>
      <td>5.48</td>
      <td>66.48</td>
      <td>33.52</td>
      <td>97.42</td>
      <td>2.58</td>
      <td>94.62</td>
      <td>5.38</td>
      <td>96.46</td>
      <td>3.54</td>
      <td>97.26</td>
      <td>2.74</td>
      <td>99.08</td>
      <td>0.92</td>
      <td>93.30</td>
      <td>6.70</td>
      <td>74.500</td>
      <td>25.500</td>
      <td>38.48</td>
      <td>67.74</td>
      <td>32.26</td>
      <td>97.86</td>
      <td>95.30</td>
      <td>4.70</td>
      <td>100.00</td>
      <td>97.92</td>
      <td>2.08</td>
      <td>94.20</td>
      <td>98.80</td>
      <td>1.20</td>
      <td>65.50</td>
      <td>97.70</td>
      <td>2.30</td>
      <td>38.02</td>
      <td>88.92</td>
      <td>11.08</td>
      <td>8.44</td>
      <td>84.000</td>
      <td>16.000</td>
      <td>2.10</td>
      <td>96.12</td>
      <td>3.88</td>
      <td>90.72</td>
      <td>9.28</td>
      <td>92.12</td>
      <td>7.88</td>
      <td>89.90</td>
      <td>10.10</td>
      <td>90.48</td>
      <td>9.52</td>
      <td>92.66</td>
      <td>7.34</td>
      <td>90.16</td>
      <td>9.84</td>
      <td>90.18</td>
      <td>89.20</td>
      <td>91.10</td>
      <td>14.32</td>
      <td>12.9</td>
      <td>15.68</td>
      <td>98.22</td>
      <td>1.78</td>
      <td>75.80</td>
      <td>24.20</td>
      <td>71.66</td>
      <td>28.34</td>
      <td>69.38</td>
      <td>30.62</td>
      <td>71.66</td>
      <td>28.34</td>
      <td>74.64</td>
      <td>25.38</td>
      <td>69.38</td>
      <td>30.62</td>
      <td>63.25</td>
      <td>36.75</td>
      <td>99.28</td>
      <td>0.72</td>
      <td>84.68</td>
      <td>15.32</td>
      <td>87.54</td>
      <td>12.46</td>
      <td>77.98</td>
      <td>22.02</td>
      <td>10.02</td>
      <td>89.98</td>
      <td>3.50</td>
      <td>14.12</td>
      <td>1.52</td>
      <td>21.16</td>
      <td>9.96</td>
      <td>10.14</td>
      <td>9.88</td>
      <td>9.66</td>
      <td>14.26</td>
      <td>9.56</td>
      <td>20.24</td>
      <td>9.60</td>
      <td>17.925</td>
      <td>15.125</td>
      <td>23.75</td>
      <td>11.50</td>
      <td>17.90</td>
      <td>12.92</td>
      <td>11.34</td>
      <td>5.14</td>
      <td>12.24</td>
      <td>14.44</td>
      <td>12.14</td>
      <td>34.28</td>
      <td>7.44</td>
      <td>12.24</td>
      <td>10.52</td>
      <td>18.06</td>
      <td>9.10</td>
      <td>9.98</td>
      <td>22.74</td>
      <td>14.74</td>
      <td>5.92</td>
      <td>7.92</td>
      <td>5.80</td>
      <td>10.14</td>
      <td>24.86</td>
      <td>14.38</td>
      <td>6.50</td>
      <td>77.12</td>
      <td>61.44</td>
      <td>68.52</td>
      <td>56.90</td>
      <td>9.94</td>
      <td>3.82</td>
      <td>1.94</td>
      <td>0.74</td>
      <td>26.80</td>
      <td>11.48</td>
      <td>14.42</td>
      <td>1.78</td>
      <td>13.96</td>
      <td>9.60</td>
      <td>1.78</td>
      <td>0.12</td>
      <td>10.02</td>
      <td>93.98</td>
      <td>6.02</td>
      <td>88.98</td>
      <td>1.66</td>
      <td>1.44</td>
      <td>1.18</td>
      <td>0.98</td>
      <td>1.22</td>
      <td>1.16</td>
      <td>3.42</td>
      <td>0.00</td>
      <td>0.28</td>
      <td>16.74</td>
      <td>22.32</td>
      <td>9.78</td>
      <td>14.82</td>
      <td>6.84</td>
      <td>6.28</td>
      <td>2.64</td>
      <td>20.30</td>
      <td>0.32</td>
      <td>0.96</td>
      <td>2.98</td>
      <td>9.34</td>
      <td>20.62</td>
      <td>22.72</td>
      <td>20.86</td>
      <td>11.92</td>
      <td>10.26</td>
      <td>0.36</td>
      <td>3.72</td>
      <td>15.48</td>
      <td>59.18</td>
      <td>17.00</td>
      <td>4.26</td>
      <td>86.96</td>
      <td>13.04</td>
      <td>15.42</td>
      <td>34.04</td>
      <td>25.38</td>
      <td>12.96</td>
      <td>7.76</td>
      <td>4.40</td>
      <td>2.48</td>
      <td>23.5</td>
      <td>41.06</td>
      <td>32.96</td>
      <td>47.38</td>
      <td>29.72</td>
      <td>8.66</td>
      <td>2.78</td>
      <td>0</td>
      <td>8.66</td>
      <td>0.1</td>
      <td>1.70</td>
      <td>0.94</td>
      <td>0.56</td>
      <td>0.68</td>
      <td>3.28</td>
      <td>98.26</td>
      <td>1.30</td>
      <td>0.42</td>
      <td>8.74</td>
      <td>17.40</td>
      <td>24.34</td>
      <td>23.10</td>
      <td>18.64</td>
      <td>6.50</td>
      <td>0.88</td>
      <td>0.40</td>
      <td>72.18</td>
      <td>27.82</td>
      <td>0.00</td>
      <td>0.80</td>
      <td>2.12</td>
      <td>11.98</td>
      <td>37.10</td>
      <td>29.40</td>
      <td>18.58</td>
      <td>1.06</td>
      <td>3.12</td>
      <td>8.02</td>
      <td>20.58</td>
      <td>67.22</td>
      <td>34.36</td>
      <td>15.32</td>
      <td>14.74</td>
      <td>8.36</td>
      <td>27.22</td>
      <td>31.00</td>
      <td>27.66</td>
      <td>12.22</td>
      <td>7.50</td>
      <td>6.32</td>
      <td>1.94</td>
      <td>13.34</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>7.74</td>
      <td>39.30</td>
      <td>30.00</td>
      <td>13.90</td>
      <td>8.28</td>
      <td>10.38</td>
      <td>18.60</td>
      <td>16.56</td>
      <td>10.12</td>
      <td>6.32</td>
      <td>38.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48003</td>
      <td>48003</td>
      <td>24.1</td>
      <td>388</td>
      <td>41</td>
      <td>10.6</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.6</td>
      <td>5.7</td>
      <td>5.7</td>
      <td>7.9</td>
      <td>3.1</td>
      <td>5.6</td>
      <td>7.5</td>
      <td>6.6</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>8.7</td>
      <td>6.6</td>
      <td>7.0</td>
      <td>13.4</td>
      <td>5.1</td>
      <td>9.6</td>
      <td>2.9</td>
      <td>7.3</td>
      <td>4.9</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>23.5</td>
      <td>29.1</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>2.3</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>22.4</td>
      <td>31.7</td>
      <td>15.2</td>
      <td>8.6</td>
      <td>29.7</td>
      <td>28.6</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>35.9</td>
      <td>8.6</td>
      <td>1.1</td>
      <td>46.0</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>35.9</td>
      <td>4.9</td>
      <td>1.2</td>
      <td>35.9</td>
      <td>5.0</td>
      <td>1.2</td>
      <td>35.9</td>
      <td>3.6</td>
      <td>0.0</td>
      <td>34.6</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.8</td>
      <td>6.9</td>
      <td>61.1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>32.7</td>
      <td>9.1</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.1</td>
      <td>1.3</td>
      <td>45.6</td>
      <td>3.2</td>
      <td>2.7</td>
      <td>1.8</td>
      <td>1.2</td>
      <td>0.0</td>
      <td>10.4</td>
      <td>1.3</td>
      <td>52.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.1</td>
      <td>1.5</td>
      <td>52.9</td>
      <td>2.4</td>
      <td>1.1</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.7</td>
      <td>2</td>
      <td>56.3</td>
      <td>8.6</td>
      <td>2.3</td>
      <td>38.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>90.9</td>
      <td>9.1</td>
      <td>59.8</td>
      <td>40.2</td>
      <td>95.1</td>
      <td>4.9</td>
      <td>91.3</td>
      <td>8.7</td>
      <td>92.6</td>
      <td>7.4</td>
      <td>96.6</td>
      <td>3.4</td>
      <td>98.1</td>
      <td>1.9</td>
      <td>88.6</td>
      <td>11.4</td>
      <td>73.1</td>
      <td>26.9</td>
      <td>38.3</td>
      <td>56.5</td>
      <td>43.5</td>
      <td>100.0</td>
      <td>89.4</td>
      <td>10.6</td>
      <td>100.0</td>
      <td>97.6</td>
      <td>2.4</td>
      <td>100.0</td>
      <td>97.5</td>
      <td>2.5</td>
      <td>31.1</td>
      <td>100</td>
      <td>0</td>
      <td>42.8</td>
      <td>84.4</td>
      <td>15.6</td>
      <td>4.3</td>
      <td>66.7</td>
      <td>33.3</td>
      <td>1.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>84.6</td>
      <td>15.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>78.9</td>
      <td>21.1</td>
      <td>91.5</td>
      <td>91.5</td>
      <td>91.4</td>
      <td>18.3</td>
      <td>17.5</td>
      <td>19.0</td>
      <td>98.5</td>
      <td>1.5</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>78.8</td>
      <td>21.2</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>74.8</td>
      <td>25.2</td>
      <td>98.9</td>
      <td>1.1</td>
      <td>80.9</td>
      <td>19.1</td>
      <td>95.5</td>
      <td>4.5</td>
      <td>64.4</td>
      <td>35.6</td>
      <td>8.3</td>
      <td>91.7</td>
      <td>0.6</td>
      <td>12.0</td>
      <td>5.8</td>
      <td>10.6</td>
      <td>9.8</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>43.9</td>
      <td>7.2</td>
      <td>30.6</td>
      <td>8.0</td>
      <td>16.5</td>
      <td>19.5</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>22.3</td>
      <td>9.9</td>
      <td>11.7</td>
      <td>3.9</td>
      <td>10.5</td>
      <td>12.6</td>
      <td>9.7</td>
      <td>32.5</td>
      <td>7.1</td>
      <td>10.5</td>
      <td>7.3</td>
      <td>15.1</td>
      <td>10.8</td>
      <td>8.4</td>
      <td>25.0</td>
      <td>10.7</td>
      <td>7.9</td>
      <td>7.5</td>
      <td>1.4</td>
      <td>8.4</td>
      <td>29.1</td>
      <td>7.9</td>
      <td>4.5</td>
      <td>79.6</td>
      <td>61.1</td>
      <td>69.9</td>
      <td>56.8</td>
      <td>10.9</td>
      <td>4.3</td>
      <td>0.8</td>
      <td>0</td>
      <td>30.2</td>
      <td>10.1</td>
      <td>20.7</td>
      <td>2.8</td>
      <td>12.5</td>
      <td>7.0</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>8.3</td>
      <td>96.7</td>
      <td>3.3</td>
      <td>84.7</td>
      <td>2.1</td>
      <td>0.4</td>
      <td>3.3</td>
      <td>2.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>6.5</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.6</td>
      <td>25.7</td>
      <td>9.5</td>
      <td>15.0</td>
      <td>5.9</td>
      <td>4.7</td>
      <td>1.8</td>
      <td>18.7</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>3.8</td>
      <td>9.0</td>
      <td>14.1</td>
      <td>24.8</td>
      <td>21.6</td>
      <td>11.5</td>
      <td>14.1</td>
      <td>0.7</td>
      <td>4.8</td>
      <td>16.6</td>
      <td>59.5</td>
      <td>15.0</td>
      <td>3.4</td>
      <td>89.0</td>
      <td>11.0</td>
      <td>8.9</td>
      <td>41.7</td>
      <td>23.7</td>
      <td>12.2</td>
      <td>9.3</td>
      <td>4.2</td>
      <td>1.8</td>
      <td>28.1</td>
      <td>44.2</td>
      <td>25.9</td>
      <td>77.4</td>
      <td>12.4</td>
      <td>4.9</td>
      <td>1.4</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>0.3</td>
      <td>0.4</td>
      <td>0</td>
      <td>0.3</td>
      <td>1.6</td>
      <td>98.9</td>
      <td>0.7</td>
      <td>0.4</td>
      <td>13.7</td>
      <td>14.1</td>
      <td>20.6</td>
      <td>25.0</td>
      <td>18.7</td>
      <td>5.7</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>71.3</td>
      <td>28.7</td>
      <td>0</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>10.8</td>
      <td>39.6</td>
      <td>29.7</td>
      <td>18.7</td>
      <td>0.0</td>
      <td>1.7</td>
      <td>8.4</td>
      <td>14.8</td>
      <td>75.0</td>
      <td>32.0</td>
      <td>14.5</td>
      <td>12.4</td>
      <td>8.9</td>
      <td>32.2</td>
      <td>29.3</td>
      <td>27.1</td>
      <td>13.1</td>
      <td>10.3</td>
      <td>7.3</td>
      <td>1.9</td>
      <td>11.0</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>16.2</td>
      <td>40.8</td>
      <td>22.8</td>
      <td>14.5</td>
      <td>1.8</td>
      <td>4.4</td>
      <td>11.4</td>
      <td>19.3</td>
      <td>8.8</td>
      <td>6.6</td>
      <td>49.6</td>
      <td>25.38</td>
      <td>9.54</td>
      <td>12.80</td>
      <td>12.68</td>
      <td>8.76</td>
      <td>5.86</td>
      <td>8.28</td>
      <td>10.86</td>
      <td>9.34</td>
      <td>8.62</td>
      <td>15.380</td>
      <td>35.68</td>
      <td>7.92</td>
      <td>7.76</td>
      <td>13.96</td>
      <td>9.74</td>
      <td>6.70</td>
      <td>3.24</td>
      <td>6.68</td>
      <td>3.80</td>
      <td>2.60</td>
      <td>5.24</td>
      <td>27.24</td>
      <td>29.88</td>
      <td>24.06</td>
      <td>8.48</td>
      <td>0.76</td>
      <td>12.10</td>
      <td>14.18</td>
      <td>22.38</td>
      <td>18.58</td>
      <td>26.46</td>
      <td>2.88</td>
      <td>45.64</td>
      <td>30.08</td>
      <td>7.08</td>
      <td>4.50</td>
      <td>22.64</td>
      <td>11.76</td>
      <td>6.74</td>
      <td>33.28</td>
      <td>6.94</td>
      <td>4.24</td>
      <td>22.64</td>
      <td>6.34</td>
      <td>3.82</td>
      <td>23.58</td>
      <td>5.90</td>
      <td>3.74</td>
      <td>20.14</td>
      <td>4.82</td>
      <td>2.12</td>
      <td>21.86</td>
      <td>0.48</td>
      <td>0.14</td>
      <td>1.72</td>
      <td>5.82</td>
      <td>5.08</td>
      <td>11.60</td>
      <td>26.44</td>
      <td>15.64</td>
      <td>35.540</td>
      <td>7.64</td>
      <td>8.80</td>
      <td>3.24</td>
      <td>21.84</td>
      <td>12.26</td>
      <td>7.48</td>
      <td>7.38</td>
      <td>11.16</td>
      <td>6.26</td>
      <td>2.20</td>
      <td>23.66</td>
      <td>3.66</td>
      <td>3.58</td>
      <td>2.88</td>
      <td>2.88</td>
      <td>2.6</td>
      <td>10.56</td>
      <td>6.66</td>
      <td>26.72</td>
      <td>16.98</td>
      <td>7.32</td>
      <td>6.54</td>
      <td>3.64</td>
      <td>23.86</td>
      <td>6.52</td>
      <td>3.84</td>
      <td>20.10</td>
      <td>9.64</td>
      <td>9.66</td>
      <td>17.86</td>
      <td>12.54</td>
      <td>38.64</td>
      <td>12.16</td>
      <td>6.62</td>
      <td>33.76</td>
      <td>2.22</td>
      <td>2.10</td>
      <td>0.00</td>
      <td>0.66</td>
      <td>0.5</td>
      <td>90.52</td>
      <td>9.48</td>
      <td>73.52</td>
      <td>26.48</td>
      <td>93.92</td>
      <td>6.08</td>
      <td>86.60</td>
      <td>13.40</td>
      <td>95.00</td>
      <td>5.00</td>
      <td>92.06</td>
      <td>7.94</td>
      <td>95.40</td>
      <td>4.60</td>
      <td>87.00</td>
      <td>13.00</td>
      <td>63.040</td>
      <td>36.960</td>
      <td>46.68</td>
      <td>75.68</td>
      <td>24.32</td>
      <td>99.34</td>
      <td>92.62</td>
      <td>7.38</td>
      <td>99.92</td>
      <td>92.80</td>
      <td>7.20</td>
      <td>96.44</td>
      <td>96.94</td>
      <td>3.06</td>
      <td>74.88</td>
      <td>91.58</td>
      <td>8.42</td>
      <td>35.90</td>
      <td>90.66</td>
      <td>9.34</td>
      <td>14.50</td>
      <td>74.300</td>
      <td>25.700</td>
      <td>2.44</td>
      <td>69.84</td>
      <td>30.16</td>
      <td>83.06</td>
      <td>16.94</td>
      <td>85.92</td>
      <td>14.08</td>
      <td>81.24</td>
      <td>18.76</td>
      <td>90.30</td>
      <td>9.70</td>
      <td>93.18</td>
      <td>6.82</td>
      <td>86.96</td>
      <td>13.04</td>
      <td>89.56</td>
      <td>88.66</td>
      <td>90.46</td>
      <td>19.24</td>
      <td>19.1</td>
      <td>19.34</td>
      <td>97.64</td>
      <td>2.36</td>
      <td>74.20</td>
      <td>25.80</td>
      <td>57.52</td>
      <td>42.48</td>
      <td>72.16</td>
      <td>27.84</td>
      <td>57.52</td>
      <td>42.48</td>
      <td>50.02</td>
      <td>49.98</td>
      <td>72.16</td>
      <td>27.84</td>
      <td>73.96</td>
      <td>26.04</td>
      <td>99.32</td>
      <td>0.68</td>
      <td>84.86</td>
      <td>15.14</td>
      <td>71.12</td>
      <td>28.88</td>
      <td>86.52</td>
      <td>13.48</td>
      <td>10.26</td>
      <td>89.74</td>
      <td>4.12</td>
      <td>14.42</td>
      <td>0.24</td>
      <td>21.44</td>
      <td>11.04</td>
      <td>9.48</td>
      <td>10.08</td>
      <td>9.56</td>
      <td>10.42</td>
      <td>9.58</td>
      <td>25.08</td>
      <td>9.60</td>
      <td>24.600</td>
      <td>16.720</td>
      <td>30.32</td>
      <td>11.08</td>
      <td>13.62</td>
      <td>15.68</td>
      <td>10.60</td>
      <td>2.50</td>
      <td>12.18</td>
      <td>14.08</td>
      <td>11.18</td>
      <td>40.02</td>
      <td>8.00</td>
      <td>12.18</td>
      <td>7.22</td>
      <td>21.62</td>
      <td>9.38</td>
      <td>10.22</td>
      <td>19.80</td>
      <td>14.82</td>
      <td>11.62</td>
      <td>7.36</td>
      <td>4.98</td>
      <td>10.34</td>
      <td>24.20</td>
      <td>15.48</td>
      <td>6.78</td>
      <td>76.20</td>
      <td>61.38</td>
      <td>65.60</td>
      <td>55.06</td>
      <td>13.08</td>
      <td>6.28</td>
      <td>0.78</td>
      <td>0.04</td>
      <td>26.88</td>
      <td>12.04</td>
      <td>14.68</td>
      <td>1.94</td>
      <td>13.88</td>
      <td>9.88</td>
      <td>1.76</td>
      <td>0.18</td>
      <td>10.26</td>
      <td>91.42</td>
      <td>8.58</td>
      <td>87.80</td>
      <td>2.08</td>
      <td>1.30</td>
      <td>1.42</td>
      <td>0.80</td>
      <td>0.74</td>
      <td>0.14</td>
      <td>5.70</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>15.50</td>
      <td>21.80</td>
      <td>11.68</td>
      <td>16.58</td>
      <td>8.84</td>
      <td>7.58</td>
      <td>2.58</td>
      <td>15.42</td>
      <td>0.54</td>
      <td>0.56</td>
      <td>2.02</td>
      <td>8.00</td>
      <td>17.88</td>
      <td>22.92</td>
      <td>19.90</td>
      <td>13.28</td>
      <td>14.92</td>
      <td>0.54</td>
      <td>2.32</td>
      <td>15.84</td>
      <td>55.28</td>
      <td>21.80</td>
      <td>4.26</td>
      <td>87.58</td>
      <td>12.42</td>
      <td>14.46</td>
      <td>34.50</td>
      <td>27.92</td>
      <td>12.34</td>
      <td>7.18</td>
      <td>3.62</td>
      <td>3.06</td>
      <td>24.6</td>
      <td>37.28</td>
      <td>35.08</td>
      <td>61.08</td>
      <td>22.60</td>
      <td>5.64</td>
      <td>4.00</td>
      <td>0</td>
      <td>5.00</td>
      <td>0.0</td>
      <td>1.10</td>
      <td>0.60</td>
      <td>0.44</td>
      <td>0.58</td>
      <td>3.08</td>
      <td>98.20</td>
      <td>1.20</td>
      <td>0.58</td>
      <td>9.10</td>
      <td>14.48</td>
      <td>18.08</td>
      <td>20.24</td>
      <td>23.36</td>
      <td>12.10</td>
      <td>1.62</td>
      <td>0.96</td>
      <td>69.22</td>
      <td>30.78</td>
      <td>0.08</td>
      <td>0.50</td>
      <td>5.40</td>
      <td>8.50</td>
      <td>32.88</td>
      <td>26.62</td>
      <td>25.96</td>
      <td>0.44</td>
      <td>2.12</td>
      <td>5.42</td>
      <td>15.54</td>
      <td>76.52</td>
      <td>32.32</td>
      <td>17.34</td>
      <td>14.08</td>
      <td>12.12</td>
      <td>24.14</td>
      <td>41.46</td>
      <td>20.44</td>
      <td>15.06</td>
      <td>7.30</td>
      <td>4.02</td>
      <td>2.60</td>
      <td>9.08</td>
      <td>1.94</td>
      <td>1.24</td>
      <td>4.66</td>
      <td>23.50</td>
      <td>20.84</td>
      <td>35.20</td>
      <td>12.62</td>
      <td>13.06</td>
      <td>11.02</td>
      <td>8.38</td>
      <td>10.36</td>
      <td>3.52</td>
      <td>53.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfm.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>desc</th>
      <th>fname</th>
      <th>fname_with_ann</th>
      <th>tablename</th>
      <th>outcode</th>
      <th>coltype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zip</td>
      <td>zip</td>
      <td>zip</td>
      <td>zip</td>
      <td>zip</td>
      <td>zip</td>
      <td>other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zcta</td>
      <td>zcta</td>
      <td>zcta</td>
      <td>zcta</td>
      <td>zcta</td>
      <td>zcta</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>perc_pre1950_housing__CLPPP</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfm['coltype'].unique()
```




    array(['other', 'census', 'close'], dtype=object)




```python
# check that the percent bll ge 5 is a percent of those tested:
#dfl['perc_bll_ge5__CLPPP'] - 100*dfl['blltot_ge5__CLPPP'] / dfl['children_tested__CLPPP']

```

Linear regression to predict % blls >= 5:


```python
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dfl, dfm)
```

    number of features in X: 354
    shape of X:  (938, 354)
    shape of y:  (938,)
    best params: {'alpha': 10}
    R^2 for this model is: 0.0665907441073



![png](testnotebook1_files/testnotebook1_14_1.png)


    Features with nonzero importances:  9



![png](testnotebook1_files/testnotebook1_14_3.png)


linear regression to predict %bll>=5, only zipcodes with %bll>=5 of greater than zero:


```python
dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm)
# examine which ones are important!!!!
```

    number of features in X: 354
    shape of X:  (566, 354)
    shape of y:  (566,)
    best params: {'alpha': 10}
    R^2 for this model is: 0.386077463245



![png](testnotebook1_files/testnotebook1_16_1.png)


    Features with nonzero importances:  9



![png](testnotebook1_files/testnotebook1_16_3.png)


linear regression to predict %bll>=10, only zipcodes with %bll>=10 of greater than zero:


```python
dflcurr = dfl.loc[dfl['perc_bll_ge10__CLPPP']>0,:]
y_name = 'perc_bll_ge10__CLPPP'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, y_name=y_name)
# doesn't work so well..
```

    number of features in X: 354
    shape of X:  (159, 354)
    shape of y:  (159,)
    best params: {'alpha': 0.3}
    R^2 for this model is: 0.229249927042



![png](testnotebook1_files/testnotebook1_18_1.png)


    Features with nonzero importances:  65



![png](testnotebook1_files/testnotebook1_18_3.png)


Try with random forest regressor (%bll>=5, all zipcodes):


```python
# try with all zipcodes:
modelType = 'randomforestregressor'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dfl, dfm, modelType=modelType)
# examine which ones are important!!!!
```

    number of features in X: 354
    shape of X:  (938, 354)
    shape of y:  (938,)
    R^2 for this model is: -0.359044565183



![png](testnotebook1_files/testnotebook1_20_1.png)


    Features with nonzero importances:  354



![png](testnotebook1_files/testnotebook1_20_3.png)


Try with random forest regressor (%bll>=5, only zipcodes with nonzero value):


```python
#  Try with random forest regressor (%bll>=5, only zipcodes with nonzero value):
##! need to kfold this..
# try making a smaller test set
dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
modelType = 'randomforestregressor'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, modelType=modelType)

```

    number of features in X: 354
    shape of X:  (566, 354)
    shape of y:  (566,)
    R^2 for this model is: 0.587984567323



![png](testnotebook1_files/testnotebook1_22_1.png)


    Features with nonzero importances:  353



![png](testnotebook1_files/testnotebook1_22_3.png)



```python
# look at the top features correlations:

#model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name
Nfeatures = 15

# take only the top features:
indices = np.argsort(importances)
indices = indices[-Nfeatures:]
X_names_top = X_names[indices]
featurenames = np.append(X_names_top,'perc_bll_ge5__CLPPP')
featuredescs = lt.feature_descriptions(dfm, featurenames)
# plot correlation
dflplot = dfl.copy()
dflplot.columns = dfm['desc'].values
dst.plot_corr(dflplot.loc[:,featuredescs],size=10, toCluster=True, nsamples=10000)

```


![png](testnotebook1_files/testnotebook1_23_0.png)


Build random forest classifier to predict which zipcodes will have *any* children with bll>=5:


```python
dflcurr = dfl.copy()
dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'perc_bll_ge5__CLPPP']
dflcurr.loc[dflcurr['any_bll_ge5']>0,'any_bll_ge5'] = 1
dflcurr.loc[dflcurr['any_bll_ge5']<=0,'any_bll_ge5'] = 0
dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'any_bll_ge5'].astype(int)
print "Unique values in y column: %s" % dflcurr['any_bll_ge5'].unique()
#dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]

y_name = 'any_bll_ge5'
modelType = 'randomforestclassifier'
MLtype = 'classification'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, y_name=y_name, modelType=modelType, MLtype=MLtype)
# examine which ones are important!!!!
```

    Unique values in y column: [1 0]
    number of features in X: 354
    shape of X:  (938, 354)
    shape of y:  (938,)
    Features with nonzero importances:  354



![png](testnotebook1_files/testnotebook1_25_1.png)



![png](testnotebook1_files/testnotebook1_25_2.png)



```python
# look at the top features correlations:

#model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name
Nfeatures = 15

# take only the top features:
indices = np.argsort(importances)
indices = indices[-Nfeatures:]
X_names_top = X_names[indices]
featurenames = np.append(X_names_top,'perc_bll_ge5__CLPPP')
featuredescs = lt.feature_descriptions(dfm, featurenames)
# plot correlation
dflplot = dfl.copy()
dflplot.columns = dfm['desc'].values
dst.plot_corr(dflplot.loc[:,featuredescs],size=10, toCluster=True, nsamples=10000)

```


![png](testnotebook1_files/testnotebook1_26_0.png)


Build logistic regression classifier to predict which zipcodes will have *any* children with bll>=5:


```python
#dflcurr = dfl.copy()
#dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'perc_bll_ge5__CLPPP']
#dflcurr.loc[dflcurr['any_bll_ge5']>0,'any_bll_ge5'] = 1
#dflcurr.loc[dflcurr['any_bll_ge5']<=0,'any_bll_ge5'] = 0
#dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'any_bll_ge5'].astype(int)
#print "Unique values in y column: %s" % dflcurr['any_bll_ge5'].unique()
##dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
#
#y_name = 'any_bll_ge5'
#modelType = 'logisticregression'
#MLtype = 'classification'
#model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, y_name=y_name, modelType=modelType, MLtype=MLtype)
## examine which ones are important!!!!
```

### map to geographies


```python
# draw the shape files for the zip codes..
#dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']<25,:]
dfl_zctashapes, dfl_zctacenters = lt.draw_zipcenters(zctacodes, zctashapes, dfl, 'perc_bll_ge5__CLPPP', gamma=0.25)
plt.show()

```


![png](testnotebook1_files/testnotebook1_30_0.png)



```python

```


```python

```


```python
# Average bll's for each city:
dfl_cities = lt.zipcode_cities(dfl)
len(np.unique(dfl_cities))


```




    798




```python
# improve the regressor:
```


```python
#  Try gradient boosting regressor (%bll>=5, only zipcodes with nonzero value):
##! need to kfold this..
# try making a smaller test set
dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
modelType = 'gradientboostregressor'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, modelType=modelType)

```

    number of features in X: 354
    shape of X:  (566, 354)
    shape of y:  (566,)
    R^2 for this model is: 0.584126821133



![png](testnotebook1_files/testnotebook1_35_1.png)


    Features with nonzero importances:  246



![png](testnotebook1_files/testnotebook1_35_3.png)



```python
#  Try with random forest regressor (%bll>=5, only zipcodes with nonzero value):
##! need to kfold this..
# try making a smaller test set
dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
modelType = 'randomforestregressor'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, modelType=modelType)

```

    number of features in X: 354
    shape of X:  (566, 354)
    shape of y:  (566,)
    best params: {'n_estimators': 1000, 'max_depth': None}
    R^2 for this model is: 0.584054816359



![png](testnotebook1_files/testnotebook1_36_1.png)


    Features with nonzero importances:  354



![png](testnotebook1_files/testnotebook1_36_3.png)



```python
#basedir = '/Users/matto/Documents/censusdata/arnhold_challenge/'
#pickle.dump( model, open( basedir + 'model_randforestregress_1000tree.p', "wb" ) )

```


```python
dfm.columns
```




    Index([u'code', u'desc', u'fname', u'fname_with_ann', u'tablename', u'outcode',
           u'coltype'],
          dtype='object')




```python
# Try with random forest regressor (%bll>=5, only zipcodes with nonzero value):
# remove # children under 6:

dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]
dflcurr = dflcurr.drop('children_under6__CLPPP', axis=1)
dfmcurr = dfm.loc[dfm['outcode']!='children_under6__CLPPP',:]

assert all(dfmcurr['outcode'].values==dflcurr.columns)
modelType = 'randomforestregressor'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfmcurr, modelType=modelType)

```

    number of features in X: 706
    False
    shape of X:  (566, 706)
    shape of y:  (566,)
    best params: {'n_estimators': 10, 'max_depth': None}
    R^2 for this model is: 0.163324262451



![png](testnotebook1_files/testnotebook1_39_1.png)


    Features with nonzero importances:  667



![png](testnotebook1_files/testnotebook1_39_3.png)



```python
dfl.columns
```




    Index([u'zip', u'zcta', u'perc_pre1950_housing__CLPPP',
           u'children_under6__CLPPP', u'children_tested__CLPPP',
           u'perc_tested__CLPPP', u'bll_lt5__CLPPP', u'bll_5to9__CLPPP',
           u'capillary_ge10__CLPPP', u'venous_10to19__CLPPP',
           ...
           u'HC03_VC185__DP04_close', u'HC03_VC186__DP04_close',
           u'HC03_VC187__DP04_close', u'HC03_VC188__DP04_close',
           u'HC03_VC197__DP04_close', u'HC03_VC198__DP04_close',
           u'HC03_VC199__DP04_close', u'HC03_VC200__DP04_close',
           u'HC03_VC201__DP04_close', u'HC03_VC202__DP04_close'],
          dtype='object', length=721)




```python
sns.regplot(dflcurr['children_under6__CLPPP'],dflcurr['perc_bll_ge5__CLPPP'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1401e62d0>




![png](testnotebook1_files/testnotebook1_41_1.png)


Build random forest classifier to predict which zipcodes will have *any* children with bll>=5:

(play with this more - try to raise the threshold above 0. also, remove children under 6 count)


```python
bllpercentcutoff = 3.999
#bllpercentcutoff = 0

dflcurr = dfl.copy()
dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'perc_bll_ge5__CLPPP']
dflcurr.loc[dflcurr['any_bll_ge5']<=bllpercentcutoff,'any_bll_ge5'] = 0
dflcurr.loc[dflcurr['any_bll_ge5']>bllpercentcutoff,'any_bll_ge5'] = 1

dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'any_bll_ge5'].astype(int)
print "Unique values in y column: %s" % dflcurr['any_bll_ge5'].unique()
#dflcurr['any_bll_ge5']
#dflcurr['perc_bll_ge5__CLPPP']
```

    0      False
    1       True
    2      False
    3      False
    4       True
    5      False
    6      False
    7      False
    8      False
    9      False
    10     False
    11     False
    12     False
    13     False
    14     False
    15     False
    16     False
    17      True
    18     False
    19     False
    20     False
    21     False
    22     False
    23      True
    24     False
    25     False
    26     False
    27      True
    28     False
    29     False
    30     False
    31     False
    32     False
    33     False
    34     False
    35     False
    36     False
    37      True
    38     False
    39     False
    40     False
    41     False
    42     False
    43     False
    44     False
    45     False
    46     False
    47     False
    48     False
    49      True
    50     False
    51     False
    52     False
    53     False
    54     False
    55     False
    56     False
    57     False
    58     False
    59     False
    60     False
    61     False
    62     False
    63     False
    64     False
    65     False
    66     False
    67     False
    68     False
    69     False
    70     False
    71     False
    72     False
    73     False
    74     False
    75     False
    76     False
    77     False
    78      True
    79     False
    80     False
    81     False
    82     False
    83     False
    84     False
    85     False
    86     False
    87     False
    88      True
    89     False
    90     False
    91     False
    92     False
    93     False
    94     False
    95     False
    96     False
    97     False
    98     False
    99     False
    100    False
    101    False
    102    False
    103     True
    104     True
    105    False
    106    False
    107    False
    108    False
    109    False
    110    False
    111    False
    112    False
    113    False
    114    False
    115    False
    116    False
    117    False
    118    False
    119    False
    120    False
    121    False
    122    False
    123    False
    124    False
    125    False
    126    False
    127    False
    128    False
    129    False
    130    False
    131    False
    132    False
    133    False
    134    False
    135    False
    136    False
    137    False
    138     True
    139     True
    140     True
    141     True
    142     True
    143     True
    144     True
    145     True
    146     True
    147     True
    148     True
    149     True
    150     True
    151     True
    152     True
    153    False
    154    False
    155     True
    156    False
    157     True
    158    False
    159     True
    160    False
    161    False
    162     True
    163     True
    164     True
    165     True
    166     True
    167     True
    168    False
    169    False
    170     True
    171    False
    172    False
    173    False
    174    False
    175    False
    176    False
    177    False
    178    False
    179    False
    180    False
    181    False
    182    False
    183    False
    184    False
    185    False
    186    False
    187    False
    188    False
    189    False
    190    False
    191    False
    192    False
    193    False
    194    False
    195    False
    196    False
    197    False
    198    False
    199    False
    200    False
    201    False
    202    False
    203    False
    204    False
    205    False
    206    False
    207    False
    208    False
    209    False
    210    False
    211    False
    212    False
    213    False
    214    False
    215    False
    216    False
    217    False
    218    False
    219    False
    220    False
    221    False
    222    False
    223    False
    224     True
    225     True
    226     True
    227    False
    228    False
    229     True
    230    False
    231    False
    232     True
    233    False
    234     True
    235     True
    236    False
    237     True
    238    False
    239    False
    240    False
    241    False
    242     True
    243    False
    244    False
    245    False
    246     True
    247    False
    248    False
    249    False
    250    False
    251     True
    252    False
    253     True
    254     True
    255    False
    256    False
    257    False
    258    False
    259    False
    260     True
    261     True
    262    False
    263    False
    264    False
    265    False
    266    False
    267    False
    268    False
    269     True
    270    False
    271    False
    272     True
    273    False
    274     True
    275     True
    276    False
    277    False
    278    False
    279     True
    280     True
    281     True
    282     True
    283    False
    284    False
    285    False
    286    False
    287    False
    288    False
    289    False
    290    False
    291    False
    292     True
    293    False
    294    False
    295     True
    296    False
    297     True
    298    False
    299    False
    300    False
    301    False
    302    False
    303    False
    304    False
    305    False
    306    False
    307    False
    308    False
    309    False
    310    False
    311    False
    312    False
    313    False
    314    False
    315    False
    316    False
    317    False
    318    False
    319    False
    320    False
    321    False
    322    False
    323    False
    324    False
    325    False
    326    False
    327    False
    328    False
    329    False
    330     True
    331     True
    332    False
    333    False
    334    False
    335    False
    336    False
    337    False
    338    False
    339    False
    340    False
    341    False
    342    False
    343     True
    344     True
    345    False
    346    False
    347    False
    348     True
    349    False
    350    False
    351    False
    352    False
    353    False
    354    False
    355    False
    356    False
    357    False
    358     True
    359    False
    360    False
    361    False
    362    False
    363    False
    364    False
    365    False
    366    False
    367    False
    368    False
    369     True
    370    False
    371    False
    372     True
    373    False
    374    False
    375     True
    376    False
    377     True
    378    False
    379    False
    380    False
    381    False
    382    False
    383     True
    384    False
    385    False
    386     True
    387    False
    388    False
    389     True
    390    False
    391    False
    392    False
    393    False
    394    False
    395    False
    396    False
    397    False
    398    False
    399    False
    400    False
    401    False
    402    False
    403    False
    404     True
    405    False
    406    False
    407    False
    408    False
    409    False
    410    False
    411    False
    412    False
    413     True
    414    False
    415    False
    416    False
    417    False
    418    False
    419    False
    420    False
    421    False
    422    False
    423    False
    424    False
    425    False
    426    False
    427    False
    428    False
    429    False
    430    False
    431     True
    432     True
    433     True
    434    False
    435    False
    436    False
    437    False
    438    False
    439    False
    440    False
    441     True
    442    False
    443     True
    444    False
    445    False
    446    False
    447    False
    448    False
    449    False
    450    False
    451    False
    452    False
    453     True
    454    False
    455    False
    456    False
    457    False
    458    False
    459    False
    460     True
    461     True
    462    False
    463     True
    464     True
    465    False
    466     True
    467     True
    468    False
    469    False
    470    False
    471     True
    472     True
    473    False
    474    False
    475     True
    476    False
    477    False
    478     True
    479    False
    480    False
    481    False
    482     True
    483    False
    484    False
    485    False
    486    False
    487    False
    488    False
    489    False
    490    False
    491    False
    492     True
    493     True
    494    False
    495     True
    496    False
    497    False
    498     True
    499    False
    500    False
    501     True
    502    False
    503     True
    504    False
    505     True
    506    False
    507    False
    508    False
    509    False
    510    False
    511    False
    512     True
    513    False
    514    False
    515    False
    516    False
    517    False
    518    False
    519    False
    520    False
    521    False
    522    False
    523    False
    524     True
    525    False
    526    False
    527    False
    528    False
    529    False
    530     True
    531    False
    532     True
    533     True
    534    False
    535     True
    536     True
    537     True
    538     True
    539     True
    540    False
    541    False
    542     True
    543     True
    544    False
    545    False
    546     True
    547    False
    548    False
    549    False
    550     True
    551    False
    552    False
    553    False
    554    False
    555    False
    556    False
    557    False
    558    False
    559     True
    560     True
    561     True
    562    False
    563     True
    564     True
    565     True
    566    False
    567    False
    568    False
    569     True
    570    False
    571    False
    572    False
    573    False
    574    False
    575    False
    576     True
    577     True
    578    False
    579     True
    580    False
    581     True
    582     True
    583    False
    584    False
    585    False
    586     True
    587    False
    588     True
    589     True
    590     True
    591     True
    592    False
    593    False
    594    False
    595     True
    596    False
    597     True
    598     True
    599    False
    600    False
    601     True
    602     True
    603    False
    604    False
    605     True
    606    False
    607    False
    608     True
    609    False
    610    False
    611     True
    612    False
    613     True
    614    False
    615    False
    616    False
    617    False
    618     True
    619    False
    620    False
    621    False
    622    False
    623    False
    624    False
    625    False
    626    False
    627    False
    628    False
    629     True
    630    False
    631     True
    632    False
    633    False
    634    False
    635    False
    636    False
    637    False
    638    False
    639    False
    640     True
    641    False
    642    False
    643    False
    644    False
    645    False
    646    False
    647    False
    648    False
    649    False
    650    False
    651    False
    652    False
    653    False
    654    False
    655     True
    656    False
    657    False
    658    False
    659    False
    660    False
    661    False
    662     True
    663    False
    664    False
    665    False
    666    False
    667    False
    668     True
    669    False
    670     True
    671    False
    672    False
    673    False
    674    False
    675     True
    676    False
    677     True
    678    False
    679     True
    680     True
    681     True
    682     True
    683    False
    684    False
    685    False
    686     True
    687    False
    688    False
    689     True
    690    False
    691     True
    692    False
    693    False
    694    False
    695     True
    696    False
    697    False
    698    False
    699     True
    700     True
    701     True
    702     True
    703     True
    704     True
    705    False
    706    False
    707    False
    708    False
    709    False
    710    False
    711    False
    712    False
    713    False
    714     True
    715     True
    716     True
    717    False
    718    False
    719    False
    720    False
    721    False
    722    False
    723     True
    724    False
    725     True
    726    False
    727    False
    728    False
    729    False
    730    False
    731    False
    732    False
    733     True
    734    False
    735    False
    736     True
    737    False
    738     True
    739    False
    740    False
    741    False
    742    False
    743    False
    744    False
    745     True
    746    False
    747     True
    748     True
    749    False
    750    False
    751     True
    752    False
    753    False
    754    False
    755    False
    756    False
    757     True
    758    False
    759     True
    760    False
    761    False
    762    False
    763    False
    764    False
    765    False
    766    False
    767    False
    768     True
    769    False
    770    False
    771     True
    772     True
    773    False
    774     True
    775    False
    776    False
    777    False
    778    False
    779    False
    780    False
    781    False
    782    False
    783    False
    784    False
    785    False
    786    False
    787    False
    788     True
    789    False
    790    False
    791     True
    792    False
    793    False
    794    False
    795    False
    796    False
    797    False
    798    False
    799    False
    800    False
    801    False
    802     True
    803    False
    804    False
    805    False
    806     True
    807    False
    808    False
    809    False
    810    False
    811    False
    812    False
    813     True
    814     True
    815    False
    816    False
    817    False
    818    False
    819     True
    820    False
    821    False
    822    False
    823    False
    824    False
    825    False
    826    False
    827    False
    828    False
    829    False
    830    False
    831    False
    832    False
    833    False
    834    False
    835    False
    836    False
    837    False
    838    False
    839    False
    840    False
    841    False
    842     True
    843    False
    844    False
    845    False
    846    False
    847    False
    848    False
    849    False
    850    False
    851    False
    852    False
    853    False
    854     True
    855    False
    856    False
    857    False
    858    False
    859     True
    860    False
    861    False
    862    False
    863    False
    864     True
    865     True
    866    False
    867    False
    868    False
    869    False
    870     True
    871    False
    872    False
    873     True
    874     True
    875    False
    876    False
    877    False
    878    False
    879     True
    880    False
    881    False
    882    False
    883    False
    884     True
    885    False
    886    False
    887    False
    888    False
    889    False
    890    False
    891    False
    892     True
    893    False
    894    False
    895    False
    896    False
    897     True
    898    False
    899    False
    900    False
    901    False
    902    False
    903    False
    904    False
    905     True
    906    False
    907    False
    908    False
    909    False
    910    False
    911    False
    912    False
    913    False
    914    False
    915     True
    916     True
    917    False
    918    False
    919     True
    920     True
    921    False
    922    False
    923    False
    924     True
    925    False
    926    False
    927    False
    928    False
    929    False
    930    False
    931    False
    932    False
    933    False
    934    False
    935    False
    936    False
    937    False
    Name: any_bll_ge5, dtype: bool
    Unique values in y column: [0 1]



```python
# create classifier y column
bllpercentcutoff = 3.999
#bllpercentcutoff = 0

dflcurr = dfl.copy()
# remove zctas with less than 100 children below 6:
dflcurr = dflcurr.loc[dflcurr['children_under6__CLPPP']>100,:]

# add y column
dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'perc_bll_ge5__CLPPP']
dflcurr.loc[dflcurr['any_bll_ge5']<=bllpercentcutoff,'any_bll_ge5'] = 0
dflcurr.loc[dflcurr['any_bll_ge5']>bllpercentcutoff,'any_bll_ge5'] = 1
dflcurr.loc[:,'any_bll_ge5'] = dflcurr.loc[:,'any_bll_ge5'].astype(int)
print "Unique values in y column: %s" % dflcurr['any_bll_ge5'].unique()
#dflcurr = dfl.loc[dfl['perc_bll_ge5__CLPPP']>0,:]

# drop children under 6
# dflcurr = dflcurr.drop('children_under6__CLPPP', axis=1)
# dfmcurr = dfm.loc[dfm['outcode']!='children_under6__CLPPP',:]
#assert all(dfmcurr['outcode'].values==dflcurr.columns)

# build model
y_name = 'any_bll_ge5'
modelType = 'randomforestclassifier'
MLtype = 'classification'
model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name = lt.build_ML(dflcurr, dfm, y_name=y_name, modelType=modelType, MLtype=MLtype)

```

    Unique values in y column: [0 1]
    number of features in X: 707
    True
    shape of X:  (722, 707)
    shape of y:  (722,)
    Features with nonzero importances:  707



![png](testnotebook1_files/testnotebook1_44_1.png)



![png](testnotebook1_files/testnotebook1_44_2.png)


#### try multiple runs of same model, variance in r^2 

what is the score?? not r^2?!


```python
dflcurr.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip</th>
      <th>zcta</th>
      <th>perc_pre1950_housing__CLPPP</th>
      <th>children_under6__CLPPP</th>
      <th>children_tested__CLPPP</th>
      <th>perc_tested__CLPPP</th>
      <th>bll_lt5__CLPPP</th>
      <th>bll_5to9__CLPPP</th>
      <th>capillary_ge10__CLPPP</th>
      <th>venous_10to19__CLPPP</th>
      <th>venous_20to44__CLPPP</th>
      <th>venous_ge45__CLPPP</th>
      <th>blltot_ge5__CLPPP</th>
      <th>blltot_ge10__CLPPP</th>
      <th>perc_bll_ge5__CLPPP</th>
      <th>perc_bll_ge10__CLPPP</th>
      <th>HC03_EST_VC01__S1701</th>
      <th>HC03_EST_VC03__S1701</th>
      <th>HC03_EST_VC04__S1701</th>
      <th>HC03_EST_VC05__S1701</th>
      <th>HC03_EST_VC06__S1701</th>
      <th>HC03_EST_VC09__S1701</th>
      <th>HC03_EST_VC10__S1701</th>
      <th>HC03_EST_VC13__S1701</th>
      <th>HC03_EST_VC14__S1701</th>
      <th>HC03_EST_VC20__S1701</th>
      <th>HC03_EST_VC22__S1701</th>
      <th>HC03_EST_VC23__S1701</th>
      <th>HC03_EST_VC26__S1701</th>
      <th>HC03_EST_VC27__S1701</th>
      <th>HC03_EST_VC28__S1701</th>
      <th>HC03_EST_VC29__S1701</th>
      <th>HC03_EST_VC30__S1701</th>
      <th>HC03_EST_VC33__S1701</th>
      <th>HC03_EST_VC34__S1701</th>
      <th>HC03_EST_VC35__S1701</th>
      <th>HC03_EST_VC36__S1701</th>
      <th>HC03_EST_VC37__S1701</th>
      <th>HC03_EST_VC38__S1701</th>
      <th>HC03_EST_VC39__S1701</th>
      <th>HC03_EST_VC42__S1701</th>
      <th>HC03_EST_VC43__S1701</th>
      <th>HC03_EST_VC44__S1701</th>
      <th>HC03_EST_VC45__S1701</th>
      <th>HC03_EST_VC54__S1701</th>
      <th>HC03_EST_VC55__S1701</th>
      <th>HC03_EST_VC56__S1701</th>
      <th>HC03_EST_VC60__S1701</th>
      <th>HC03_EST_VC61__S1701</th>
      <th>HC03_EST_VC62__S1701</th>
      <th>HC02_EST_VC01__S1702</th>
      <th>HC04_EST_VC01__S1702</th>
      <th>HC06_EST_VC01__S1702</th>
      <th>HC02_EST_VC02__S1702</th>
      <th>HC04_EST_VC02__S1702</th>
      <th>HC06_EST_VC02__S1702</th>
      <th>HC02_EST_VC06__S1702</th>
      <th>HC04_EST_VC06__S1702</th>
      <th>HC06_EST_VC06__S1702</th>
      <th>HC02_EST_VC07__S1702</th>
      <th>HC04_EST_VC07__S1702</th>
      <th>HC06_EST_VC07__S1702</th>
      <th>HC02_EST_VC16__S1702</th>
      <th>HC04_EST_VC16__S1702</th>
      <th>HC06_EST_VC16__S1702</th>
      <th>HC02_EST_VC18__S1702</th>
      <th>HC04_EST_VC18__S1702</th>
      <th>HC06_EST_VC18__S1702</th>
      <th>HC02_EST_VC19__S1702</th>
      <th>HC04_EST_VC19__S1702</th>
      <th>HC06_EST_VC19__S1702</th>
      <th>HC02_EST_VC21__S1702</th>
      <th>HC04_EST_VC21__S1702</th>
      <th>HC06_EST_VC21__S1702</th>
      <th>HC02_EST_VC23__S1702</th>
      <th>HC04_EST_VC23__S1702</th>
      <th>HC06_EST_VC23__S1702</th>
      <th>HC02_EST_VC24__S1702</th>
      <th>HC04_EST_VC24__S1702</th>
      <th>HC06_EST_VC24__S1702</th>
      <th>HC02_EST_VC27__S1702</th>
      <th>HC04_EST_VC27__S1702</th>
      <th>HC02_EST_VC28__S1702</th>
      <th>HC04_EST_VC28__S1702</th>
      <th>HC06_EST_VC28__S1702</th>
      <th>HC02_EST_VC29__S1702</th>
      <th>HC04_EST_VC29__S1702</th>
      <th>HC06_EST_VC29__S1702</th>
      <th>HC02_EST_VC30__S1702</th>
      <th>HC04_EST_VC30__S1702</th>
      <th>HC02_EST_VC33__S1702</th>
      <th>HC04_EST_VC33__S1702</th>
      <th>HC06_EST_VC33__S1702</th>
      <th>HC02_EST_VC34__S1702</th>
      <th>HC04_EST_VC34__S1702</th>
      <th>HC06_EST_VC34__S1702</th>
      <th>HC02_EST_VC35__S1702</th>
      <th>HC04_EST_VC35__S1702</th>
      <th>HC02_EST_VC39__S1702</th>
      <th>HC04_EST_VC39__S1702</th>
      <th>HC06_EST_VC39__S1702</th>
      <th>HC02_EST_VC40__S1702</th>
      <th>HC04_EST_VC40__S1702</th>
      <th>HC06_EST_VC40__S1702</th>
      <th>HC02_EST_VC41__S1702</th>
      <th>HC04_EST_VC41__S1702</th>
      <th>HC02_EST_VC45__S1702</th>
      <th>HC04_EST_VC45__S1702</th>
      <th>HC06_EST_VC45__S1702</th>
      <th>HC02_EST_VC46__S1702</th>
      <th>HC04_EST_VC46__S1702</th>
      <th>HC06_EST_VC46__S1702</th>
      <th>HC02_EST_VC47__S1702</th>
      <th>HC04_EST_VC47__S1702</th>
      <th>HC06_EST_VC47__S1702</th>
      <th>HC02_EST_VC48__S1702</th>
      <th>HC04_EST_VC48__S1702</th>
      <th>HC02_EST_VC01__S1401</th>
      <th>HC03_EST_VC01__S1401</th>
      <th>HC02_EST_VC02__S1401</th>
      <th>HC03_EST_VC02__S1401</th>
      <th>HC02_EST_VC03__S1401</th>
      <th>HC03_EST_VC03__S1401</th>
      <th>HC02_EST_VC04__S1401</th>
      <th>HC03_EST_VC04__S1401</th>
      <th>HC02_EST_VC05__S1401</th>
      <th>HC03_EST_VC05__S1401</th>
      <th>HC02_EST_VC06__S1401</th>
      <th>HC03_EST_VC06__S1401</th>
      <th>HC02_EST_VC07__S1401</th>
      <th>HC03_EST_VC07__S1401</th>
      <th>HC02_EST_VC08__S1401</th>
      <th>HC03_EST_VC08__S1401</th>
      <th>HC02_EST_VC09__S1401</th>
      <th>HC03_EST_VC09__S1401</th>
      <th>HC01_EST_VC12__S1401</th>
      <th>HC02_EST_VC12__S1401</th>
      <th>HC03_EST_VC12__S1401</th>
      <th>HC01_EST_VC13__S1401</th>
      <th>HC02_EST_VC13__S1401</th>
      <th>HC03_EST_VC13__S1401</th>
      <th>HC01_EST_VC14__S1401</th>
      <th>HC02_EST_VC14__S1401</th>
      <th>HC03_EST_VC14__S1401</th>
      <th>HC01_EST_VC15__S1401</th>
      <th>HC02_EST_VC15__S1401</th>
      <th>HC03_EST_VC15__S1401</th>
      <th>HC01_EST_VC16__S1401</th>
      <th>HC02_EST_VC16__S1401</th>
      <th>HC03_EST_VC16__S1401</th>
      <th>HC01_EST_VC17__S1401</th>
      <th>HC02_EST_VC17__S1401</th>
      <th>HC03_EST_VC17__S1401</th>
      <th>HC01_EST_VC18__S1401</th>
      <th>HC02_EST_VC18__S1401</th>
      <th>HC03_EST_VC18__S1401</th>
      <th>HC01_EST_VC19__S1401</th>
      <th>HC02_EST_VC19__S1401</th>
      <th>HC03_EST_VC19__S1401</th>
      <th>HC02_EST_VC22__S1401</th>
      <th>HC03_EST_VC22__S1401</th>
      <th>HC02_EST_VC24__S1401</th>
      <th>HC03_EST_VC24__S1401</th>
      <th>HC02_EST_VC26__S1401</th>
      <th>HC03_EST_VC26__S1401</th>
      <th>HC02_EST_VC29__S1401</th>
      <th>HC03_EST_VC29__S1401</th>
      <th>HC02_EST_VC31__S1401</th>
      <th>HC03_EST_VC31__S1401</th>
      <th>HC02_EST_VC33__S1401</th>
      <th>HC03_EST_VC33__S1401</th>
      <th>HC01_EST_VC16__S1501</th>
      <th>HC02_EST_VC16__S1501</th>
      <th>HC03_EST_VC16__S1501</th>
      <th>HC01_EST_VC17__S1501</th>
      <th>HC02_EST_VC17__S1501</th>
      <th>HC03_EST_VC17__S1501</th>
      <th>HC02_EST_VC01__S1601</th>
      <th>HC03_EST_VC01__S1601</th>
      <th>HC02_EST_VC03__S1601</th>
      <th>HC03_EST_VC03__S1601</th>
      <th>HC02_EST_VC04__S1601</th>
      <th>HC03_EST_VC04__S1601</th>
      <th>HC02_EST_VC05__S1601</th>
      <th>HC03_EST_VC05__S1601</th>
      <th>HC02_EST_VC10__S1601</th>
      <th>HC03_EST_VC10__S1601</th>
      <th>HC02_EST_VC12__S1601</th>
      <th>HC03_EST_VC12__S1601</th>
      <th>HC02_EST_VC14__S1601</th>
      <th>HC03_EST_VC14__S1601</th>
      <th>HC02_EST_VC16__S1601</th>
      <th>HC03_EST_VC16__S1601</th>
      <th>HC02_EST_VC28__S1601</th>
      <th>HC03_EST_VC28__S1601</th>
      <th>HC02_EST_VC30__S1601</th>
      <th>HC03_EST_VC30__S1601</th>
      <th>HC02_EST_VC31__S1601</th>
      <th>HC03_EST_VC31__S1601</th>
      <th>HC02_EST_VC32__S1601</th>
      <th>HC03_EST_VC32__S1601</th>
      <th>HC03_EST_VC01__S2701</th>
      <th>HC05_EST_VC01__S2701</th>
      <th>HC03_EST_VC04__S2701</th>
      <th>HC03_EST_VC05__S2701</th>
      <th>HC03_EST_VC06__S2701</th>
      <th>HC03_EST_VC08__S2701</th>
      <th>HC03_EST_VC11__S2701</th>
      <th>HC03_EST_VC12__S2701</th>
      <th>HC03_EST_VC15__S2701</th>
      <th>HC03_EST_VC16__S2701</th>
      <th>HC03_EST_VC22__S2701</th>
      <th>HC03_EST_VC24__S2701</th>
      <th>HC03_EST_VC25__S2701</th>
      <th>HC03_EST_VC28__S2701</th>
      <th>HC03_EST_VC29__S2701</th>
      <th>HC03_EST_VC30__S2701</th>
      <th>HC03_EST_VC31__S2701</th>
      <th>HC03_EST_VC34__S2701</th>
      <th>HC03_EST_VC35__S2701</th>
      <th>HC03_EST_VC36__S2701</th>
      <th>HC03_EST_VC37__S2701</th>
      <th>HC03_EST_VC38__S2701</th>
      <th>HC03_EST_VC41__S2701</th>
      <th>HC03_EST_VC42__S2701</th>
      <th>HC03_EST_VC43__S2701</th>
      <th>HC03_EST_VC44__S2701</th>
      <th>HC03_EST_VC45__S2701</th>
      <th>HC03_EST_VC48__S2701</th>
      <th>HC03_EST_VC49__S2701</th>
      <th>HC03_EST_VC50__S2701</th>
      <th>HC03_EST_VC51__S2701</th>
      <th>HC03_EST_VC54__S2701</th>
      <th>HC03_EST_VC55__S2701</th>
      <th>HC03_EST_VC56__S2701</th>
      <th>HC03_EST_VC57__S2701</th>
      <th>HC03_EST_VC58__S2701</th>
      <th>HC03_EST_VC59__S2701</th>
      <th>HC03_EST_VC62__S2701</th>
      <th>HC03_EST_VC63__S2701</th>
      <th>HC03_EST_VC64__S2701</th>
      <th>HC03_EST_VC65__S2701</th>
      <th>HC05_EST_VC68__S2701</th>
      <th>HC05_EST_VC69__S2701</th>
      <th>HC05_EST_VC70__S2701</th>
      <th>HC05_EST_VC71__S2701</th>
      <th>HC05_EST_VC72__S2701</th>
      <th>HC05_EST_VC73__S2701</th>
      <th>HC05_EST_VC74__S2701</th>
      <th>HC05_EST_VC75__S2701</th>
      <th>HC05_EST_VC76__S2701</th>
      <th>HC05_EST_VC77__S2701</th>
      <th>HC05_EST_VC78__S2701</th>
      <th>HC05_EST_VC79__S2701</th>
      <th>HC05_EST_VC80__S2701</th>
      <th>HC05_EST_VC81__S2701</th>
      <th>HC05_EST_VC82__S2701</th>
      <th>HC05_EST_VC83__S2701</th>
      <th>HC05_EST_VC84__S2701</th>
      <th>HC03_VC04__DP04</th>
      <th>HC03_VC05__DP04</th>
      <th>HC03_VC14__DP04</th>
      <th>HC03_VC15__DP04</th>
      <th>HC03_VC16__DP04</th>
      <th>HC03_VC17__DP04</th>
      <th>HC03_VC18__DP04</th>
      <th>HC03_VC19__DP04</th>
      <th>HC03_VC20__DP04</th>
      <th>HC03_VC21__DP04</th>
      <th>HC03_VC22__DP04</th>
      <th>HC03_VC27__DP04</th>
      <th>HC03_VC28__DP04</th>
      <th>HC03_VC29__DP04</th>
      <th>HC03_VC30__DP04</th>
      <th>HC03_VC31__DP04</th>
      <th>HC03_VC32__DP04</th>
      <th>HC03_VC33__DP04</th>
      <th>HC03_VC34__DP04</th>
      <th>HC03_VC35__DP04</th>
      <th>HC03_VC40__DP04</th>
      <th>HC03_VC41__DP04</th>
      <th>HC03_VC42__DP04</th>
      <th>HC03_VC43__DP04</th>
      <th>HC03_VC44__DP04</th>
      <th>HC03_VC45__DP04</th>
      <th>HC03_VC46__DP04</th>
      <th>HC03_VC47__DP04</th>
      <th>HC03_VC48__DP04</th>
      <th>HC03_VC54__DP04</th>
      <th>HC03_VC55__DP04</th>
      <th>HC03_VC56__DP04</th>
      <th>HC03_VC57__DP04</th>
      <th>HC03_VC58__DP04</th>
      <th>HC03_VC59__DP04</th>
      <th>HC03_VC64__DP04</th>
      <th>HC03_VC65__DP04</th>
      <th>HC03_VC74__DP04</th>
      <th>HC03_VC75__DP04</th>
      <th>HC03_VC76__DP04</th>
      <th>HC03_VC77__DP04</th>
      <th>HC03_VC78__DP04</th>
      <th>HC03_VC79__DP04</th>
      <th>HC03_VC84__DP04</th>
      <th>HC03_VC85__DP04</th>
      <th>HC03_VC86__DP04</th>
      <th>HC03_VC87__DP04</th>
      <th>HC03_VC92__DP04</th>
      <th>HC03_VC93__DP04</th>
      <th>HC03_VC94__DP04</th>
      <th>HC03_VC95__DP04</th>
      <th>HC03_VC96__DP04</th>
      <th>HC03_VC97__DP04</th>
      <th>HC03_VC98__DP04</th>
      <th>HC03_VC99__DP04</th>
      <th>HC03_VC100__DP04</th>
      <th>HC03_VC105__DP04</th>
      <th>HC03_VC106__DP04</th>
      <th>HC03_VC107__DP04</th>
      <th>HC03_VC112__DP04</th>
      <th>HC03_VC113__DP04</th>
      <th>HC03_VC114__DP04</th>
      <th>HC03_VC119__DP04</th>
      <th>HC03_VC120__DP04</th>
      <th>HC03_VC121__DP04</th>
      <th>HC03_VC122__DP04</th>
      <th>HC03_VC123__DP04</th>
      <th>HC03_VC124__DP04</th>
      <th>HC03_VC125__DP04</th>
      <th>HC03_VC126__DP04</th>
      <th>HC03_VC132__DP04</th>
      <th>HC03_VC133__DP04</th>
      <th>HC03_VC138__DP04</th>
      <th>HC03_VC139__DP04</th>
      <th>HC03_VC140__DP04</th>
      <th>HC03_VC141__DP04</th>
      <th>HC03_VC142__DP04</th>
      <th>HC03_VC143__DP04</th>
      <th>HC03_VC144__DP04</th>
      <th>HC03_VC148__DP04</th>
      <th>HC03_VC149__DP04</th>
      <th>HC03_VC150__DP04</th>
      <th>HC03_VC151__DP04</th>
      <th>HC03_VC152__DP04</th>
      <th>HC03_VC158__DP04</th>
      <th>HC03_VC159__DP04</th>
      <th>HC03_VC160__DP04</th>
      <th>HC03_VC161__DP04</th>
      <th>HC03_VC162__DP04</th>
      <th>HC03_VC168__DP04</th>
      <th>HC03_VC169__DP04</th>
      <th>HC03_VC170__DP04</th>
      <th>HC03_VC171__DP04</th>
      <th>HC03_VC172__DP04</th>
      <th>HC03_VC173__DP04</th>
      <th>HC03_VC174__DP04</th>
      <th>HC03_VC182__DP04</th>
      <th>HC03_VC183__DP04</th>
      <th>HC03_VC184__DP04</th>
      <th>HC03_VC185__DP04</th>
      <th>HC03_VC186__DP04</th>
      <th>HC03_VC187__DP04</th>
      <th>HC03_VC188__DP04</th>
      <th>HC03_VC197__DP04</th>
      <th>HC03_VC198__DP04</th>
      <th>HC03_VC199__DP04</th>
      <th>HC03_VC200__DP04</th>
      <th>HC03_VC201__DP04</th>
      <th>HC03_VC202__DP04</th>
      <th>perc_pre1950_housing__CLPPP_close</th>
      <th>HC03_EST_VC01__S1701_close</th>
      <th>HC03_EST_VC03__S1701_close</th>
      <th>HC03_EST_VC04__S1701_close</th>
      <th>HC03_EST_VC05__S1701_close</th>
      <th>HC03_EST_VC06__S1701_close</th>
      <th>HC03_EST_VC09__S1701_close</th>
      <th>HC03_EST_VC10__S1701_close</th>
      <th>HC03_EST_VC13__S1701_close</th>
      <th>HC03_EST_VC14__S1701_close</th>
      <th>HC03_EST_VC20__S1701_close</th>
      <th>HC03_EST_VC22__S1701_close</th>
      <th>HC03_EST_VC23__S1701_close</th>
      <th>HC03_EST_VC26__S1701_close</th>
      <th>HC03_EST_VC27__S1701_close</th>
      <th>HC03_EST_VC28__S1701_close</th>
      <th>HC03_EST_VC29__S1701_close</th>
      <th>HC03_EST_VC30__S1701_close</th>
      <th>HC03_EST_VC33__S1701_close</th>
      <th>HC03_EST_VC34__S1701_close</th>
      <th>HC03_EST_VC35__S1701_close</th>
      <th>HC03_EST_VC36__S1701_close</th>
      <th>HC03_EST_VC37__S1701_close</th>
      <th>HC03_EST_VC38__S1701_close</th>
      <th>HC03_EST_VC39__S1701_close</th>
      <th>HC03_EST_VC42__S1701_close</th>
      <th>HC03_EST_VC43__S1701_close</th>
      <th>HC03_EST_VC44__S1701_close</th>
      <th>HC03_EST_VC45__S1701_close</th>
      <th>HC03_EST_VC54__S1701_close</th>
      <th>HC03_EST_VC55__S1701_close</th>
      <th>HC03_EST_VC56__S1701_close</th>
      <th>HC03_EST_VC60__S1701_close</th>
      <th>HC03_EST_VC61__S1701_close</th>
      <th>HC03_EST_VC62__S1701_close</th>
      <th>HC02_EST_VC01__S1702_close</th>
      <th>HC04_EST_VC01__S1702_close</th>
      <th>HC06_EST_VC01__S1702_close</th>
      <th>HC02_EST_VC02__S1702_close</th>
      <th>HC04_EST_VC02__S1702_close</th>
      <th>HC06_EST_VC02__S1702_close</th>
      <th>HC02_EST_VC06__S1702_close</th>
      <th>HC04_EST_VC06__S1702_close</th>
      <th>HC06_EST_VC06__S1702_close</th>
      <th>HC02_EST_VC07__S1702_close</th>
      <th>HC04_EST_VC07__S1702_close</th>
      <th>HC06_EST_VC07__S1702_close</th>
      <th>HC02_EST_VC16__S1702_close</th>
      <th>HC04_EST_VC16__S1702_close</th>
      <th>HC06_EST_VC16__S1702_close</th>
      <th>HC02_EST_VC18__S1702_close</th>
      <th>HC04_EST_VC18__S1702_close</th>
      <th>HC06_EST_VC18__S1702_close</th>
      <th>HC02_EST_VC19__S1702_close</th>
      <th>HC04_EST_VC19__S1702_close</th>
      <th>HC06_EST_VC19__S1702_close</th>
      <th>HC02_EST_VC21__S1702_close</th>
      <th>HC04_EST_VC21__S1702_close</th>
      <th>HC06_EST_VC21__S1702_close</th>
      <th>HC02_EST_VC23__S1702_close</th>
      <th>HC04_EST_VC23__S1702_close</th>
      <th>HC06_EST_VC23__S1702_close</th>
      <th>HC02_EST_VC24__S1702_close</th>
      <th>HC04_EST_VC24__S1702_close</th>
      <th>HC06_EST_VC24__S1702_close</th>
      <th>HC02_EST_VC27__S1702_close</th>
      <th>HC04_EST_VC27__S1702_close</th>
      <th>HC02_EST_VC28__S1702_close</th>
      <th>HC04_EST_VC28__S1702_close</th>
      <th>HC06_EST_VC28__S1702_close</th>
      <th>HC02_EST_VC29__S1702_close</th>
      <th>HC04_EST_VC29__S1702_close</th>
      <th>HC06_EST_VC29__S1702_close</th>
      <th>HC02_EST_VC30__S1702_close</th>
      <th>HC04_EST_VC30__S1702_close</th>
      <th>HC02_EST_VC33__S1702_close</th>
      <th>HC04_EST_VC33__S1702_close</th>
      <th>HC06_EST_VC33__S1702_close</th>
      <th>HC02_EST_VC34__S1702_close</th>
      <th>HC04_EST_VC34__S1702_close</th>
      <th>HC06_EST_VC34__S1702_close</th>
      <th>HC02_EST_VC35__S1702_close</th>
      <th>HC04_EST_VC35__S1702_close</th>
      <th>HC02_EST_VC39__S1702_close</th>
      <th>HC04_EST_VC39__S1702_close</th>
      <th>HC06_EST_VC39__S1702_close</th>
      <th>HC02_EST_VC40__S1702_close</th>
      <th>HC04_EST_VC40__S1702_close</th>
      <th>HC06_EST_VC40__S1702_close</th>
      <th>HC02_EST_VC41__S1702_close</th>
      <th>HC04_EST_VC41__S1702_close</th>
      <th>HC02_EST_VC45__S1702_close</th>
      <th>HC04_EST_VC45__S1702_close</th>
      <th>HC06_EST_VC45__S1702_close</th>
      <th>HC02_EST_VC46__S1702_close</th>
      <th>HC04_EST_VC46__S1702_close</th>
      <th>HC06_EST_VC46__S1702_close</th>
      <th>HC02_EST_VC47__S1702_close</th>
      <th>HC04_EST_VC47__S1702_close</th>
      <th>HC06_EST_VC47__S1702_close</th>
      <th>HC02_EST_VC48__S1702_close</th>
      <th>HC04_EST_VC48__S1702_close</th>
      <th>HC02_EST_VC01__S1401_close</th>
      <th>HC03_EST_VC01__S1401_close</th>
      <th>HC02_EST_VC02__S1401_close</th>
      <th>HC03_EST_VC02__S1401_close</th>
      <th>HC02_EST_VC03__S1401_close</th>
      <th>HC03_EST_VC03__S1401_close</th>
      <th>HC02_EST_VC04__S1401_close</th>
      <th>HC03_EST_VC04__S1401_close</th>
      <th>HC02_EST_VC05__S1401_close</th>
      <th>HC03_EST_VC05__S1401_close</th>
      <th>HC02_EST_VC06__S1401_close</th>
      <th>HC03_EST_VC06__S1401_close</th>
      <th>HC02_EST_VC07__S1401_close</th>
      <th>HC03_EST_VC07__S1401_close</th>
      <th>HC02_EST_VC08__S1401_close</th>
      <th>HC03_EST_VC08__S1401_close</th>
      <th>HC02_EST_VC09__S1401_close</th>
      <th>HC03_EST_VC09__S1401_close</th>
      <th>HC01_EST_VC12__S1401_close</th>
      <th>HC02_EST_VC12__S1401_close</th>
      <th>HC03_EST_VC12__S1401_close</th>
      <th>HC01_EST_VC13__S1401_close</th>
      <th>HC02_EST_VC13__S1401_close</th>
      <th>HC03_EST_VC13__S1401_close</th>
      <th>HC01_EST_VC14__S1401_close</th>
      <th>HC02_EST_VC14__S1401_close</th>
      <th>HC03_EST_VC14__S1401_close</th>
      <th>HC01_EST_VC15__S1401_close</th>
      <th>HC02_EST_VC15__S1401_close</th>
      <th>HC03_EST_VC15__S1401_close</th>
      <th>HC01_EST_VC16__S1401_close</th>
      <th>HC02_EST_VC16__S1401_close</th>
      <th>HC03_EST_VC16__S1401_close</th>
      <th>HC01_EST_VC17__S1401_close</th>
      <th>HC02_EST_VC17__S1401_close</th>
      <th>HC03_EST_VC17__S1401_close</th>
      <th>HC01_EST_VC18__S1401_close</th>
      <th>HC02_EST_VC18__S1401_close</th>
      <th>HC03_EST_VC18__S1401_close</th>
      <th>HC01_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC19__S1401_close</th>
      <th>HC03_EST_VC19__S1401_close</th>
      <th>HC02_EST_VC22__S1401_close</th>
      <th>HC03_EST_VC22__S1401_close</th>
      <th>HC02_EST_VC24__S1401_close</th>
      <th>HC03_EST_VC24__S1401_close</th>
      <th>HC02_EST_VC26__S1401_close</th>
      <th>HC03_EST_VC26__S1401_close</th>
      <th>HC02_EST_VC29__S1401_close</th>
      <th>HC03_EST_VC29__S1401_close</th>
      <th>HC02_EST_VC31__S1401_close</th>
      <th>HC03_EST_VC31__S1401_close</th>
      <th>HC02_EST_VC33__S1401_close</th>
      <th>HC03_EST_VC33__S1401_close</th>
      <th>HC01_EST_VC16__S1501_close</th>
      <th>HC02_EST_VC16__S1501_close</th>
      <th>HC03_EST_VC16__S1501_close</th>
      <th>HC01_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC17__S1501_close</th>
      <th>HC03_EST_VC17__S1501_close</th>
      <th>HC02_EST_VC01__S1601_close</th>
      <th>HC03_EST_VC01__S1601_close</th>
      <th>HC02_EST_VC03__S1601_close</th>
      <th>HC03_EST_VC03__S1601_close</th>
      <th>HC02_EST_VC04__S1601_close</th>
      <th>HC03_EST_VC04__S1601_close</th>
      <th>HC02_EST_VC05__S1601_close</th>
      <th>HC03_EST_VC05__S1601_close</th>
      <th>HC02_EST_VC10__S1601_close</th>
      <th>HC03_EST_VC10__S1601_close</th>
      <th>HC02_EST_VC12__S1601_close</th>
      <th>HC03_EST_VC12__S1601_close</th>
      <th>HC02_EST_VC14__S1601_close</th>
      <th>HC03_EST_VC14__S1601_close</th>
      <th>HC02_EST_VC16__S1601_close</th>
      <th>HC03_EST_VC16__S1601_close</th>
      <th>HC02_EST_VC28__S1601_close</th>
      <th>HC03_EST_VC28__S1601_close</th>
      <th>HC02_EST_VC30__S1601_close</th>
      <th>HC03_EST_VC30__S1601_close</th>
      <th>HC02_EST_VC31__S1601_close</th>
      <th>HC03_EST_VC31__S1601_close</th>
      <th>HC02_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC32__S1601_close</th>
      <th>HC03_EST_VC01__S2701_close</th>
      <th>HC05_EST_VC01__S2701_close</th>
      <th>HC03_EST_VC04__S2701_close</th>
      <th>HC03_EST_VC05__S2701_close</th>
      <th>HC03_EST_VC06__S2701_close</th>
      <th>HC03_EST_VC08__S2701_close</th>
      <th>HC03_EST_VC11__S2701_close</th>
      <th>HC03_EST_VC12__S2701_close</th>
      <th>HC03_EST_VC15__S2701_close</th>
      <th>HC03_EST_VC16__S2701_close</th>
      <th>HC03_EST_VC22__S2701_close</th>
      <th>HC03_EST_VC24__S2701_close</th>
      <th>HC03_EST_VC25__S2701_close</th>
      <th>HC03_EST_VC28__S2701_close</th>
      <th>HC03_EST_VC29__S2701_close</th>
      <th>HC03_EST_VC30__S2701_close</th>
      <th>HC03_EST_VC31__S2701_close</th>
      <th>HC03_EST_VC34__S2701_close</th>
      <th>HC03_EST_VC35__S2701_close</th>
      <th>HC03_EST_VC36__S2701_close</th>
      <th>HC03_EST_VC37__S2701_close</th>
      <th>HC03_EST_VC38__S2701_close</th>
      <th>HC03_EST_VC41__S2701_close</th>
      <th>HC03_EST_VC42__S2701_close</th>
      <th>HC03_EST_VC43__S2701_close</th>
      <th>HC03_EST_VC44__S2701_close</th>
      <th>HC03_EST_VC45__S2701_close</th>
      <th>HC03_EST_VC48__S2701_close</th>
      <th>HC03_EST_VC49__S2701_close</th>
      <th>HC03_EST_VC50__S2701_close</th>
      <th>HC03_EST_VC51__S2701_close</th>
      <th>HC03_EST_VC54__S2701_close</th>
      <th>HC03_EST_VC55__S2701_close</th>
      <th>HC03_EST_VC56__S2701_close</th>
      <th>HC03_EST_VC57__S2701_close</th>
      <th>HC03_EST_VC58__S2701_close</th>
      <th>HC03_EST_VC59__S2701_close</th>
      <th>HC03_EST_VC62__S2701_close</th>
      <th>HC03_EST_VC63__S2701_close</th>
      <th>HC03_EST_VC64__S2701_close</th>
      <th>HC03_EST_VC65__S2701_close</th>
      <th>HC05_EST_VC68__S2701_close</th>
      <th>HC05_EST_VC69__S2701_close</th>
      <th>HC05_EST_VC70__S2701_close</th>
      <th>HC05_EST_VC71__S2701_close</th>
      <th>HC05_EST_VC72__S2701_close</th>
      <th>HC05_EST_VC73__S2701_close</th>
      <th>HC05_EST_VC74__S2701_close</th>
      <th>HC05_EST_VC75__S2701_close</th>
      <th>HC05_EST_VC76__S2701_close</th>
      <th>HC05_EST_VC77__S2701_close</th>
      <th>HC05_EST_VC78__S2701_close</th>
      <th>HC05_EST_VC79__S2701_close</th>
      <th>HC05_EST_VC80__S2701_close</th>
      <th>HC05_EST_VC81__S2701_close</th>
      <th>HC05_EST_VC82__S2701_close</th>
      <th>HC05_EST_VC83__S2701_close</th>
      <th>HC05_EST_VC84__S2701_close</th>
      <th>HC03_VC04__DP04_close</th>
      <th>HC03_VC05__DP04_close</th>
      <th>HC03_VC14__DP04_close</th>
      <th>HC03_VC15__DP04_close</th>
      <th>HC03_VC16__DP04_close</th>
      <th>HC03_VC17__DP04_close</th>
      <th>HC03_VC18__DP04_close</th>
      <th>HC03_VC19__DP04_close</th>
      <th>HC03_VC20__DP04_close</th>
      <th>HC03_VC21__DP04_close</th>
      <th>HC03_VC22__DP04_close</th>
      <th>HC03_VC27__DP04_close</th>
      <th>HC03_VC28__DP04_close</th>
      <th>HC03_VC29__DP04_close</th>
      <th>HC03_VC30__DP04_close</th>
      <th>HC03_VC31__DP04_close</th>
      <th>HC03_VC32__DP04_close</th>
      <th>HC03_VC33__DP04_close</th>
      <th>HC03_VC34__DP04_close</th>
      <th>HC03_VC35__DP04_close</th>
      <th>HC03_VC40__DP04_close</th>
      <th>HC03_VC41__DP04_close</th>
      <th>HC03_VC42__DP04_close</th>
      <th>HC03_VC43__DP04_close</th>
      <th>HC03_VC44__DP04_close</th>
      <th>HC03_VC45__DP04_close</th>
      <th>HC03_VC46__DP04_close</th>
      <th>HC03_VC47__DP04_close</th>
      <th>HC03_VC48__DP04_close</th>
      <th>HC03_VC54__DP04_close</th>
      <th>HC03_VC55__DP04_close</th>
      <th>HC03_VC56__DP04_close</th>
      <th>HC03_VC57__DP04_close</th>
      <th>HC03_VC58__DP04_close</th>
      <th>HC03_VC59__DP04_close</th>
      <th>HC03_VC64__DP04_close</th>
      <th>HC03_VC65__DP04_close</th>
      <th>HC03_VC74__DP04_close</th>
      <th>HC03_VC75__DP04_close</th>
      <th>HC03_VC76__DP04_close</th>
      <th>HC03_VC77__DP04_close</th>
      <th>HC03_VC78__DP04_close</th>
      <th>HC03_VC79__DP04_close</th>
      <th>HC03_VC84__DP04_close</th>
      <th>HC03_VC85__DP04_close</th>
      <th>HC03_VC86__DP04_close</th>
      <th>HC03_VC87__DP04_close</th>
      <th>HC03_VC92__DP04_close</th>
      <th>HC03_VC93__DP04_close</th>
      <th>HC03_VC94__DP04_close</th>
      <th>HC03_VC95__DP04_close</th>
      <th>HC03_VC96__DP04_close</th>
      <th>HC03_VC97__DP04_close</th>
      <th>HC03_VC98__DP04_close</th>
      <th>HC03_VC99__DP04_close</th>
      <th>HC03_VC100__DP04_close</th>
      <th>HC03_VC105__DP04_close</th>
      <th>HC03_VC106__DP04_close</th>
      <th>HC03_VC107__DP04_close</th>
      <th>HC03_VC112__DP04_close</th>
      <th>HC03_VC113__DP04_close</th>
      <th>HC03_VC114__DP04_close</th>
      <th>HC03_VC119__DP04_close</th>
      <th>HC03_VC120__DP04_close</th>
      <th>HC03_VC121__DP04_close</th>
      <th>HC03_VC122__DP04_close</th>
      <th>HC03_VC123__DP04_close</th>
      <th>HC03_VC124__DP04_close</th>
      <th>HC03_VC125__DP04_close</th>
      <th>HC03_VC126__DP04_close</th>
      <th>HC03_VC132__DP04_close</th>
      <th>HC03_VC133__DP04_close</th>
      <th>HC03_VC138__DP04_close</th>
      <th>HC03_VC139__DP04_close</th>
      <th>HC03_VC140__DP04_close</th>
      <th>HC03_VC141__DP04_close</th>
      <th>HC03_VC142__DP04_close</th>
      <th>HC03_VC143__DP04_close</th>
      <th>HC03_VC144__DP04_close</th>
      <th>HC03_VC148__DP04_close</th>
      <th>HC03_VC149__DP04_close</th>
      <th>HC03_VC150__DP04_close</th>
      <th>HC03_VC151__DP04_close</th>
      <th>HC03_VC152__DP04_close</th>
      <th>HC03_VC158__DP04_close</th>
      <th>HC03_VC159__DP04_close</th>
      <th>HC03_VC160__DP04_close</th>
      <th>HC03_VC161__DP04_close</th>
      <th>HC03_VC162__DP04_close</th>
      <th>HC03_VC168__DP04_close</th>
      <th>HC03_VC169__DP04_close</th>
      <th>HC03_VC170__DP04_close</th>
      <th>HC03_VC171__DP04_close</th>
      <th>HC03_VC172__DP04_close</th>
      <th>HC03_VC173__DP04_close</th>
      <th>HC03_VC174__DP04_close</th>
      <th>HC03_VC182__DP04_close</th>
      <th>HC03_VC183__DP04_close</th>
      <th>HC03_VC184__DP04_close</th>
      <th>HC03_VC185__DP04_close</th>
      <th>HC03_VC186__DP04_close</th>
      <th>HC03_VC187__DP04_close</th>
      <th>HC03_VC188__DP04_close</th>
      <th>HC03_VC197__DP04_close</th>
      <th>HC03_VC198__DP04_close</th>
      <th>HC03_VC199__DP04_close</th>
      <th>HC03_VC200__DP04_close</th>
      <th>HC03_VC201__DP04_close</th>
      <th>HC03_VC202__DP04_close</th>
      <th>any_bll_ge5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48001</td>
      <td>48001</td>
      <td>25.7</td>
      <td>660</td>
      <td>168</td>
      <td>25.5</td>
      <td>165</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.8</td>
      <td>0</td>
      <td>11.8</td>
      <td>14.6</td>
      <td>13.9</td>
      <td>11.7</td>
      <td>9.2</td>
      <td>10.6</td>
      <td>12.9</td>
      <td>11.8</td>
      <td>11.6</td>
      <td>7.7</td>
      <td>36.2</td>
      <td>11.4</td>
      <td>10.2</td>
      <td>15.9</td>
      <td>12.0</td>
      <td>8.6</td>
      <td>5.3</td>
      <td>7.2</td>
      <td>4.7</td>
      <td>3.4</td>
      <td>6.2</td>
      <td>24.8</td>
      <td>22.8</td>
      <td>27.6</td>
      <td>11.0</td>
      <td>1.7</td>
      <td>10.3</td>
      <td>18.1</td>
      <td>21.4</td>
      <td>18.8</td>
      <td>23.9</td>
      <td>1.9</td>
      <td>24.8</td>
      <td>31.8</td>
      <td>7.9</td>
      <td>5.5</td>
      <td>19.8</td>
      <td>9.1</td>
      <td>7.4</td>
      <td>20.8</td>
      <td>8.0</td>
      <td>5.6</td>
      <td>19.8</td>
      <td>7.9</td>
      <td>5.6</td>
      <td>19.1</td>
      <td>7.7</td>
      <td>5.3</td>
      <td>19.3</td>
      <td>3.8</td>
      <td>2.3</td>
      <td>12.0</td>
      <td>2.1</td>
      <td>2.4</td>
      <td>1.6</td>
      <td>4.8</td>
      <td>3.9</td>
      <td>18.2</td>
      <td>11.0</td>
      <td>3.2</td>
      <td>0.0</td>
      <td>5.5</td>
      <td>5.1</td>
      <td>10.5</td>
      <td>12.7</td>
      <td>16.1</td>
      <td>7.5</td>
      <td>6.2</td>
      <td>20</td>
      <td>9.1</td>
      <td>4.7</td>
      <td>25.4</td>
      <td>4.1</td>
      <td>0.0</td>
      <td>7.2</td>
      <td>4.6</td>
      <td>18.6</td>
      <td>4.8</td>
      <td>5.4</td>
      <td>5.8</td>
      <td>34.2</td>
      <td>18.1</td>
      <td>6.6</td>
      <td>3.6</td>
      <td>17.0</td>
      <td>7.7</td>
      <td>6.8</td>
      <td>15.3</td>
      <td>22.3</td>
      <td>14.0</td>
      <td>20.9</td>
      <td>14</td>
      <td>62.5</td>
      <td>4.6</td>
      <td>4.7</td>
      <td>7.6</td>
      <td>3.9</td>
      <td>2.0</td>
      <td>17.7</td>
      <td>0</td>
      <td>0</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>74.5</td>
      <td>25.5</td>
      <td>91.3</td>
      <td>8.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>77.3</td>
      <td>22.7</td>
      <td>93.6</td>
      <td>6.4</td>
      <td>95.3</td>
      <td>4.7</td>
      <td>82.3</td>
      <td>17.7</td>
      <td>42.0</td>
      <td>58.0</td>
      <td>64.6</td>
      <td>72.9</td>
      <td>27.1</td>
      <td>97.5</td>
      <td>80.4</td>
      <td>19.6</td>
      <td>97.8</td>
      <td>94.1</td>
      <td>5.9</td>
      <td>87.5</td>
      <td>94.4</td>
      <td>5.6</td>
      <td>46.8</td>
      <td>100</td>
      <td>0</td>
      <td>25.1</td>
      <td>74.2</td>
      <td>25.8</td>
      <td>8.4</td>
      <td>83.5</td>
      <td>16.5</td>
      <td>2.8</td>
      <td>71.6</td>
      <td>28.4</td>
      <td>76.5</td>
      <td>23.5</td>
      <td>62.1</td>
      <td>37.9</td>
      <td>82.0</td>
      <td>18.0</td>
      <td>78.0</td>
      <td>22.0</td>
      <td>51.8</td>
      <td>48.2</td>
      <td>89.7</td>
      <td>10.3</td>
      <td>88.2</td>
      <td>84.9</td>
      <td>91.4</td>
      <td>16.0</td>
      <td>16.3</td>
      <td>15.7</td>
      <td>98.7</td>
      <td>1.3</td>
      <td>60.2</td>
      <td>39.8</td>
      <td>69.7</td>
      <td>30.3</td>
      <td>63.6</td>
      <td>36.4</td>
      <td>69.7</td>
      <td>30.3</td>
      <td>54.4</td>
      <td>45.6</td>
      <td>63.6</td>
      <td>36.4</td>
      <td>68.9</td>
      <td>31.1</td>
      <td>98.7</td>
      <td>1.3</td>
      <td>59.4</td>
      <td>40.6</td>
      <td>77.6</td>
      <td>22.4</td>
      <td>54.0</td>
      <td>46.0</td>
      <td>11.8</td>
      <td>88.2</td>
      <td>3.3</td>
      <td>17.8</td>
      <td>0.0</td>
      <td>25.9</td>
      <td>12.2</td>
      <td>11.3</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>0.0</td>
      <td>11.6</td>
      <td>32.6</td>
      <td>11.9</td>
      <td>7.3</td>
      <td>0.0</td>
      <td>18.9</td>
      <td>12.8</td>
      <td>22.7</td>
      <td>12.4</td>
      <td>13.2</td>
      <td>5.6</td>
      <td>13.7</td>
      <td>17.6</td>
      <td>13.9</td>
      <td>43.7</td>
      <td>8.4</td>
      <td>13.7</td>
      <td>11.1</td>
      <td>20.6</td>
      <td>11.0</td>
      <td>11.7</td>
      <td>26.9</td>
      <td>10.2</td>
      <td>9.1</td>
      <td>4.9</td>
      <td>7.8</td>
      <td>11.8</td>
      <td>29.3</td>
      <td>11.7</td>
      <td>6.9</td>
      <td>71.8</td>
      <td>52.7</td>
      <td>60.4</td>
      <td>47.3</td>
      <td>14.5</td>
      <td>5.4</td>
      <td>1.0</td>
      <td>0</td>
      <td>34.3</td>
      <td>13.7</td>
      <td>22.7</td>
      <td>3.7</td>
      <td>16.2</td>
      <td>10.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>11.8</td>
      <td>84.5</td>
      <td>15.5</td>
      <td>82.0</td>
      <td>3.5</td>
      <td>1.2</td>
      <td>1.4</td>
      <td>3.8</td>
      <td>0.6</td>
      <td>1.4</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.2</td>
      <td>7.4</td>
      <td>14.0</td>
      <td>9.4</td>
      <td>17.5</td>
      <td>12.0</td>
      <td>14.8</td>
      <td>9.5</td>
      <td>14.9</td>
      <td>0.5</td>
      <td>1.4</td>
      <td>5.2</td>
      <td>12.9</td>
      <td>23.1</td>
      <td>23.8</td>
      <td>13.8</td>
      <td>11.3</td>
      <td>7.9</td>
      <td>0.7</td>
      <td>8.7</td>
      <td>25.6</td>
      <td>49.4</td>
      <td>13.6</td>
      <td>2.0</td>
      <td>78.2</td>
      <td>21.8</td>
      <td>21.5</td>
      <td>34.7</td>
      <td>19.7</td>
      <td>11.6</td>
      <td>6.9</td>
      <td>5.6</td>
      <td>7.7</td>
      <td>35.5</td>
      <td>38.3</td>
      <td>18.5</td>
      <td>87.1</td>
      <td>1.3</td>
      <td>7.5</td>
      <td>0.6</td>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.2</td>
      <td>2.6</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.5</td>
      <td>24.8</td>
      <td>18.5</td>
      <td>18.0</td>
      <td>14.8</td>
      <td>8.0</td>
      <td>1.6</td>
      <td>0.8</td>
      <td>60.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>2.5</td>
      <td>9.7</td>
      <td>19.5</td>
      <td>33.6</td>
      <td>25.9</td>
      <td>8.8</td>
      <td>1.3</td>
      <td>2.3</td>
      <td>7.0</td>
      <td>16.8</td>
      <td>72.6</td>
      <td>34.7</td>
      <td>17.8</td>
      <td>10.3</td>
      <td>9.8</td>
      <td>27.3</td>
      <td>26.4</td>
      <td>19.6</td>
      <td>17.0</td>
      <td>13.6</td>
      <td>4.8</td>
      <td>2.4</td>
      <td>16.2</td>
      <td>1.1</td>
      <td>8.9</td>
      <td>4.7</td>
      <td>33.4</td>
      <td>32.7</td>
      <td>14.5</td>
      <td>4.6</td>
      <td>9.4</td>
      <td>7.3</td>
      <td>5.7</td>
      <td>20.7</td>
      <td>14.2</td>
      <td>42.7</td>
      <td>24.66</td>
      <td>11.66</td>
      <td>17.26</td>
      <td>16.56</td>
      <td>11.38</td>
      <td>6.06</td>
      <td>10.58</td>
      <td>12.72</td>
      <td>11.48</td>
      <td>10.98</td>
      <td>27.225</td>
      <td>53.54</td>
      <td>10.18</td>
      <td>9.18</td>
      <td>27.68</td>
      <td>8.50</td>
      <td>7.22</td>
      <td>5.14</td>
      <td>8.72</td>
      <td>5.46</td>
      <td>4.32</td>
      <td>6.52</td>
      <td>31.80</td>
      <td>34.02</td>
      <td>29.62</td>
      <td>10.46</td>
      <td>2.26</td>
      <td>14.06</td>
      <td>15.76</td>
      <td>24.44</td>
      <td>22.02</td>
      <td>26.98</td>
      <td>4.72</td>
      <td>36.66</td>
      <td>34.04</td>
      <td>8.32</td>
      <td>4.82</td>
      <td>25.44</td>
      <td>13.54</td>
      <td>6.68</td>
      <td>35.28</td>
      <td>8.22</td>
      <td>4.74</td>
      <td>25.32</td>
      <td>7.98</td>
      <td>4.78</td>
      <td>23.76</td>
      <td>7.48</td>
      <td>4.30</td>
      <td>22.86</td>
      <td>7.34</td>
      <td>4.16</td>
      <td>24.78</td>
      <td>2.78</td>
      <td>2.74</td>
      <td>4.10</td>
      <td>1.74</td>
      <td>1.38</td>
      <td>2.94</td>
      <td>16.00</td>
      <td>12.14</td>
      <td>29.800</td>
      <td>3.48</td>
      <td>1.96</td>
      <td>11.54</td>
      <td>37.04</td>
      <td>38.00</td>
      <td>7.18</td>
      <td>2.58</td>
      <td>39.90</td>
      <td>6.56</td>
      <td>3.54</td>
      <td>19.78</td>
      <td>5.20</td>
      <td>1.00</td>
      <td>4.66</td>
      <td>4.04</td>
      <td>7.1</td>
      <td>12.22</td>
      <td>4.62</td>
      <td>33.80</td>
      <td>17.18</td>
      <td>10.60</td>
      <td>6.34</td>
      <td>4.76</td>
      <td>16.68</td>
      <td>10.04</td>
      <td>3.18</td>
      <td>38.18</td>
      <td>8.60</td>
      <td>8.14</td>
      <td>14.28</td>
      <td>6.60</td>
      <td>42.26</td>
      <td>13.74</td>
      <td>5.90</td>
      <td>30.36</td>
      <td>5.00</td>
      <td>5.20</td>
      <td>5.76</td>
      <td>0.24</td>
      <td>0.0</td>
      <td>89.76</td>
      <td>10.24</td>
      <td>87.36</td>
      <td>12.64</td>
      <td>90.06</td>
      <td>9.94</td>
      <td>96.20</td>
      <td>3.80</td>
      <td>85.32</td>
      <td>14.68</td>
      <td>87.28</td>
      <td>12.72</td>
      <td>92.88</td>
      <td>7.12</td>
      <td>84.56</td>
      <td>15.44</td>
      <td>66.825</td>
      <td>33.175</td>
      <td>49.28</td>
      <td>93.12</td>
      <td>6.88</td>
      <td>86.04</td>
      <td>84.42</td>
      <td>15.58</td>
      <td>92.78</td>
      <td>87.94</td>
      <td>12.06</td>
      <td>99.28</td>
      <td>91.22</td>
      <td>8.78</td>
      <td>80.84</td>
      <td>97.26</td>
      <td>2.74</td>
      <td>40.52</td>
      <td>90.52</td>
      <td>9.48</td>
      <td>9.98</td>
      <td>75.925</td>
      <td>24.075</td>
      <td>3.48</td>
      <td>69.72</td>
      <td>30.28</td>
      <td>82.96</td>
      <td>17.04</td>
      <td>82.80</td>
      <td>17.20</td>
      <td>82.78</td>
      <td>17.22</td>
      <td>92.44</td>
      <td>7.56</td>
      <td>92.06</td>
      <td>7.94</td>
      <td>91.46</td>
      <td>8.54</td>
      <td>90.56</td>
      <td>90.00</td>
      <td>91.18</td>
      <td>20.08</td>
      <td>19.4</td>
      <td>20.68</td>
      <td>99.42</td>
      <td>0.58</td>
      <td>89.42</td>
      <td>10.58</td>
      <td>83.96</td>
      <td>16.04</td>
      <td>92.28</td>
      <td>7.72</td>
      <td>83.96</td>
      <td>16.04</td>
      <td>81.30</td>
      <td>18.70</td>
      <td>92.28</td>
      <td>7.72</td>
      <td>95.90</td>
      <td>4.10</td>
      <td>99.56</td>
      <td>0.44</td>
      <td>90.16</td>
      <td>9.84</td>
      <td>88.20</td>
      <td>11.80</td>
      <td>87.44</td>
      <td>12.56</td>
      <td>11.50</td>
      <td>88.50</td>
      <td>4.76</td>
      <td>16.76</td>
      <td>0.16</td>
      <td>27.96</td>
      <td>12.74</td>
      <td>10.28</td>
      <td>11.44</td>
      <td>11.18</td>
      <td>22.90</td>
      <td>11.18</td>
      <td>9.92</td>
      <td>11.48</td>
      <td>11.760</td>
      <td>9.460</td>
      <td>18.05</td>
      <td>12.14</td>
      <td>17.54</td>
      <td>13.80</td>
      <td>11.34</td>
      <td>7.04</td>
      <td>13.48</td>
      <td>17.24</td>
      <td>13.96</td>
      <td>43.12</td>
      <td>6.74</td>
      <td>13.48</td>
      <td>12.22</td>
      <td>20.10</td>
      <td>9.38</td>
      <td>11.50</td>
      <td>22.28</td>
      <td>18.44</td>
      <td>12.76</td>
      <td>3.70</td>
      <td>3.44</td>
      <td>11.52</td>
      <td>28.14</td>
      <td>17.92</td>
      <td>6.16</td>
      <td>74.74</td>
      <td>54.22</td>
      <td>62.60</td>
      <td>48.02</td>
      <td>15.36</td>
      <td>5.54</td>
      <td>1.32</td>
      <td>0.66</td>
      <td>32.76</td>
      <td>12.16</td>
      <td>20.64</td>
      <td>2.48</td>
      <td>14.18</td>
      <td>9.40</td>
      <td>1.66</td>
      <td>0.26</td>
      <td>11.50</td>
      <td>81.08</td>
      <td>18.92</td>
      <td>78.24</td>
      <td>4.58</td>
      <td>1.96</td>
      <td>1.16</td>
      <td>1.66</td>
      <td>1.74</td>
      <td>0.56</td>
      <td>10.08</td>
      <td>0.00</td>
      <td>0.30</td>
      <td>12.48</td>
      <td>17.72</td>
      <td>13.96</td>
      <td>14.52</td>
      <td>7.86</td>
      <td>10.78</td>
      <td>5.30</td>
      <td>17.12</td>
      <td>0.50</td>
      <td>0.86</td>
      <td>4.62</td>
      <td>11.90</td>
      <td>21.40</td>
      <td>23.78</td>
      <td>16.78</td>
      <td>10.10</td>
      <td>10.04</td>
      <td>0.68</td>
      <td>4.20</td>
      <td>24.18</td>
      <td>53.42</td>
      <td>14.34</td>
      <td>3.14</td>
      <td>84.58</td>
      <td>15.42</td>
      <td>17.08</td>
      <td>35.90</td>
      <td>23.66</td>
      <td>12.16</td>
      <td>7.18</td>
      <td>4.04</td>
      <td>3.94</td>
      <td>29.0</td>
      <td>42.74</td>
      <td>24.32</td>
      <td>84.44</td>
      <td>6.26</td>
      <td>4.40</td>
      <td>1.32</td>
      <td>0</td>
      <td>2.84</td>
      <td>0.0</td>
      <td>0.58</td>
      <td>0.18</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>2.50</td>
      <td>99.14</td>
      <td>0.74</td>
      <td>0.10</td>
      <td>13.58</td>
      <td>16.26</td>
      <td>16.40</td>
      <td>18.54</td>
      <td>18.86</td>
      <td>10.82</td>
      <td>4.60</td>
      <td>0.92</td>
      <td>59.86</td>
      <td>40.14</td>
      <td>0.28</td>
      <td>1.42</td>
      <td>6.32</td>
      <td>14.58</td>
      <td>30.58</td>
      <td>24.54</td>
      <td>22.28</td>
      <td>0.36</td>
      <td>2.26</td>
      <td>8.16</td>
      <td>19.60</td>
      <td>69.60</td>
      <td>40.08</td>
      <td>13.68</td>
      <td>12.22</td>
      <td>6.82</td>
      <td>27.20</td>
      <td>35.98</td>
      <td>13.94</td>
      <td>16.16</td>
      <td>11.28</td>
      <td>4.26</td>
      <td>3.14</td>
      <td>15.26</td>
      <td>0.00</td>
      <td>1.30</td>
      <td>4.96</td>
      <td>31.72</td>
      <td>23.58</td>
      <td>32.24</td>
      <td>6.20</td>
      <td>20.60</td>
      <td>8.68</td>
      <td>6.80</td>
      <td>14.24</td>
      <td>12.00</td>
      <td>37.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48002</td>
      <td>48002</td>
      <td>25.8</td>
      <td>219</td>
      <td>24</td>
      <td>11.0</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4.2</td>
      <td>0</td>
      <td>5.0</td>
      <td>2.8</td>
      <td>2.2</td>
      <td>4.1</td>
      <td>10.8</td>
      <td>4.3</td>
      <td>5.6</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>32.1</td>
      <td>3.1</td>
      <td>4.5</td>
      <td>15.6</td>
      <td>5.8</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>1.2</td>
      <td>1.8</td>
      <td>0.6</td>
      <td>17.4</td>
      <td>14.8</td>
      <td>21.3</td>
      <td>5.7</td>
      <td>0.0</td>
      <td>6.1</td>
      <td>11.1</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>15.7</td>
      <td>4.2</td>
      <td>4.2</td>
      <td>5.5</td>
      <td>4.1</td>
      <td>3.7</td>
      <td>7.0</td>
      <td>4.2</td>
      <td>4.2</td>
      <td>5.5</td>
      <td>3.0</td>
      <td>2.8</td>
      <td>6.2</td>
      <td>3.0</td>
      <td>2.9</td>
      <td>6.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>8.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>11.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.1</td>
      <td>12.8</td>
      <td>0.0</td>
      <td>19.4</td>
      <td>21.7</td>
      <td>5.4</td>
      <td>5.7</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>4.9</td>
      <td>4.2</td>
      <td>12.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.5</td>
      <td>5.0</td>
      <td>16.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>12.8</td>
      <td>18.5</td>
      <td>20</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>23.5</td>
      <td>2.5</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>93.5</td>
      <td>6.5</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>95.3</td>
      <td>4.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>90.7</td>
      <td>9.3</td>
      <td>94.7</td>
      <td>5.3</td>
      <td>89.9</td>
      <td>10.1</td>
      <td>61.3</td>
      <td>38.7</td>
      <td>58.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>91.9</td>
      <td>8.1</td>
      <td>97.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>69.2</td>
      <td>75</td>
      <td>25</td>
      <td>27.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>19.7</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>3.3</td>
      <td>88.2</td>
      <td>11.8</td>
      <td>84.4</td>
      <td>15.6</td>
      <td>82.4</td>
      <td>17.6</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>88.9</td>
      <td>11.1</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>25.0</td>
      <td>90.9</td>
      <td>90.7</td>
      <td>91.0</td>
      <td>13.3</td>
      <td>12.4</td>
      <td>14.2</td>
      <td>97.2</td>
      <td>2.8</td>
      <td>62.0</td>
      <td>38.0</td>
      <td>56.7</td>
      <td>43.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>56.7</td>
      <td>43.3</td>
      <td>51.0</td>
      <td>49.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>99.6</td>
      <td>0.4</td>
      <td>90.2</td>
      <td>9.8</td>
      <td>86.9</td>
      <td>13.1</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>8.9</td>
      <td>91.1</td>
      <td>0.0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>25.6</td>
      <td>9.9</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>7.8</td>
      <td>27.0</td>
      <td>7.4</td>
      <td>49.6</td>
      <td>25.6</td>
      <td>63.5</td>
      <td>10.1</td>
      <td>16.6</td>
      <td>13.7</td>
      <td>8.6</td>
      <td>0.0</td>
      <td>11.3</td>
      <td>13.0</td>
      <td>7.9</td>
      <td>68.3</td>
      <td>7.9</td>
      <td>11.3</td>
      <td>3.3</td>
      <td>21.9</td>
      <td>10.0</td>
      <td>8.9</td>
      <td>10.7</td>
      <td>13.7</td>
      <td>12.9</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>8.9</td>
      <td>22.6</td>
      <td>19.2</td>
      <td>5.5</td>
      <td>79.2</td>
      <td>63.8</td>
      <td>64.2</td>
      <td>56.0</td>
      <td>16.1</td>
      <td>7.8</td>
      <td>0.8</td>
      <td>0</td>
      <td>26.8</td>
      <td>10.7</td>
      <td>17.6</td>
      <td>2.3</td>
      <td>9.6</td>
      <td>8.3</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>8.9</td>
      <td>91.1</td>
      <td>8.9</td>
      <td>98.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>0</td>
      <td>0.0</td>
      <td>12.6</td>
      <td>21.9</td>
      <td>7.9</td>
      <td>24.3</td>
      <td>5.9</td>
      <td>10.0</td>
      <td>1.7</td>
      <td>15.6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>15.6</td>
      <td>31.0</td>
      <td>20.2</td>
      <td>14.8</td>
      <td>9.8</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>10.0</td>
      <td>63.2</td>
      <td>18.7</td>
      <td>5.6</td>
      <td>93.5</td>
      <td>6.5</td>
      <td>4.8</td>
      <td>27.2</td>
      <td>37.8</td>
      <td>13.6</td>
      <td>11.6</td>
      <td>5.0</td>
      <td>1.1</td>
      <td>18.4</td>
      <td>39.6</td>
      <td>40.8</td>
      <td>37.6</td>
      <td>40.8</td>
      <td>3.8</td>
      <td>5.1</td>
      <td>0</td>
      <td>11.8</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>97.4</td>
      <td>0.9</td>
      <td>1.7</td>
      <td>1.0</td>
      <td>19.9</td>
      <td>17.9</td>
      <td>29.7</td>
      <td>24.1</td>
      <td>6.3</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>67.7</td>
      <td>32.3</td>
      <td>0</td>
      <td>0.9</td>
      <td>5.3</td>
      <td>4.7</td>
      <td>33.4</td>
      <td>41.0</td>
      <td>14.6</td>
      <td>1.7</td>
      <td>1.4</td>
      <td>4.4</td>
      <td>9.4</td>
      <td>83.1</td>
      <td>32.8</td>
      <td>16.6</td>
      <td>11.9</td>
      <td>20.1</td>
      <td>18.6</td>
      <td>36.2</td>
      <td>19.6</td>
      <td>14.6</td>
      <td>19.1</td>
      <td>1.7</td>
      <td>7.2</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31.5</td>
      <td>0.0</td>
      <td>44.4</td>
      <td>24.1</td>
      <td>20.4</td>
      <td>0.0</td>
      <td>9.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.4</td>
      <td>28.64</td>
      <td>10.42</td>
      <td>15.02</td>
      <td>14.82</td>
      <td>9.46</td>
      <td>4.54</td>
      <td>8.88</td>
      <td>11.88</td>
      <td>10.14</td>
      <td>10.06</td>
      <td>14.260</td>
      <td>13.98</td>
      <td>9.80</td>
      <td>8.34</td>
      <td>19.18</td>
      <td>7.04</td>
      <td>8.48</td>
      <td>4.96</td>
      <td>7.06</td>
      <td>4.62</td>
      <td>2.60</td>
      <td>6.86</td>
      <td>26.42</td>
      <td>27.56</td>
      <td>22.66</td>
      <td>8.92</td>
      <td>2.22</td>
      <td>11.40</td>
      <td>14.72</td>
      <td>26.40</td>
      <td>24.10</td>
      <td>29.00</td>
      <td>7.04</td>
      <td>46.56</td>
      <td>30.88</td>
      <td>6.66</td>
      <td>3.26</td>
      <td>31.52</td>
      <td>11.92</td>
      <td>5.38</td>
      <td>42.58</td>
      <td>6.32</td>
      <td>2.78</td>
      <td>31.52</td>
      <td>5.98</td>
      <td>2.66</td>
      <td>31.60</td>
      <td>6.02</td>
      <td>2.68</td>
      <td>31.38</td>
      <td>4.30</td>
      <td>1.76</td>
      <td>20.10</td>
      <td>2.22</td>
      <td>1.78</td>
      <td>6.22</td>
      <td>3.80</td>
      <td>3.50</td>
      <td>12.50</td>
      <td>26.58</td>
      <td>13.56</td>
      <td>57.025</td>
      <td>6.74</td>
      <td>5.80</td>
      <td>22.68</td>
      <td>12.68</td>
      <td>9.52</td>
      <td>5.64</td>
      <td>3.80</td>
      <td>14.34</td>
      <td>7.28</td>
      <td>1.46</td>
      <td>42.40</td>
      <td>3.50</td>
      <td>3.68</td>
      <td>2.02</td>
      <td>1.82</td>
      <td>3.6</td>
      <td>10.62</td>
      <td>3.50</td>
      <td>39.18</td>
      <td>10.18</td>
      <td>3.56</td>
      <td>5.74</td>
      <td>2.36</td>
      <td>24.10</td>
      <td>6.26</td>
      <td>2.20</td>
      <td>41.90</td>
      <td>5.06</td>
      <td>3.74</td>
      <td>18.08</td>
      <td>9.90</td>
      <td>65.68</td>
      <td>11.68</td>
      <td>4.66</td>
      <td>30.06</td>
      <td>0.80</td>
      <td>0.68</td>
      <td>3.34</td>
      <td>1.60</td>
      <td>1.6</td>
      <td>94.52</td>
      <td>5.48</td>
      <td>66.48</td>
      <td>33.52</td>
      <td>97.42</td>
      <td>2.58</td>
      <td>94.62</td>
      <td>5.38</td>
      <td>96.46</td>
      <td>3.54</td>
      <td>97.26</td>
      <td>2.74</td>
      <td>99.08</td>
      <td>0.92</td>
      <td>93.30</td>
      <td>6.70</td>
      <td>74.500</td>
      <td>25.500</td>
      <td>38.48</td>
      <td>67.74</td>
      <td>32.26</td>
      <td>97.86</td>
      <td>95.30</td>
      <td>4.70</td>
      <td>100.00</td>
      <td>97.92</td>
      <td>2.08</td>
      <td>94.20</td>
      <td>98.80</td>
      <td>1.20</td>
      <td>65.50</td>
      <td>97.70</td>
      <td>2.30</td>
      <td>38.02</td>
      <td>88.92</td>
      <td>11.08</td>
      <td>8.44</td>
      <td>84.000</td>
      <td>16.000</td>
      <td>2.10</td>
      <td>96.12</td>
      <td>3.88</td>
      <td>90.72</td>
      <td>9.28</td>
      <td>92.12</td>
      <td>7.88</td>
      <td>89.90</td>
      <td>10.10</td>
      <td>90.48</td>
      <td>9.52</td>
      <td>92.66</td>
      <td>7.34</td>
      <td>90.16</td>
      <td>9.84</td>
      <td>90.18</td>
      <td>89.20</td>
      <td>91.10</td>
      <td>14.32</td>
      <td>12.9</td>
      <td>15.68</td>
      <td>98.22</td>
      <td>1.78</td>
      <td>75.80</td>
      <td>24.20</td>
      <td>71.66</td>
      <td>28.34</td>
      <td>69.38</td>
      <td>30.62</td>
      <td>71.66</td>
      <td>28.34</td>
      <td>74.64</td>
      <td>25.38</td>
      <td>69.38</td>
      <td>30.62</td>
      <td>63.25</td>
      <td>36.75</td>
      <td>99.28</td>
      <td>0.72</td>
      <td>84.68</td>
      <td>15.32</td>
      <td>87.54</td>
      <td>12.46</td>
      <td>77.98</td>
      <td>22.02</td>
      <td>10.02</td>
      <td>89.98</td>
      <td>3.50</td>
      <td>14.12</td>
      <td>1.52</td>
      <td>21.16</td>
      <td>9.96</td>
      <td>10.14</td>
      <td>9.88</td>
      <td>9.66</td>
      <td>14.26</td>
      <td>9.56</td>
      <td>20.24</td>
      <td>9.60</td>
      <td>17.925</td>
      <td>15.125</td>
      <td>23.75</td>
      <td>11.50</td>
      <td>17.90</td>
      <td>12.92</td>
      <td>11.34</td>
      <td>5.14</td>
      <td>12.24</td>
      <td>14.44</td>
      <td>12.14</td>
      <td>34.28</td>
      <td>7.44</td>
      <td>12.24</td>
      <td>10.52</td>
      <td>18.06</td>
      <td>9.10</td>
      <td>9.98</td>
      <td>22.74</td>
      <td>14.74</td>
      <td>5.92</td>
      <td>7.92</td>
      <td>5.80</td>
      <td>10.14</td>
      <td>24.86</td>
      <td>14.38</td>
      <td>6.50</td>
      <td>77.12</td>
      <td>61.44</td>
      <td>68.52</td>
      <td>56.90</td>
      <td>9.94</td>
      <td>3.82</td>
      <td>1.94</td>
      <td>0.74</td>
      <td>26.80</td>
      <td>11.48</td>
      <td>14.42</td>
      <td>1.78</td>
      <td>13.96</td>
      <td>9.60</td>
      <td>1.78</td>
      <td>0.12</td>
      <td>10.02</td>
      <td>93.98</td>
      <td>6.02</td>
      <td>88.98</td>
      <td>1.66</td>
      <td>1.44</td>
      <td>1.18</td>
      <td>0.98</td>
      <td>1.22</td>
      <td>1.16</td>
      <td>3.42</td>
      <td>0.00</td>
      <td>0.28</td>
      <td>16.74</td>
      <td>22.32</td>
      <td>9.78</td>
      <td>14.82</td>
      <td>6.84</td>
      <td>6.28</td>
      <td>2.64</td>
      <td>20.30</td>
      <td>0.32</td>
      <td>0.96</td>
      <td>2.98</td>
      <td>9.34</td>
      <td>20.62</td>
      <td>22.72</td>
      <td>20.86</td>
      <td>11.92</td>
      <td>10.26</td>
      <td>0.36</td>
      <td>3.72</td>
      <td>15.48</td>
      <td>59.18</td>
      <td>17.00</td>
      <td>4.26</td>
      <td>86.96</td>
      <td>13.04</td>
      <td>15.42</td>
      <td>34.04</td>
      <td>25.38</td>
      <td>12.96</td>
      <td>7.76</td>
      <td>4.40</td>
      <td>2.48</td>
      <td>23.5</td>
      <td>41.06</td>
      <td>32.96</td>
      <td>47.38</td>
      <td>29.72</td>
      <td>8.66</td>
      <td>2.78</td>
      <td>0</td>
      <td>8.66</td>
      <td>0.1</td>
      <td>1.70</td>
      <td>0.94</td>
      <td>0.56</td>
      <td>0.68</td>
      <td>3.28</td>
      <td>98.26</td>
      <td>1.30</td>
      <td>0.42</td>
      <td>8.74</td>
      <td>17.40</td>
      <td>24.34</td>
      <td>23.10</td>
      <td>18.64</td>
      <td>6.50</td>
      <td>0.88</td>
      <td>0.40</td>
      <td>72.18</td>
      <td>27.82</td>
      <td>0.00</td>
      <td>0.80</td>
      <td>2.12</td>
      <td>11.98</td>
      <td>37.10</td>
      <td>29.40</td>
      <td>18.58</td>
      <td>1.06</td>
      <td>3.12</td>
      <td>8.02</td>
      <td>20.58</td>
      <td>67.22</td>
      <td>34.36</td>
      <td>15.32</td>
      <td>14.74</td>
      <td>8.36</td>
      <td>27.22</td>
      <td>31.00</td>
      <td>27.66</td>
      <td>12.22</td>
      <td>7.50</td>
      <td>6.32</td>
      <td>1.94</td>
      <td>13.34</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>7.74</td>
      <td>39.30</td>
      <td>30.00</td>
      <td>13.90</td>
      <td>8.28</td>
      <td>10.38</td>
      <td>18.60</td>
      <td>16.56</td>
      <td>10.12</td>
      <td>6.32</td>
      <td>38.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48003</td>
      <td>48003</td>
      <td>24.1</td>
      <td>388</td>
      <td>41</td>
      <td>10.6</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.6</td>
      <td>5.7</td>
      <td>5.7</td>
      <td>7.9</td>
      <td>3.1</td>
      <td>5.6</td>
      <td>7.5</td>
      <td>6.6</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>8.7</td>
      <td>6.6</td>
      <td>7.0</td>
      <td>13.4</td>
      <td>5.1</td>
      <td>9.6</td>
      <td>2.9</td>
      <td>7.3</td>
      <td>4.9</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>23.5</td>
      <td>29.1</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>2.3</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>22.4</td>
      <td>31.7</td>
      <td>15.2</td>
      <td>8.6</td>
      <td>29.7</td>
      <td>28.6</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>35.9</td>
      <td>8.6</td>
      <td>1.1</td>
      <td>46.0</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>35.9</td>
      <td>4.9</td>
      <td>1.2</td>
      <td>35.9</td>
      <td>5.0</td>
      <td>1.2</td>
      <td>35.9</td>
      <td>3.6</td>
      <td>0.0</td>
      <td>34.6</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.8</td>
      <td>6.9</td>
      <td>61.1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>32.7</td>
      <td>9.1</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.1</td>
      <td>1.3</td>
      <td>45.6</td>
      <td>3.2</td>
      <td>2.7</td>
      <td>1.8</td>
      <td>1.2</td>
      <td>0.0</td>
      <td>10.4</td>
      <td>1.3</td>
      <td>52.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.1</td>
      <td>1.5</td>
      <td>52.9</td>
      <td>2.4</td>
      <td>1.1</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.7</td>
      <td>2</td>
      <td>56.3</td>
      <td>8.6</td>
      <td>2.3</td>
      <td>38.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>90.9</td>
      <td>9.1</td>
      <td>59.8</td>
      <td>40.2</td>
      <td>95.1</td>
      <td>4.9</td>
      <td>91.3</td>
      <td>8.7</td>
      <td>92.6</td>
      <td>7.4</td>
      <td>96.6</td>
      <td>3.4</td>
      <td>98.1</td>
      <td>1.9</td>
      <td>88.6</td>
      <td>11.4</td>
      <td>73.1</td>
      <td>26.9</td>
      <td>38.3</td>
      <td>56.5</td>
      <td>43.5</td>
      <td>100.0</td>
      <td>89.4</td>
      <td>10.6</td>
      <td>100.0</td>
      <td>97.6</td>
      <td>2.4</td>
      <td>100.0</td>
      <td>97.5</td>
      <td>2.5</td>
      <td>31.1</td>
      <td>100</td>
      <td>0</td>
      <td>42.8</td>
      <td>84.4</td>
      <td>15.6</td>
      <td>4.3</td>
      <td>66.7</td>
      <td>33.3</td>
      <td>1.3</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>86.6</td>
      <td>13.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>84.6</td>
      <td>15.4</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>78.9</td>
      <td>21.1</td>
      <td>91.5</td>
      <td>91.5</td>
      <td>91.4</td>
      <td>18.3</td>
      <td>17.5</td>
      <td>19.0</td>
      <td>98.5</td>
      <td>1.5</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>80.1</td>
      <td>19.9</td>
      <td>78.8</td>
      <td>21.2</td>
      <td>76.9</td>
      <td>23.1</td>
      <td>74.8</td>
      <td>25.2</td>
      <td>98.9</td>
      <td>1.1</td>
      <td>80.9</td>
      <td>19.1</td>
      <td>95.5</td>
      <td>4.5</td>
      <td>64.4</td>
      <td>35.6</td>
      <td>8.3</td>
      <td>91.7</td>
      <td>0.6</td>
      <td>12.0</td>
      <td>5.8</td>
      <td>10.6</td>
      <td>9.8</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>43.9</td>
      <td>7.2</td>
      <td>30.6</td>
      <td>8.0</td>
      <td>16.5</td>
      <td>19.5</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>22.3</td>
      <td>9.9</td>
      <td>11.7</td>
      <td>3.9</td>
      <td>10.5</td>
      <td>12.6</td>
      <td>9.7</td>
      <td>32.5</td>
      <td>7.1</td>
      <td>10.5</td>
      <td>7.3</td>
      <td>15.1</td>
      <td>10.8</td>
      <td>8.4</td>
      <td>25.0</td>
      <td>10.7</td>
      <td>7.9</td>
      <td>7.5</td>
      <td>1.4</td>
      <td>8.4</td>
      <td>29.1</td>
      <td>7.9</td>
      <td>4.5</td>
      <td>79.6</td>
      <td>61.1</td>
      <td>69.9</td>
      <td>56.8</td>
      <td>10.9</td>
      <td>4.3</td>
      <td>0.8</td>
      <td>0</td>
      <td>30.2</td>
      <td>10.1</td>
      <td>20.7</td>
      <td>2.8</td>
      <td>12.5</td>
      <td>7.0</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>8.3</td>
      <td>96.7</td>
      <td>3.3</td>
      <td>84.7</td>
      <td>2.1</td>
      <td>0.4</td>
      <td>3.3</td>
      <td>2.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>6.5</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.6</td>
      <td>25.7</td>
      <td>9.5</td>
      <td>15.0</td>
      <td>5.9</td>
      <td>4.7</td>
      <td>1.8</td>
      <td>18.7</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>3.8</td>
      <td>9.0</td>
      <td>14.1</td>
      <td>24.8</td>
      <td>21.6</td>
      <td>11.5</td>
      <td>14.1</td>
      <td>0.7</td>
      <td>4.8</td>
      <td>16.6</td>
      <td>59.5</td>
      <td>15.0</td>
      <td>3.4</td>
      <td>89.0</td>
      <td>11.0</td>
      <td>8.9</td>
      <td>41.7</td>
      <td>23.7</td>
      <td>12.2</td>
      <td>9.3</td>
      <td>4.2</td>
      <td>1.8</td>
      <td>28.1</td>
      <td>44.2</td>
      <td>25.9</td>
      <td>77.4</td>
      <td>12.4</td>
      <td>4.9</td>
      <td>1.4</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>0.3</td>
      <td>0.4</td>
      <td>0</td>
      <td>0.3</td>
      <td>1.6</td>
      <td>98.9</td>
      <td>0.7</td>
      <td>0.4</td>
      <td>13.7</td>
      <td>14.1</td>
      <td>20.6</td>
      <td>25.0</td>
      <td>18.7</td>
      <td>5.7</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>71.3</td>
      <td>28.7</td>
      <td>0</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>10.8</td>
      <td>39.6</td>
      <td>29.7</td>
      <td>18.7</td>
      <td>0.0</td>
      <td>1.7</td>
      <td>8.4</td>
      <td>14.8</td>
      <td>75.0</td>
      <td>32.0</td>
      <td>14.5</td>
      <td>12.4</td>
      <td>8.9</td>
      <td>32.2</td>
      <td>29.3</td>
      <td>27.1</td>
      <td>13.1</td>
      <td>10.3</td>
      <td>7.3</td>
      <td>1.9</td>
      <td>11.0</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>16.2</td>
      <td>40.8</td>
      <td>22.8</td>
      <td>14.5</td>
      <td>1.8</td>
      <td>4.4</td>
      <td>11.4</td>
      <td>19.3</td>
      <td>8.8</td>
      <td>6.6</td>
      <td>49.6</td>
      <td>25.38</td>
      <td>9.54</td>
      <td>12.80</td>
      <td>12.68</td>
      <td>8.76</td>
      <td>5.86</td>
      <td>8.28</td>
      <td>10.86</td>
      <td>9.34</td>
      <td>8.62</td>
      <td>15.380</td>
      <td>35.68</td>
      <td>7.92</td>
      <td>7.76</td>
      <td>13.96</td>
      <td>9.74</td>
      <td>6.70</td>
      <td>3.24</td>
      <td>6.68</td>
      <td>3.80</td>
      <td>2.60</td>
      <td>5.24</td>
      <td>27.24</td>
      <td>29.88</td>
      <td>24.06</td>
      <td>8.48</td>
      <td>0.76</td>
      <td>12.10</td>
      <td>14.18</td>
      <td>22.38</td>
      <td>18.58</td>
      <td>26.46</td>
      <td>2.88</td>
      <td>45.64</td>
      <td>30.08</td>
      <td>7.08</td>
      <td>4.50</td>
      <td>22.64</td>
      <td>11.76</td>
      <td>6.74</td>
      <td>33.28</td>
      <td>6.94</td>
      <td>4.24</td>
      <td>22.64</td>
      <td>6.34</td>
      <td>3.82</td>
      <td>23.58</td>
      <td>5.90</td>
      <td>3.74</td>
      <td>20.14</td>
      <td>4.82</td>
      <td>2.12</td>
      <td>21.86</td>
      <td>0.48</td>
      <td>0.14</td>
      <td>1.72</td>
      <td>5.82</td>
      <td>5.08</td>
      <td>11.60</td>
      <td>26.44</td>
      <td>15.64</td>
      <td>35.540</td>
      <td>7.64</td>
      <td>8.80</td>
      <td>3.24</td>
      <td>21.84</td>
      <td>12.26</td>
      <td>7.48</td>
      <td>7.38</td>
      <td>11.16</td>
      <td>6.26</td>
      <td>2.20</td>
      <td>23.66</td>
      <td>3.66</td>
      <td>3.58</td>
      <td>2.88</td>
      <td>2.88</td>
      <td>2.6</td>
      <td>10.56</td>
      <td>6.66</td>
      <td>26.72</td>
      <td>16.98</td>
      <td>7.32</td>
      <td>6.54</td>
      <td>3.64</td>
      <td>23.86</td>
      <td>6.52</td>
      <td>3.84</td>
      <td>20.10</td>
      <td>9.64</td>
      <td>9.66</td>
      <td>17.86</td>
      <td>12.54</td>
      <td>38.64</td>
      <td>12.16</td>
      <td>6.62</td>
      <td>33.76</td>
      <td>2.22</td>
      <td>2.10</td>
      <td>0.00</td>
      <td>0.66</td>
      <td>0.5</td>
      <td>90.52</td>
      <td>9.48</td>
      <td>73.52</td>
      <td>26.48</td>
      <td>93.92</td>
      <td>6.08</td>
      <td>86.60</td>
      <td>13.40</td>
      <td>95.00</td>
      <td>5.00</td>
      <td>92.06</td>
      <td>7.94</td>
      <td>95.40</td>
      <td>4.60</td>
      <td>87.00</td>
      <td>13.00</td>
      <td>63.040</td>
      <td>36.960</td>
      <td>46.68</td>
      <td>75.68</td>
      <td>24.32</td>
      <td>99.34</td>
      <td>92.62</td>
      <td>7.38</td>
      <td>99.92</td>
      <td>92.80</td>
      <td>7.20</td>
      <td>96.44</td>
      <td>96.94</td>
      <td>3.06</td>
      <td>74.88</td>
      <td>91.58</td>
      <td>8.42</td>
      <td>35.90</td>
      <td>90.66</td>
      <td>9.34</td>
      <td>14.50</td>
      <td>74.300</td>
      <td>25.700</td>
      <td>2.44</td>
      <td>69.84</td>
      <td>30.16</td>
      <td>83.06</td>
      <td>16.94</td>
      <td>85.92</td>
      <td>14.08</td>
      <td>81.24</td>
      <td>18.76</td>
      <td>90.30</td>
      <td>9.70</td>
      <td>93.18</td>
      <td>6.82</td>
      <td>86.96</td>
      <td>13.04</td>
      <td>89.56</td>
      <td>88.66</td>
      <td>90.46</td>
      <td>19.24</td>
      <td>19.1</td>
      <td>19.34</td>
      <td>97.64</td>
      <td>2.36</td>
      <td>74.20</td>
      <td>25.80</td>
      <td>57.52</td>
      <td>42.48</td>
      <td>72.16</td>
      <td>27.84</td>
      <td>57.52</td>
      <td>42.48</td>
      <td>50.02</td>
      <td>49.98</td>
      <td>72.16</td>
      <td>27.84</td>
      <td>73.96</td>
      <td>26.04</td>
      <td>99.32</td>
      <td>0.68</td>
      <td>84.86</td>
      <td>15.14</td>
      <td>71.12</td>
      <td>28.88</td>
      <td>86.52</td>
      <td>13.48</td>
      <td>10.26</td>
      <td>89.74</td>
      <td>4.12</td>
      <td>14.42</td>
      <td>0.24</td>
      <td>21.44</td>
      <td>11.04</td>
      <td>9.48</td>
      <td>10.08</td>
      <td>9.56</td>
      <td>10.42</td>
      <td>9.58</td>
      <td>25.08</td>
      <td>9.60</td>
      <td>24.600</td>
      <td>16.720</td>
      <td>30.32</td>
      <td>11.08</td>
      <td>13.62</td>
      <td>15.68</td>
      <td>10.60</td>
      <td>2.50</td>
      <td>12.18</td>
      <td>14.08</td>
      <td>11.18</td>
      <td>40.02</td>
      <td>8.00</td>
      <td>12.18</td>
      <td>7.22</td>
      <td>21.62</td>
      <td>9.38</td>
      <td>10.22</td>
      <td>19.80</td>
      <td>14.82</td>
      <td>11.62</td>
      <td>7.36</td>
      <td>4.98</td>
      <td>10.34</td>
      <td>24.20</td>
      <td>15.48</td>
      <td>6.78</td>
      <td>76.20</td>
      <td>61.38</td>
      <td>65.60</td>
      <td>55.06</td>
      <td>13.08</td>
      <td>6.28</td>
      <td>0.78</td>
      <td>0.04</td>
      <td>26.88</td>
      <td>12.04</td>
      <td>14.68</td>
      <td>1.94</td>
      <td>13.88</td>
      <td>9.88</td>
      <td>1.76</td>
      <td>0.18</td>
      <td>10.26</td>
      <td>91.42</td>
      <td>8.58</td>
      <td>87.80</td>
      <td>2.08</td>
      <td>1.30</td>
      <td>1.42</td>
      <td>0.80</td>
      <td>0.74</td>
      <td>0.14</td>
      <td>5.70</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>15.50</td>
      <td>21.80</td>
      <td>11.68</td>
      <td>16.58</td>
      <td>8.84</td>
      <td>7.58</td>
      <td>2.58</td>
      <td>15.42</td>
      <td>0.54</td>
      <td>0.56</td>
      <td>2.02</td>
      <td>8.00</td>
      <td>17.88</td>
      <td>22.92</td>
      <td>19.90</td>
      <td>13.28</td>
      <td>14.92</td>
      <td>0.54</td>
      <td>2.32</td>
      <td>15.84</td>
      <td>55.28</td>
      <td>21.80</td>
      <td>4.26</td>
      <td>87.58</td>
      <td>12.42</td>
      <td>14.46</td>
      <td>34.50</td>
      <td>27.92</td>
      <td>12.34</td>
      <td>7.18</td>
      <td>3.62</td>
      <td>3.06</td>
      <td>24.6</td>
      <td>37.28</td>
      <td>35.08</td>
      <td>61.08</td>
      <td>22.60</td>
      <td>5.64</td>
      <td>4.00</td>
      <td>0</td>
      <td>5.00</td>
      <td>0.0</td>
      <td>1.10</td>
      <td>0.60</td>
      <td>0.44</td>
      <td>0.58</td>
      <td>3.08</td>
      <td>98.20</td>
      <td>1.20</td>
      <td>0.58</td>
      <td>9.10</td>
      <td>14.48</td>
      <td>18.08</td>
      <td>20.24</td>
      <td>23.36</td>
      <td>12.10</td>
      <td>1.62</td>
      <td>0.96</td>
      <td>69.22</td>
      <td>30.78</td>
      <td>0.08</td>
      <td>0.50</td>
      <td>5.40</td>
      <td>8.50</td>
      <td>32.88</td>
      <td>26.62</td>
      <td>25.96</td>
      <td>0.44</td>
      <td>2.12</td>
      <td>5.42</td>
      <td>15.54</td>
      <td>76.52</td>
      <td>32.32</td>
      <td>17.34</td>
      <td>14.08</td>
      <td>12.12</td>
      <td>24.14</td>
      <td>41.46</td>
      <td>20.44</td>
      <td>15.06</td>
      <td>7.30</td>
      <td>4.02</td>
      <td>2.60</td>
      <td>9.08</td>
      <td>1.94</td>
      <td>1.24</td>
      <td>4.66</td>
      <td>23.50</td>
      <td>20.84</td>
      <td>35.20</td>
      <td>12.62</td>
      <td>13.06</td>
      <td>11.02</td>
      <td>8.38</td>
      <td>10.36</td>
      <td>3.52</td>
      <td>53.64</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfm['coltype'].unique()
```




    array(['other', 'census', 'close'], dtype=object)




```python
# remove zipcodes with <50 children under age of 6:
sns.regplot(dfl['children_under6__CLPPP'],dfl['perc_bll_ge5__CLPPP'])
plt.show()
dflcurr = dfl.loc[dfl['children_under6__CLPPP']>100,:]
sns.regplot(dflcurr['perc_tested__CLPPP'],dflcurr['perc_bll_ge5__CLPPP'])
plt.show()



```


![png](testnotebook1_files/testnotebook1_48_0.png)



![png](testnotebook1_files/testnotebook1_48_1.png)



```python

```
