import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts//')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import winsorize as wz
import forecasting as fcst
from constants import ATTR, OUTLIER_METHOD, TIME_INVARIANT_ATTR, TIME_VARIANT_ATTR, SPLIT_PCT, IDENTIFIERS


def get_n_clusters(df,n_cluster_ll,n_cluster_ul):
    cluster_dict = {}
    for iteration in range(n_cluster_ll,n_cluster_ul):
        clusterer = KMeans(n_clusters=iteration, random_state=42)
        cluster_labels = clusterer.fit_predict(df)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        cluster_dict[iteration] = silhouette_avg
        print('Score for clusters # {} is {}'.format(iteration,silhouette_avg))
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1])}
    n_clusters = list(cluster_dict.keys())[len(cluster_dict)-1]

    return n_clusters


if __name__ == '__main__':
    ads = ads_crtn.create_ads()
    train_ads, test_ads = eda.train_test_split(ads,'INVERTER_ID',SPLIT_PCT)

    #leaving out total yield since it value since it value increase exponentially and rightly so
    outlier_feature = [feature for feature in ATTR if feature != 'TOTAL_SYIELD'] 
    clip_model = wz.winsorize(OUTLIER_METHOD)
    clip_model.fit(train_ads[outlier_feature],'INVERTER_ID')
    train_ads[outlier_feature] = clip_model.transform(train_ads[outlier_feature])
    test_ads[outlier_feature] = clip_model.transform(test_ads[outlier_feature])

    #Filtering for plant 4136001, since we are having trouble with that only

    train_ads_clst = train_ads.loc[train_ads['PLANT_ID']==4136001.0].reset_index(drop=True)
    test_ads_clst = test_ads.loc[test_ads['PLANT_ID']==4136001.0].reset_index(drop=True)


    #running the below loop to give us the number of clusters that are optimal for this clustering exercise
    #10 clusters are coming out to be most optimal, so lets go ahead witthat

    n_clust = get_n_clusters(train_ads_clst.loc[:,~train_ads_clst.columns.isin(IDENTIFIERS)],2,11)

    #kmeans clustering with 10 clusters
    clusterer = KMeans(n_clusters=n_clust, random_state=42)
    cluster_labels = clusterer.fit_predict(train_ads_clst.loc[:,~train_ads_clst.columns.isin(IDENTIFIERS)])

    train_ads_clst = pd.concat([pd.DataFrame({'CLUSTER_ID':list(cluster_labels)}),train_ads_clst],axis=1)

    #putting the cluster IDs in the test ads as well
    invtr_clst_map = train_ads_clst[['INVERTER_ID','CLUSTER_ID']].drop_duplicates().reset_index(drop=True)
    test_ads_clst = pd.merge(test_ads_clst,invtr_clst_map,on='INVERTER_ID',how='inner')

    train_ads_clst = fcst.complt_ts_panel_lvl(train_ads_clst,'INVERTER_ID','DATE')
    test_ads_clst = fcst.complt_ts_panel_lvl(test_ads_clst,'INVERTER_ID','DATE')

    #making ADS stationary
    non_stnry_invtr_list = eda.return_non_stnry_invtr_list(train_ads_clst,'INVERTER_ID')
    train_ads_clst = eda.make_ads_stnry(train_ads_clst,non_stnry_invtr_list,'INVERTER_ID','DATE')

    non_stnry_invtr_list = eda.return_non_stnry_invtr_list(test_ads_clst,'INVERTER_ID')
    test_ads_clst = eda.make_ads_stnry(test_ads_clst,non_stnry_invtr_list,'INVERTER_ID','DATE')

    #Creating lagged features which are lagged by exactly 2 days so that they can then be used for prediction

    train_ads_clst = feat_engg.create_lagged_features(train_ads_clst,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
    test_ads_clst = feat_engg.create_lagged_features(test_ads_clst, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

    features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
    features = features + TIME_INVARIANT_ATTR

    models,metrics = regr.panel_wise_model(train_ads_clst,test_ads_clst,
                                            'CLUSTER_ID','PER_TS_YIELD',
                                            RandomForestRegressor(random_state=42),
                                            features=features)