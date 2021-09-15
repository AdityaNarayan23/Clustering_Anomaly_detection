import warnings  
warnings.filterwarnings('ignore')

import csv
import happybase as hb
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#########################################################
####                    FUNCTIONS                    ####
#########################################################

#Establish Connection with Hbase
def establish_Hbase_connection(host,port,namespace,table_name):
    conn = hb.Connection(host=host,port=port,table_prefix = namespace,table_prefix_separator = ":")
    table = conn.table(table_name)
    conn.open()
    print("HBase connection is Open via thrift server")
    return table,conn

#Close Connection with Hbase
def close_Hbase_connection(conn):
    conn.close()
    print("HBase connection is Close via thrift server")
    return None

#Pull Data from Hbase Table
def get_data_Hbase(host,port,namespace,table_name):

    #establish connection with Hbase
    host='localhost'
    port=9090
    namespace = "emp_data"
    table_name = "login_data"
    table,conn = establish_Hbase_connection(host,port,namespace,table_name)

    #pull data from Hbase
    row_key_list = []
    col_list_arr = []
    val_list_arr = []
    df = pd.DataFrame()

    for row_key, col_val in table.scan():
        row_key = row_key.decode("utf-8")
        col_list = []
        val_list = []
        for col, val in col_val.items():
            col_list.append(col.decode("utf-8"))  #this will be the columns in the dataframe
            val_list.append(int(val.decode("utf-8"))) #this will be the values for respective columns in the dataframe
        temp = pd.DataFrame(data=[val_list],columns=col_list,index=[row_key])
        df = df.append(temp)

    print("Employee DataFrame  :\n",df.head())

    #close Hbase connection
    close_Hbase_connection(conn)

    return df

#Outlier Removal
def remove_outlier(df):
    #Calculate min, max from IQR using 25th,50th and 75th percentile 
    quan_arr = df.quantile([0.25,0.50,0.75],axis=0,interpolation='midpoint').to_numpy().astype('float64')
    iqr_list = (quan_arr[2] - quan_arr[0])
    min_list = (quan_arr[0] - 1.5*iqr_list)
    max_list = (quan_arr[2] + 1.5*iqr_list)
    med_list = (quan_arr[1])
    #Replace outlier with Median value
    arr = df.to_numpy().astype('float64')
    for i in range(len(arr)):
        arr[i] = np.where((arr[i] < min_list) |
                             (arr[i] > max_list), med_list, arr[i])
    return arr

#PCA - Dimensionality Reduction
def dimensionality_reduction(arr):
    pca = PCA(svd_solver = 'auto')
    X_pca = pca.fit_transform(arr)
    prct_cont = pca.explained_variance_ratio_
    #tot_prct_cont = np.sum(prct_cont)
    prct_cont_thres = 0
    col_cnt = []
    for i in range(len(prct_cont)):
        if prct_cont_thres < 0.65:
            prct_cont_thres += prct_cont[i]
            col_cnt.append(prct_cont[i])
    X_red_df = pd.DataFrame(data=X_pca[0:,0:len(col_cnt)],columns=[f"PC{x+1}" for x in range(len(col_cnt))])
    
    return X_red_df

#Determining best cluster size for KMeans Clustering 
def K_best_cluster(df,clust_range,):
    silhouette_scores = []
    max_sil_score = 0
    for x in clust_range:
        km = KMeans(n_clusters=x, init='k-means++')
        sil_score = silhouette_score(df,km.fit_predict(df))
        print("Silhouette Score :",sil_score)
        if sil_score > max_sil_score:
            max_sil_score = sil_score 
            best_cluster = x
        silhouette_scores.append(sil_score)
    print("Max Sil Score:",max_sil_score,"for the best cluster:", best_cluster)
    return(best_cluster,silhouette_scores,)

#Fit best cluster and determine cluster centroids and radius respectively
def fit_best_K(df,best_cluster,valid_density,):
    km_final = KMeans(n_clusters=best_cluster,init='k-means++')
    km_final.fit_predict(df)
    cluster_centers = km_final.cluster_centers_
    clust_center_dict = {}
    print("Cluster Centers : ",cluster_centers)
    for i in range(len(cluster_centers)):
        clust_center_dict[i] = cluster_centers[i]
    print("Cluster Dict :" ,clust_center_dict)
    print("Cluster Labels : ",km_final.labels_)
    counter = Counter(km_final.labels_)
    #print("Counter :",counter)
    valid_density_dict = {}
    for x in counter:
        key = x
        #print(key)
        if (counter[key] >= valid_density):
            #print("Counter[key,cluster density]:", x,counter[key])
            valid_density_dict[x] = counter[key] 
    
    #Distance of data-points from cluster centroids
    alldist = km_final.fit_transform(df)
    mindist = np.min(alldist, axis=1)
    #print(mindist)
    #Assign the labels to the data points
    df['Cluster'] = pd.Series(km_final.labels_).values
    df['Dist'] = pd.Series(mindist).values
    
    return(clust_center_dict,valid_density_dict,df)

#Write Final Dataframe with all the required parameters
def generate_final_df(clust_center_dict,valid_density_dict,output_df,alpha,beta):
    valid_df = output_df[output_df['Cluster'].isin(valid_density_dict.keys())]

    #Fetch max cluster distance of data point of valid cluster
    max_df = pd.DataFrame({'Max_dist' : valid_df.groupby('Cluster')['Dist'].max()}).reset_index()
    min_df = pd.DataFrame({'Min_dist' : valid_df.groupby('Cluster')['Dist'].min()}).reset_index()
    mean_df = pd.DataFrame({'Mean_dist' : valid_df.groupby('Cluster')['Dist'].mean()}).reset_index()

    merge_df = pd.merge(max_df,min_df, on='Cluster')
    final_df = pd.merge(merge_df,mean_df, on='Cluster')

    final_df['Cluster_center'] = ''
    final_df['Cluster_density'] = ''
    for i in range(len(final_df)):
        final_df['Cluster_center'][i] = clust_center_dict[final_df['Cluster'][i]]
        final_df['Cluster_density'][i] = valid_density_dict[final_df['Cluster'][i]]

    alpha = alpha
    beta = final_df['Max_dist']
    final_df['Threshold'] = alpha*final_df['Mean_dist'] +  beta
    print("Final df with all the parameters :\n",final_df)

    return final_df

#write into Hbase table
def write_into_hbase(host,port,namespace,table_name,df):
    #establish connection with Hbase
    host='localhost'
    port=9090
    namespace = "emp_data"
    table_name = "cluster_data"
    table,conn = establish_Hbase_connection(host,port,namespace,table_name)

    #write data into Hbase
    out_row_key_list = []
    out_col_list = []
    out_val_list = []

    out_col_list = list(df.columns)
    print("out_col_list :", out_col_list)
    out_row_key_list = list(df.index)
    print("out_row_key_list :", out_row_key_list)
    out_val_list = df.values
    print("out_val_list :", out_val_list)
       
    for i in range(len(out_row_key_list)):
        print("Row Key:",out_row_key_list[i])
        table.put(str(out_row_key_list[i]), { "cluster_info:CLUSTER_ID": str(out_val_list[i][0]),
                                 "cluster_info:MAX_DIST": str(out_val_list[i][1]), "cluster_info:MIN_DIST": str(out_val_list[i][2]),
                                 "cluster_info:MEAN_DIST": str(out_val_list[i][3]),
                                 "cluster_info:CLUSTER_CENTER": str(out_val_list[i][4]),
                                 "cluster_info:CLUSTER_DENSITY": str(out_val_list[i][5]), "cluster_info:THRESHOLD": str(out_val_list[i][6])
                                }  
              )
        print("Row Key, inserted into HBase :",out_row_key_list[i])

    #close Hbase connection
    close_Hbase_connection(conn)

    return None

#########################################################
####               MAIN FUNCTIONS                    ####
#########################################################

def main():
    
    #pull data from Hbase
    host='localhost'
    port=9090
    namespace = "emp_data"
    table_name = "login_data"

    print("Pull Data from HBase")
    emp_df = get_data_Hbase(host,port,namespace,table_name)

    #remove outlier
    print("Remove Outlier from Data")
    emp_arr = remove_outlier(emp_df)
    #print(emp_arr)

    #Dimensionality reduction
    print("Dimensionality reduction on Data")
    emp_X_red_df = dimensionality_reduction(emp_arr)
    #print(emp_X_red_df)

    #find best cluster
    print("KMeans Cluster tuning to find best cluster")
    best_cluster,silhouette_scores = K_best_cluster(emp_X_red_df,range(2,6))

    #fit best cluster and save parameters
    print("Fit Cluster with best K value")
    clust_center_dict,valid_density_dict,output_df = fit_best_K(emp_X_red_df,best_cluster,10)

    #generate final dataframe
    print("Generate Final Dataframe")
    alpha = 1.5
    beta = 0
    final_df = generate_final_df(clust_center_dict,valid_density_dict,output_df,alpha,beta)

    #Write in local as a csv file
    print("Write final dataframe in local")
    final_df.to_csv('final_df.csv',index=False)

    #write data into Hbase
    host='localhost'
    port=9090
    namespace = "emp_data"
    table_name = "cluster_data"

    print("Write data into HBase table")
    write_into_hbase(host,port,namespace,table_name,final_df)

    return None

#########################################################
####            ENTRYPOINT                           ####
#########################################################

print("Welcome to Python Session!")

if __name__ == "__main__":
    print("Executing the Main function")
    main()
else:
    print("Not a main function")

print("Program Ends!")

