import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator



def main():
    path='/mnt/c/Codefiles/User_Activity_Profiling/'
    #path = 'hdfs://hadpdlnn01.p22.eng.sjc01.qualys.com/anarayan/'
    df = spark.read.csv(path+'emp_login_data1.csv', inferSchema=True, header=True)
    df.show(5)
    df.printSchema()

    # Converting Pyspark Dataframe to Pandas Dataframe
    # Load all the data into Driver's memory (using toPandas()), and Perform Data Pre-processing
    emp_df = df.toPandas()
    emp_df = emp_df.set_index('Date')

    emp_arr = remove_outlier(emp_df)
    emp_X_red_df = dimensionality_reduction(emp_arr)
    emp_X_red_df.to_csv('PCA_df.csv', index=False)
    
    # Converting Pandas df into Spark DF
    emp_df_sp = spark.createDataFrame(emp_X_red_df)

    input_columns = emp_df_sp.columns
    vecAssembler = VectorAssembler(inputCols=input_columns, outputCol="features")
    df_kmeans = vecAssembler.transform(emp_df_sp)
    df_kmeans.show(5)

    best_cluster, silhouette_scores = find_best_cluster(df_kmeans, 10)

    model, output_df = fit_best_cluster(df_kmeans, best_cluster)
    final_df = prepare_final_dataset(model, output_df, valid_density=10, alpha=1.5)
    final_df.show()

    final_df_pandas = final_df.toPandas()
    final_df_pandas.to_csv('final_df1.csv', index=False)
    print("Final_df is written in csv")
    return None


# Outlier Removal
def remove_outlier(df):
    # Calculate min, max from IQR using 25th,50th and 75th percentile
    quan_arr = df.quantile([0.25, 0.50, 0.75], axis=0, interpolation='midpoint').to_numpy().astype('float64')
    iqr_list = (quan_arr[2] - quan_arr[0])
    min_list = (quan_arr[0] - 1.5 * iqr_list)
    max_list = (quan_arr[2] + 1.5 * iqr_list)
    med_list = (quan_arr[1])
    # Replace outlier with Median value
    arr = df.to_numpy().astype('float64')
    for i in range(len(arr)):
        arr[i] = np.where((arr[i] < min_list) |
                          (arr[i] > max_list), med_list, arr[i])
    return arr


# PCA - Dimensionality Reduction
def dimensionality_reduction(arr):
    pca = PCA(svd_solver='auto')
    X_pca = pca.fit_transform(arr)
    prct_cont = pca.explained_variance_ratio_
    # tot_prct_cont = np.sum(prct_cont)
    prct_cont_thres = 0
    col_cnt = []
    for i in range(len(prct_cont)):
        if prct_cont_thres < 0.65:
            prct_cont_thres += prct_cont[i]
            col_cnt.append(prct_cont[i])
    X_red_df = pd.DataFrame(data=X_pca[0:, 0:len(col_cnt)], columns=[f"PC{x + 1}" for x in range(len(col_cnt))])
    #X_red_df = pd.DataFrame(data=X_pca[0:, 0:2], columns=[f"PC{x + 1}" for x in range(len(col_cnt))])

    return X_red_df


# Find the best cluster value
def find_best_cluster(df, clust_range):
    silhouette_scores = []
    best_sil_score = 0.0
    best_cluster = 0
    evaluator = ClusteringEvaluator()
    for k in range(2, clust_range):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(df)
        predictions = model.transform(df)
        silhouette = evaluator.evaluate(predictions)
        # print("Silhouette with squared euclidean distance = ", str(silhouette))
        silhouette_scores.append(str(silhouette))
        if silhouette > best_sil_score:
            best_sil_score = silhouette
            best_cluster = k
    print('Sil Score List:', silhouette_scores)
    print('Best Sil Score :', best_sil_score)
    print('Best K :', best_cluster)
    return best_cluster, silhouette_scores


def fit_best_cluster(df, best_cluster):
    kmeans = KMeans().setK(best_cluster).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df)
    output_df = model.transform(df)
    return model, output_df


def prepare_final_dataset(model, output_df, valid_density=10, alpha=1.5):
    cluster_centers = model.clusterCenters()
    clusters_center_dict = {int(i): [float(cluster_centers[i][j]) for j in range(len(cluster_centers[i]))] for i in
                            range(len(cluster_centers))}
    cluster_centers_df = spark.sparkContext.parallelize([(k,) + (v,) for k, v in clusters_center_dict.items()]).toDF(
        ['prediction', 'center'])
    output_df = output_df.withColumn('prediction', F.col('prediction').cast(IntegerType()))
    output_df = output_df.join(cluster_centers_df, on='prediction', how='left')
    get_dist = F.udf(lambda features, center: float(features.squared_distance(center)), FloatType())
    output_df = output_df.withColumn('dist', get_dist(F.col('features'), F.col('center')))

    counter = output_df.groupby("prediction").count()
    valid_density_dict = {row['prediction']: row['count'] for row in counter.collect() if
                          (row['count'] >= valid_density)}
    valid_density_df = spark.sparkContext.parallelize([(k,) + (v,) for k, v in valid_density_dict.items()]).toDF(
        ['prediction', 'density'])
    valid_density_centers_df = valid_density_df.join(cluster_centers_df, on='prediction', how='left')
    valid_df = output_df.filter(output_df.prediction.isin(list(valid_density_dict.keys())))
    summary_df = valid_df.groupby('prediction').agg(min(valid_df.dist).alias('min'), max(valid_df.dist).alias('max'),
                                                    mean(valid_df.dist).alias('mean'))
    final_df = summary_df.join(valid_density_centers_df, on='prediction', how='left')
    final_df = final_df.withColumn("Threshold", (alpha * col("mean") + col("max")))

    return final_df


if __name__ == "__main__":
    print("Executing the Main function")
    spark = SparkSession.builder.appName("Process KMeans").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Welcome to Pyspark Session!")
    main()
    print("Program Ends!")
else:
    print("Not a main function")


