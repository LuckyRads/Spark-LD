import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, to_timestamp, udf
from pyspark.sql.types import *

spark = SparkSession.builder.appName('Laboras3').getOrCreate()

text_file = spark.sparkContext.textFile("duom_cut.txt")


def parsinam(line):
    return line[2:len(line)-2].split('}}{{')


def parsinam2(line):
    objs = line.split('}{')
    k1 = None
    k3 = None
    k4 = None
    k5 = None
    for at in objs:
        temp = at.split('=')
        if (len(temp) < 2):
            break
        key, val = at.split('=')
        if (key == 'marsrutas'):
            k1 = val
        if (key == 'sustojimo data'):
            k3 = val
        if (key == 'Masinos tipas'):
            k4 = val
        if (key == 'svoris'):
            k5 = val
    if (k1 != None and k3 != None and k4 != None and k5 != None):
        return (k1+"_"+k3, (k4, float(k5)))
    else:
        return (0, (1, "invalid"))


fmap = text_file.flatMap(parsinam)
# Parse file and filter out invalid lines
mmap = fmap.map(parsinam2).filter(lambda x: x[0] != 0)

mmap.take(5)

# Aggregate byt key and sum values (svoris)
mmap = mmap.reduceByKey(lambda x, y: (x[0], x[1]+y[1]))
mmap.take(5)

# Get unique 'Masinos tipas' values
unique = mmap.map(lambda x: x[1][0]).distinct()
unique.collect()

# Jusu darbas cia:

# Kodas, kito failo nuskaitymas ... duomenu agregavimas
routes = spark.read.option("header", True).csv("RouteSummary.txt")
routes.printSchema()
# Drop unused columns
routes = routes.drop("M", "BendrasAtstumas", "BendrasLaikas", "BendraKaina")


def makeID(str1, str2):
    return str1+"_"+str2


makeID_UDF = udf(lambda z1, z2: makeID(z1, z2), StringType())

routes2 = routes.withColumn('ID', makeID_UDF(
    "marsrutas", "sustojimo data"))\
    .drop("marsrutas", "sustojimo data")\
    .filter(col('BendrasSvoris').isNotNull())\
    .withColumn('BendrasSvoris', col('BendrasSvoris').cast(FloatType()))
routes2.printSchema()

for u in unique.collect():
    print(f'Value "Masinos tipas": {u}')

    # Filter by 'Masinos tipas'
    filtered_data = mmap.filter(lambda x: x[1][0] == u)\
        .map(lambda x: (x[0], x[1][1]))

    # Convter RDD to DataFrame
    data_frame = filtered_data.toDF(['ID', 'Svoris'])
    joined_data_frame = data_frame.join(routes2, 'ID')

    # Create feature vector
    vector_assembler = VectorAssembler(
        inputCols=['Svoris'], outputCol='features')
    assembled_vector = vector_assembler.transform(joined_data_frame)\
        .drop('Svoris').drop('ID')
    
    linear_regression = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='features', labelCol='BendrasSvoris')
    linear_regression_model = linear_regression.fit(assembled_vector)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(linear_regression_model.coefficients))
    print("Intercept: %s" % str(linear_regression_model.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = linear_regression_model.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    assembled_vector.printSchema()

    pandasDF = assembled_vector.toPandas()
    pandasDF.head()

    labels = pandasDF['BendrasSvoris'].to_list()
    values = pandasDF['features'].to_list()

    print(labels)
    print(values)


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes[2].scatter(labels, labels, s=10)
    
    plt.show()




# # ----------------------------

# # training data formato: ("prognozuojama reiskme", "parametras")
# training = mmap.toDF(['ID', 'BendrasLaikas'])

# # regression ...


# # Load training data
# # training = spark.read.format("libsvm")\
# #    .load("sample_linear_regression_data.txt")

# lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# # Fit the model
# lrModel = lr.fit(training)

# # Print the coefficients and intercept for linear regression
# print("Coefficients: %s" % str(lrModel.coefficients))
# print("Intercept: %s" % str(lrModel.intercept))

# # Summarize the model over the training set and print out some metrics
# trainingSummary = lrModel.summary
# print("numIterations: %d" % trainingSummary.totalIterations)
# print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
# trainingSummary.residuals.show()
# print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# print("r2: %f" % trainingSummary.r2)


# # dfFromRDD1.show()
# # training = training.withColumnRenamed('marsrutas', 'parametrai')


# training.printSchema()

# pandasDF = training.toPandas()
# pandasDF.head()

# labels = pandasDF['_1'].to_list()

# values = pandasDF['_2'].to_list()
# print(labels)
# print(values)


# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
# axes[2].scatter(labels, labels, s=10)

# # training.plot.scatter(x='label',
# #                      y='label',
# #                      c='ats',colormap='viridis',ax=axes[0])
