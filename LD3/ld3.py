import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *

spark = SparkSession.builder.appName('Laboras3').getOrCreate()

text_file = spark.sparkContext.textFile("duom_cut.txt")
# text_file = spark.sparkContext.textFile("duom_full.txt")


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
routes = routes.drop("M", "BendrasAtstumas", "BendrasSvoris", "BendraKaina")


def makeID(str1, str2):
    return str1+"_"+str2


makeID_UDF = udf(lambda z1, z2: makeID(z1, z2), StringType())


def convert_time(string):
    hours, minutes = string.split(':')
    return int(hours) * 60 + int(minutes)


convert_time_udf = udf(lambda z: convert_time(z), IntegerType())

# Make ID and drop those columns
routes2 = routes.withColumn('ID', makeID_UDF(
    "marsrutas", "sustojimo data"))\
    .drop("marsrutas", "sustojimo data")

routes2 = routes2.filter(col('BendrasLaikas').isNotNull())\
    .withColumn('BendrasLaikas', convert_time_udf('BendrasLaikas'))
routes2.printSchema()

for u in unique.collect():
    print(f'Value "Masinos tipas": {u}')

    # Filter by 'Masinos tipas'
    filtered_data = mmap.filter(lambda x: x[1][0] == u)\
        .map(lambda x: (x[0], x[1][1]))

    # Convter RDD to DataFrame
    data_frame = filtered_data.toDF(['ID', 'Svoris'])
    joined_data_frame = data_frame.join(routes2, 'ID')
    joined_data_frame = joined_data_frame.withColumn(
        'BendrasLaikas', joined_data_frame.BendrasLaikas.cast(IntegerType()))

    # Create feature vector
    vector_assembler = VectorAssembler(
        inputCols=['Svoris'], outputCol='features')
    assembled_vector = vector_assembler.transform(joined_data_frame)\
        .drop('Svoris').drop('ID')

    linear_regression = LinearRegression(
        maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='features', labelCol='BendrasLaikas')
    linear_regression_model = linear_regression.fit(assembled_vector)

    intercept = linear_regression_model.intercept
    coefficients = linear_regression_model.coefficients[0]

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(coefficients))
    print("Intercept: %s" % str(intercept))

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

    labels = pandasDF['BendrasLaikas'].to_list()
    values = pandasDF['features'].to_list()

    print(labels)
    print(values)

    X = pandasDF['BendrasLaikas'].to_list()
    Y = pandasDF['features'].to_list()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
    axes.scatter(X, Y, s=10)
    axes.set_xlabel('BendrasLaikas')
    axes.set_ylabel('Svoris')
    # x_line = [min(X), max(Y)]
    # y_line = [intercept + coefficients * x_line[0],
    #           intercept + coefficients * x_line[1]]
    # axes.plot(x_line, y_line, color='red')

    plt.show()
