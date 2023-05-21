#from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Laboras3').getOrCreate()


from pyspark.sql.functions import udf, log
from pyspark.sql.types import *

#nuo spark 3.0 
#spark.conf.set("spark.sql.legacy.setCommandRejectsSparkCoreConfs","false")

#spark.conf.set("spark.dynamicAllocation.enabled", "true")
#spark.conf.set("spark.executor.instances", 4)
#spark.conf.set("spark.executor.cores", 4)

text_file = spark.sparkContext.textFile("duom_cut.txt")
#text_file = sc.textFile("hdfs:///user/stud0/labt/duom_cut.txt")

def parsinam(line):
	#line=line[2:len(line)-2]
	return line[2:len(line)-2].split('}}{{')

def parsinam2(line):
	objs = line.split('}{')
	k1=None
	k3=None
	for at in objs:
		temp = at.split('=')
		if(len(temp)<2): 
			break
		key,val=at.split('=')
		if(key == 'marsrutas'):
			k1=val
		if(key == 'sustojimo data'):
			k3=val
	if(k1!=None and k3!=None):
		return (k1+"_"+k3,(k3,len(str(k3))))
	#else:
	#	return (0,(0,0))

fmap = text_file.flatMap(parsinam)
#fmap = text_file.flatMap(lambda line:line[2:len(line)-2].split('}}{{'))
mmap = fmap.map(parsinam2)

##Jusu darbas cia:

##Kodas, kito failo nuskaitymas ... duomenu agregavimas
routes = spark.read.option("header",True).csv("RouteSummary.txt")
routes.printSchema()
routes = routes.drop("M", "BendrasAtstumas","BendrasLaikas")

def makeID(str1, str2):
    return str1+"_"+str2
	
makeID_UDF = udf(lambda z1,z2: makeID(z1,z2),StringType())

routes2 = routes.withColumn('ID', makeID_UDF("marsrutas", "sustojimo data")).drop("marsrutas", "sustojimo data")
routes2.printSchema()

routes2.write.csv("lentele3")

#listas = routes2.rdd.collect()
#textfile = open("out.txt", "w")
#for element in listas:
#	textfile.write(element + "\n")
#textfile.close()






##training data formato: ("prognozuojama reiskme", "parametras")
training = mmap.toDF()

#regression ...

from pyspark.ml.regression import LinearRegression

# Load training data
#training = spark.read.format("libsvm")\
#    .load("sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)




#dfFromRDD1.show()
#training = training.withColumnRenamed('marsrutas', 'parametrai')


training.printSchema()

pandasDF = training.toPandas()
pandasDF.head()

labels = pandasDF['_1'].to_list()

values = pandasDF['_2'].to_list()
print(labels)
print(values)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
axes[2].scatter(labels,labels, s = 10)

#training.plot.scatter(x='label',
#                      y='label',
#                      c='ats',colormap='viridis',ax=axes[0])
plt.show()
