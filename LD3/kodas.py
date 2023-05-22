!pip install pyspark
# from pyspark import SparkContext, SparkConf

# conf = SparkConf().setAppName('MyApp')
# sc = SparkContext(conf=conf)

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Laboras3').getOrCreate()


from pyspark.sql.functions import udf, log, col
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
text_file = spark.sparkContext.textFile("duom_cut.txt")

def parsinam(line):
	return line[2:len(line)-2].split('}}{{')

def parsinam2(line):
	objs = line.split('}{')
	marsrutas=None
	sustojimo_data=None
	masinos_tipas=None
	siuntu_skaicius=None

	for at in objs:
		temp = at.split('=')
		if(len(temp)<2): 
			break
		key,val=at.split('=')
		if(key == 'marsrutas'):
			marsrutas = val
		if(key == 'sustojimo data'):
			sustojimo_data=val
		if(key == 'Masinos tipas'):
			masinos_tipas = val
		if (key =="siuntu skaicius"):
			siuntu_skaicius = val
	if(marsrutas!=None and sustojimo_data!=None and masinos_tipas!=None and siuntu_skaicius!=None):
			key = f"{marsrutas}_{sustojimo_data}"
			return ((marsrutas+'_'+sustojimo_data), (masinos_tipas, int(siuntu_skaicius)))

fmap = text_file.flatMap(parsinam)
mmap = fmap.map(parsinam2).filter(lambda x: x is not None).reduceByKey(lambda a,b: (a[0] if a[0]!=None else a[0], a[1] + b[1])) # Agreguojam duomenis pagal rakta
#Isrenkame masinos tipus
tipai = fmap.map(parsinam2).filter(lambda kv: kv != None).map(lambda t:(t[1][0],0)).reduceByKey(lambda a,b : a).map(lambda t:t[0])

#Kodas, kito failo nuskaitymas ... duomenu agregavimas
routes = spark.read.option("header",True).csv("RouteSummary.txt")
routes = routes.drop("M", "BendrasAtstumas", "BendrasSvoris", "BendrasLaikas") #ismetame nereikalingus duomenis

def makeID(str1, str2):
    return str1+"_"+str2
	
makeID_UDF = udf(lambda z1,z2: makeID(z1,z2),StringType())
routes = routes.withColumn('ID', makeID_UDF("marsrutas", "sustojimo data")).drop("marsrutas", "sustojimo data")
routes = routes.withColumn('BendraKaina', col('BendraKaina').cast(FloatType())) # Einame per BendraKaina stulpeli ir paverciame is string i float tipa
routes = routes.filter(col('BendraKaina').isNotNull()) # Taip pat isfiltruojame duomenis is BendraKaina stulpelio kurie yra ne None ir t.t


for tipas in tipai.collect():
  print(f'Masinos tipas - {tipas}')

  # Issirenkame duomenis pagal masinos tipa ir susimapiname kad butu (ID, siuntuSkaicius)
  filteredData = mmap.filter(lambda x : x[1][0] == tipas).map(lambda t: (t[0], t[1][1]))

  data = filteredData.toDF(["ID", "SiuntuSkaicius"]) # konvertuojam RDD elementus i dataset'a  
  joined_data = data.join(routes, "ID") # apjungiame dataset'us pagal "ID"
  # Features = siuntu skaicius (vektorius)
  # Label = BendraKaina (dataset'as)
  assembler = VectorAssembler(inputCols=["SiuntuSkaicius"], outputCol="features")

  # Transform the data by assembling the features
  assembled_data = assembler.transform(joined_data)
  assembled_data = assembled_data.drop('SiuntuSkaicius') # ismetam siuntu skaicius, kuris yra nebereikalingas, nes turim features vektoriu, kuriame yra siuntuskaicius
  assembled_data = assembled_data.drop('ID')             # ismetame ID nes irgi nebera reikalingas, nes buvo reikalingas tik, kad apjungtume duomenis

  # Create a LinearRegression model and fit the data
  lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol="features", labelCol="BendraKaina")
  lrModel = lr.fit(assembled_data)

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

  pandasDF = assembled_data.toPandas()
  pandasDF.head()

  labels = pandasDF['BendraKaina'].to_list()

  values = pandasDF['features'].to_list()

  import matplotlib.pyplot as plt

  plt.figure(figsize=(10, 5))
  plt.scatter(labels, values, s=10)
  plt.xlabel('BendraKaina')
  plt.ylabel('SiuntuSkaicius')

  plt.show()
