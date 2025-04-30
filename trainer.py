import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms

class SparkConfig:
    appName = "CIFAR10"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 43210
    batch_interval = 2

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split:str, 
                 spark_config:SparkConfig, 
                 transforms: Transforms) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc,self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc,self.ssc,self.sqlContext,self.sparkConf,self.transforms)

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)])

            df = self.sqlContext.createDataFrame(rdd, schema)
            
            predictions, accuracy, precision, recall, f1 = self.model.train(df)

            print("="*10)
            print(f"Predictions = {predictions}")
            print(f"Accuracy = {accuracy}")
            print(f"Precision = {precision}")
            print(f"Recall = {recall}")
            print(f"F1 Score = {f1}")
            print("="*10)
        
        print("Total Batch Size of RDD Received :",rdd.count())
        print("+"*20)

        
        