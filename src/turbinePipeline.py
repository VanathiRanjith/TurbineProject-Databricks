from pyspark.sql import SparkSession
from pyspark.sql import functions as F


class WindTurbinePipeline:
    """
    A class to encapsulate the data pipeline for processing wind turbine data.
    This class reads data from the Bronze layer, detects outliers, handles missing values, writes the cleaned data to the Silver layer,
    summarizes the power output, detects anomalies, and stores results in the Gold layer.
    """

    def __init__(self, storage_account, container, access_key):
        """
        Initializes the WindTurbinePipeline object.

        :param storage_account: Name of the Azure storage account.
        :param container: Name of the container in the storage account.
        :param access_key: Access key for the storage account.
        """
        self.storage_account = storage_account
        self.container = container
        self.bronze_layer_path = f"wasbs://{container}@{storage_account}.blob.core.windows.net/Bronze/"
        self.silver_layer_path = f"wasbs://{container}@{storage_account}.blob.core.windows.net/Silver"
        self.gold_layer_path = f"wasbs://{container}@{storage_account}.blob.core.windows.net/Gold"

        self.spark = self.initialize_spark_session()
        self.configure_access_key(access_key)

    def initialize_spark_session(self, app_name="WindTurbineProcessing"):
        """
        Initializes a Spark session.

        :param app_name: Name of the Spark application.
        :return: A SparkSession object.
        """
        try:
            return SparkSession.builder.appName(app_name).getOrCreate()
        except Exception as e:
            print(f"Error in initializing Spark session: {e}")
            return None

    def configure_access_key(self, access_key):
        """
        Configures the Azure storage account access key.

        :param access_key: The access key for the Azure storage account.
        """
        self.spark.conf.set(f"fs.azure.account.key.{self.storage_account}.blob.core.windows.net", access_key)

    def read_bronze_layer(self):
        """
        Reads CSV files from the Bronze layer.

        :return: A DataFrame containing the data read from the Bronze layer.
        """
        try:
            df = self.spark.read.csv(self.bronze_layer_path + "/*.csv", header=True, inferSchema=True)
            return df
        except Exception as e:
            print(f"Error reading CSV from bronze layer: {e}")
            return None

    def impute_missing_values(self, df):
        """
        Handles missing values in the DataFrame by filling null values with the median or a placeholder.

        :param df: Input DataFrame containing wind turbine data.
        :return: DataFrame with missing values handled.
        """
        try:
            numeric_columns = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]
            for col in numeric_columns:
                median_value = df.approxQuantile(col, [0.5], 0.05)[0]
                df = df.fillna({col: median_value})

            string_columns = [col for col, dtype in df.dtypes if dtype == 'string']
            df = df.fillna({col: 'Unknown' for col in string_columns})

            return df
        except Exception as e:
            print(f"Error in handling missing values: {e}")
            return None

    def detect_outliers(self, df):
        """
        Detects outliers in the wind turbine data based on the interquartile range (IQR).

        :param df: Input DataFrame containing wind turbine data.
        :return: DataFrame with outlier detection columns added.
        """
        try:
            columns_to_check = ['wind_speed', 'power_output', 'wind_direction']
            quantiles = df.select(*[F.percentile_approx(c, [0.25, 0.75], 10000).alias(f"{c}_quantiles") for c in columns_to_check]).first()

            bounds = {}
            for col in columns_to_check:
                Q1, Q3 = quantiles[f"{col}_quantiles"]
                IQR = Q3 - Q1
                bounds[col] = {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR}

            for col in columns_to_check:
                df = df.withColumn(f"{col}_is_outlier", (F.col(col) < bounds[col]['lower']) | (F.col(col) > bounds[col]['upper']))

            df = df.withColumn("is_any_outlier", F.array_contains(F.array(*[F.col(f"{col}_is_outlier") for col in columns_to_check]), True))
            return df
        except Exception as e:
            print(f"Error in detecting outliers: {e}")
            return None

    def detect_anomalies(self, df):
        """
        Detects anomalies in power output based on deviation from the mean.
        """
        try:
            stats_df = df.groupBy("turbine_id").agg(
                F.mean("power_output").alias("mean_output"),
                F.stddev("power_output").alias("std_dev")
            )

            df_with_stats = df.join(stats_df, on="turbine_id")
            anomalies = df_with_stats.filter(
                (df_with_stats["power_output"] < (df_with_stats["mean_output"] - 2 * df_with_stats["std_dev"])) |
                (df_with_stats["power_output"] > (df_with_stats["mean_output"] + 2 * df_with_stats["std_dev"]))
            )
            return anomalies.select(df.columns)
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return None

    def write_to_silver(self, df):
        """
        Writes the cleaned DataFrame to the Silver layer in Parquet format.
        """
        try:
            df.write.mode("overwrite").parquet(self.silver_layer_path)
        except Exception as e:
            print(f"Error writing to silver layer: {e}")

    def summarize_statistics(self, df):
        """
        Summarizes power output statistics and saves results to the Gold layer.
        """
        try:
            df_with_date = df.withColumn("date", F.to_date("timestamp"))
            summary_stats = df_with_date.groupBy("date", "turbine_id").agg(
                F.min("power_output").alias("min_power_output"),
                F.max("power_output").alias("max_power_output"),
                F.avg("power_output").alias("avg_power_output")
            ).orderBy("date")
            summary_stats.write.mode("append").parquet(self.gold_layer_path)
        except Exception as e:
            print(f"Error summarizing statistics: {e}")

    def process(self):
        df = self.read_bronze_layer()
        df = self.impute_missing_values(df)
        df = self.detect_outliers(df)
        df = self.detect_anomalies(df)
        self.write_to_silver(df)
        self.summarize_statistics(df)


if __name__ == "__main__":
    pipeline = WindTurbinePipeline("collibri", "codingc", "ACCESS_KEY")
    pipeline.process()
