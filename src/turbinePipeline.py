# Databricks Wind Turbine Data Pipeline

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException

# Define ADLS Gen2 Storage Variables
STORAGE_ACCOUNT_NAME = "your_adls_account"
CONTAINER_NAME = "your_container"
MOUNT_NAME = "your_mount_name"

# Define ADLS Paths
BRONZE_PATH = f"dbfs:/mnt/{MOUNT_NAME}/bronze/wind_turbine_data"
SILVER_PATH = f"dbfs:/mnt/{MOUNT_NAME}/silver/wind_turbine_data"
GOLD_PATH = f"dbfs:/mnt/{MOUNT_NAME}/gold/wind_turbine_summary"

# Define Checkpoint Paths for Streaming
CHECKPOINT_SILVER = f"dbfs:/mnt/{MOUNT_NAME}/checkpoints/silver"
CHECKPOINT_GOLD = f"dbfs:/mnt/{MOUNT_NAME}/checkpoints/gold"

# Mount ADLS Gen2 Storage
def mount_adls():
    """Mounts Azure Data Lake Storage (ADLS Gen2) to Databricks."""
    configs = {
        "fs.azure.account.auth.type": "OAuth",
        "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
        "fs.azure.account.oauth2.client.id": "your_client_id",
        "fs.azure.account.oauth2.client.secret": "your_client_secret",
        "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/your_tenant_id/oauth2/token"
    }

    try:
        dbutils.fs.mount(
            source=f"abfss://{CONTAINER_NAME}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/",
            mount_point=f"/mnt/{MOUNT_NAME}",
            extra_configs=configs
        )
        print(f"ADLS mounted successfully at `/mnt/{MOUNT_NAME}`")
    except Exception as e:
        print(f"ADLS Mount Failed: {e}")


# Initialize Spark Session (Databricks auto-initializes Spark)
spark = SparkSession.builder.appName("WindTurbineProcessing").config("spark.sql.extensions",
                                                                     "io.delta.sql.DeltaSparkSessionExtension").config(
    "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog").getOrCreate()


def read_bronze_layer() -> DataFrame:
    """Reads raw data from the Bronze Layer using Databricks AutoLoader."""
    try:
        return spark.readStream.format("cloudFiles") \
            .option("cloudFiles.format", "csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(BRONZE_PATH)
    except AnalysisException as e:
        print(f"Error reading data from Bronze Layer: {e}")
        return None


def impute_missing_values(df: DataFrame) -> DataFrame:
    """Handles missing values by filling numeric columns with median and categorical with 'Unknown'."""
    numeric_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]

    # Compute median values for all numeric columns in one pass
    median_values = {col: df.approxQuantile(col, [0.5], 0.05)[0] for col in numeric_cols}

    # Apply median imputation
    df = df.fillna(median_values)
    df = df.fillna({'turbine_id': 'Unknown'})  # Handle missing categorical values

    return df


def detect_outliers(df: DataFrame) -> DataFrame:
    """Detects outliers using the Interquartile Range (IQR) method."""
    cols_to_check = ['wind_speed', 'power_output', 'wind_direction']

    # Compute Q1, Q3, and IQR in a single pass
    quantiles = df.select(
        *[F.percentile_approx(c, [0.25, 0.75], 10000).alias(f"{c}_quantiles") for c in cols_to_check]).first()

    bounds = {col: {'lower': quantiles[f"{col}_quantiles"][0] - 1.5 * (
                quantiles[f"{col}_quantiles"][1] - quantiles[f"{col}_quantiles"][0]),
                    'upper': quantiles[f"{col}_quantiles"][1] + 1.5 * (
                                quantiles[f"{col}_quantiles"][1] - quantiles[f"{col}_quantiles"][0])}
              for col in cols_to_check}

    # Apply outlier detection in a single batch operation
    for col in cols_to_check:
        df = df.withColumn(f"{col}_is_outlier",
                           (F.col(col) < bounds[col]['lower']) | (F.col(col) > bounds[col]['upper']))

    df = df.withColumn("is_any_outlier",
                       F.array_contains(F.array(*[F.col(f"{col}_is_outlier") for col in cols_to_check]), True))

    return df


def detect_anomalies(df: DataFrame) -> DataFrame:
    """Detects anomalies in power output using standard deviation thresholding."""
    stats_df = df.groupBy("turbine_id").agg(
        F.mean("power_output").alias("mean_output"),
        F.stddev("power_output").alias("std_dev")
    )

    # Join statistics with the main DataFrame
    df = df.join(stats_df, on="turbine_id", how="left")

    # Detect anomalies in a batch operation
    df = df.withColumn("is_anomaly",
                       (F.col("power_output") < (F.col("mean_output") - 2 * F.col("std_dev"))) |
                       (F.col("power_output") > (F.col("mean_output") + 2 * F.col("std_dev"))))

    return df

def write_to_silver_layer(df: DataFrame):
    """Writes the cleaned DataFrame to the Silver Layer in Delta format."""
    df.writeStream.format("delta") \
        .option("checkpointLocation", CHECKPOINT_SILVER) \
        .outputMode("append") \
        .start(SILVER_PATH)
    print(f"Cleaned data successfully written to Silver Layer (Delta): {SILVER_PATH}")


def summarize_statistics(df: DataFrame) -> DataFrame:
    """Generates summary statistics and writes them to the Gold Layer in Delta format."""
    df_with_date = df.withColumn("date", F.to_date("timestamp"))

    summary_stats = df_with_date.groupBy("date", "turbine_id").agg(
        F.min("power_output").alias("min_power_output"),
        F.max("power_output").alias("max_power_output"),
        F.avg("power_output").alias("avg_power_output")
    ).orderBy("date")

    summary_stats.write.format("delta").mode("overwrite").save(GOLD_PATH)
    print(f"Summary statistics successfully written to Gold Layer (Delta): {GOLD_PATH}")

    return summary_stats


# **RUN THE PIPELINE**
try:
    print("Starting Wind Turbine Data Pipeline...\n")

    # Step 1: Read Data
    df = read_bronze_layer()
    if df is None:
        raise Exception("Data read failed!")

    # Step 2: Handle Missing Values
    df = impute_missing_values(df)

    # Step 3: Detect Outliers
    df = detect_outliers(df)

    # Step 4: Detect Anomalies
    df = detect_anomalies(df)

    # Step 5: Write to Silver Layer (Delta Format)
    write_to_silver_layer(df)

    # Step 6: Compute & Write Summary Statistics to Gold Layer (Delta Format)
    summary_stats = summarize_statistics(df)

    print("Pipeline execution completed successfully!")
except Exception as e:
    print(f"Pipeline execution failed: {e}")
