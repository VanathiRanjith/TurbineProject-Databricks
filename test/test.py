# Databricks-compatible test script for WindTurbinePipeline

# Import required libraries
import pytest
from pyspark.sql import Row, SparkSession

# Import the WindTurbinePipeline notebook
%run "/Repos/MyProject/turbine_pipeline"

@pytest.fixture(scope="module")
def spark():
    """Initialize a Databricks-compatible Spark session for testing."""
    spark_session = SparkSession.builder.appName("WindTurbinePipelineTest").getOrCreate()
    yield spark_session
    spark_session.stop()


@pytest.fixture(scope="module")
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        Row(turbine_id=1, wind_speed=10.0, power_output=100.0, wind_direction=180, timestamp="2023-09-28 00:00:00"),
        Row(turbine_id=1, wind_speed=None, power_output=120.0, wind_direction=185, timestamp="2023-09-28 01:00:00"),
        Row(turbine_id=2, wind_speed=15.0, power_output=130.0, wind_direction=190, timestamp="2023-09-28 02:00:00"),
        Row(turbine_id=2, wind_speed=50.0, power_output=500.0, wind_direction=200, timestamp="2023-09-28 03:00:00"),  # Outlier
    ]
    return spark.createDataFrame(data)


@pytest.fixture(scope="module")
def pipeline():
    """Initialize the WindTurbinePipeline with Databricks paths."""
    return WindTurbinePipeline(
        bronze_path="dbfs:/mnt/bronze/wind_turbine_data",
        silver_path="dbfs:/mnt/silver/wind_turbine_data",
        gold_path="dbfs:/mnt/gold/wind_turbine_summary"
    )


def test_spark_session_initialization(pipeline):
    """Test if Spark session is initialized properly."""
    assert pipeline.spark is not None, "Spark session should be initialized properly"


def test_read_bronze_layer(pipeline, spark):
    """Test reading data from the Bronze layer."""
    df = pipeline.read_bronze_layer()
    assert df is not None, "Data should be read from the Bronze layer"
    assert len(df.columns) > 0, "DataFrame should have some columns"


def test_impute_missing_values(pipeline, sample_df):
    """Test missing value imputation."""
    df_filled = pipeline.impute_missing_values(sample_df)
    assert df_filled.filter(df_filled.wind_speed.isNull()).count() == 0, "Missing wind_speed values should be filled"


def test_detect_outliers(pipeline, sample_df):
    """Test outlier detection logic."""
    df_outliers = pipeline.detect_outliers(sample_df)
    assert "wind_speed_is_outlier" in df_outliers.columns, "Outlier column should be added for wind_speed"
    assert "is_any_outlier" in df_outliers.columns, "is_any_outlier column should be added"


def test_detect_anomalies(pipeline, sample_df):
    """Test anomaly detection based on power output."""
    df_with_outliers = pipeline.detect_outliers(sample_df)
    anomalies_df = pipeline.detect_anomalies(df_with_outliers)
    assert anomalies_df is not None, "Anomalies DataFrame should not be None"
    assert len(anomalies_df.columns) == len(sample_df.columns), "Anomalies DataFrame should have the same columns as input"


def test_write_to_silver(pipeline, sample_df):
    """Test writing cleaned DataFrame to the Silver layer using Delta format."""
    try:
        pipeline.write_to_silver_layer(sample_df)
        df = pipeline.spark.read.format("delta").load("dbfs:/mnt/silver/wind_turbine_data")
        assert df.count() > 0, "Silver Layer should contain data"
    except Exception as e:
        pytest.fail(f"Writing to Silver layer failed: {e}")


def test_summarize_statistics(pipeline, sample_df):
    """Test summarizing power output and writing to Gold layer using Delta format."""
    try:
        pipeline.summarize_statistics(sample_df)
        df = pipeline.spark.read.format("delta").load("dbfs:/mnt/gold/wind_turbine_summary")
        assert df.count() > 0, "Gold Layer should contain summary data"
    except Exception as e:
        pytest.fail(f"Summary statistics failed: {e}")


def test_pipeline_end_to_end(pipeline):
    """Test the complete pipeline execution in Databricks."""
    try:
        pipeline.process()
        df_silver = pipeline.spark.read.format("delta").load("dbfs:/mnt/silver/wind_turbine_data")
        df_gold = pipeline.spark.read.format("delta").load("dbfs:/mnt/gold/wind_turbine_summary")
        assert df_silver.count() > 0, "Silver Layer should have processed data"
        assert df_gold.count() > 0, "Gold Layer should have summary statistics"
    except Exception as e:
        pytest.fail(f"Pipeline processing failed: {e}")
