import pytest
from pyspark.sql import Row, SparkSession
from src.turbinePipeline import WindTurbinePipeline


@pytest.fixture(scope="module")
def spark():
    """Fixture to initialize and provide a Spark session for the tests."""
    spark_session = SparkSession.builder.appName("WindTurbinePipelineTest").getOrCreate()
    yield spark_session
    spark_session.stop()


@pytest.fixture(scope="module")
def sample_df(spark):
    """Fixture to create and provide a sample DataFrame for testing."""
    data = [
        Row(turbine_id=1, wind_speed=10.0, power_output=100.0, wind_direction=180, timestamp="2023-09-28 00:00:00"),
        Row(turbine_id=1, wind_speed=None, power_output=120.0, wind_direction=185, timestamp="2023-09-28 01:00:00"),
        Row(turbine_id=2, wind_speed=15.0, power_output=130.0, wind_direction=190, timestamp="2023-09-28 02:00:00"),
        Row(turbine_id=2, wind_speed=50.0, power_output=500.0, wind_direction=200, timestamp="2023-09-28 03:00:00"),  # Outlier
    ]
    return spark.createDataFrame(data)


def test_spark_session_initialization():
    """Test if Spark session is initialized properly."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    assert pipeline.spark is not None, "Spark session should be initialized properly"


def test_read_bronze_layer(spark):
    """Test the reading from the Bronze layer."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    df = pipeline.read_bronze_layer()
    assert df is not None, "Data should be read from the Bronze layer"
    assert len(df.columns) > 0, "DataFrame should have some columns"


def test_impute_missing_values(sample_df):
    """Test missing value imputation."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    df_filled = pipeline.impute_missing_values(sample_df)
    assert df_filled.filter(df_filled.wind_speed.isNull()).count() == 0, "Missing wind_speed values should be filled"


def test_detect_outliers(sample_df):
    """Test outlier detection logic."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    df_outliers = pipeline.detect_outliers(sample_df)
    assert "wind_speed_is_outlier" in df_outliers.columns, "Outlier column should be added for wind_speed"
    assert "is_any_outlier" in df_outliers.columns, "is_any_outlier column should be added"


def test_detect_anomalies(sample_df):
    """Test anomaly detection based on power output."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    df_with_outliers = pipeline.detect_outliers(sample_df)
    anomalies_df = pipeline.detect_anomalies(df_with_outliers)
    assert anomalies_df is not None, "Anomalies DataFrame should not be None"
    assert len(anomalies_df.columns) == len(sample_df.columns), "Anomalies DataFrame should have the same columns as input"


def test_write_to_silver(sample_df):
    """Test writing cleaned DataFrame to the Silver layer."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    try:
        pipeline.write_to_silver(sample_df)
        assert True, "Writing to Silver layer should succeed"
    except Exception as e:
        pytest.fail(f"Writing to Silver layer failed: {e}")


def test_summarize_statistics(sample_df):
    """Test summarizing power output and writing to Gold layer."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    try:
        pipeline.summarize_statistics(sample_df)
        assert True, "Summarizing power output and saving to Gold layer should succeed"
    except Exception as e:
        pytest.fail(f"Summary statistics failed: {e}")


def test_pipeline_end_to_end(sample_df):
    """Test the complete pipeline processing."""
    pipeline = WindTurbinePipeline("acc_name", "container_name", "access_key")
    try:
        pipeline.process()
        assert True, "Pipeline should process data successfully"
    except Exception as e:
        pytest.fail(f"Pipeline processing failed: {e}")
