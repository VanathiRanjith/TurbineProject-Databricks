# Overview
This project implements a data processing pipeline for a renewable energy company that operates a farm of wind turbines. The pipeline ingests raw data from the turbines, cleans it by handling missing values and outliers, calculates summary statistics for power output, identifies anomalies, and stores the processed data in an Azure-based storage system using Spark. The code is designed to be scalable and testable, making it suitable for a production environment.

# Functional Requirements
The pipeline performs the following key operations:
1. Data Cleaning:
Imputes missing values in the dataset.
Detects outliers based on the interquartile range (IQR) method.

2. Summary Statistics:
Calculates the minimum, maximum, and average power output for each turbine over a specified time period.

3. Anomaly Detection:
Identifies turbines with power output deviations beyond ±2 standard deviations from the mean.

4. Data Storage:
Writes the cleaned data to the Silver layer in Parquet format.
Stores the summary statistics in the Gold layer.

# Code Structure
1. import required libraries
2. define Delta Lake storage paths in Databricks File System (DBFS) and initialize Spark Session (Databricks auto-initializes Spark)
3. read_bronze: Reads raw data from the Bronze layer.
4. imputed_missing_value: Handles missing values in the DataFrame.
5. detect_outliers: Detects outliers in the dataset.
6. detect_anomalies: Identifies anomalies based on power output deviations.
7. write_to_silver_layer: Writes the cleaned data to the Silver layer.
8. summary_statistics: Calculates and writes summary statistics to the Gold layer.
9. process_turbine: Orchestrates the entire data processing pipeline.

# Testing
1. Data cleaning operations (missing value imputation and outlier detection).
2. Summary statistics calculations.
3. Anomaly detection logic.
