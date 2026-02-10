import os
import json
import io
import csv
import boto3
import datetime
from decimal import Decimal
from typing import List

sagemaker_rt = boto3.client("sagemaker-runtime")
s3 = boto3.client("s3")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "mlops-iris-endpoint")
DATA_BUCKET = os.environ.get("DATA_BUCKET")  # optional if you fetch from S3
REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

def parse_date(s: str) -> datetime.date:
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()

def load_data_for_range_from_s3(start_date: str, end_date: str) -> List[List[float]]:
    """
    Example stub that: 
    - Lists S3 objects in a path partitioned by date
    - Reads CSV rows
    - Extracts features in the correct order

    Replace with your actual source (Athena/RDS/Redshift/Glue etc.)
    """
    # Example layout: s3://bucket/app/data/dt=YYYY-MM-DD/file.csv
    # Adjust prefix to match your actual paths
    bucket = DATA_BUCKET
    features = []

    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    cur = start
    while cur <= end:
        prefix = f"app/data/dt={cur.strftime('%Y-%m-%d')}/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            if not obj["Key"].endswith(".csv"):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(body))
            for row in reader:
                # Map your columns to model feature order here:
                # Example for iris-style features:
                sl = float(row["sepal_length"])
                sw = float(row["sepal_width"])
                pl = float(row["petal_length"])
                pw = float(row["petal_width"])
                features.append([sl, sw, pl, pw])
        cur += datetime.timedelta(days=1)

    return features

def to_csv_payload(rows: List[List[float]]) -> str:
    # newline-separated CSV rows, no header
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerows(rows)
    return out.getvalue()

def chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def invoke_endpoint_batch(rows: List[List[float]]) -> List[float]:
    """
    Invokes the endpoint in chunks to avoid payload size limits.
    """
    predictions = []
    for batch in chunk(rows, 500):  # tune chunk size; keep body < ~5 MB
        payload = to_csv_payload(batch)
        response = sagemaker_rt.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload.encode("utf-8")
        )
        result = response["Body"].read().decode("utf-8").strip()
        # Each line is the prediction for a row
        if result:
            for line in result.splitlines():
                try:
                    predictions.append(float(line))
                except ValueError:
                    predictions.append(line)  # if your model returns strings
    return predictions

def lambda_handler(event, context):
    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        start_date = body.get("start_date")
        end_date = body.get("end_date")

        if not start_date or not end_date:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "start_date and end_date are required (YYYY-MM-DD)."})
            }

        # 1) Load data for the range (replace with your source). If you already have features ready, skip this.
        rows = load_data_for_range_from_s3(start_date, end_date)
        if not rows:
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"count": 0, "predictions": []})
            }

        # 2) Invoke SageMaker
        preds = invoke_endpoint_batch(rows)

        # 3) Return results
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "count": len(preds),
                "predictions": preds,
                "start_date": start_date,
                "end_date": end_date
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }