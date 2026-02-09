import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker import image_uris
import os
from botocore.exceptions import ClientError  # Use ClientError instead of SageMakerError
import datetime

# Print SageMaker SDK version for debugging
print(f"SageMaker SDK version: {sagemaker.__version__}")

# Initialize AWS session
region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
session = boto3.Session(region_name=region)
sagemaker_session = Session(boto_session=session)

# Define S3 paths and IAM role
data_bucket = "sagemaker-us-east-1-866824485776"
iris_data_uri = f"s3://{data_bucket}/iris.libsvm"  # Ensure iris.libsvm is correctly formatted and uploaded
output_path = f"s3://{data_bucket}/model-artifacts/"  # Model artifacts go here
role = "arn:aws:iam::866824485776:role/service-role/AmazonSageMaker-ExecutionRole-20240913T125305"

# Retrieve the XGBoost built-in container image URI
xgboost_image = image_uris.retrieve("xgboost", region=region, version="1.2-1")

# Create an Estimator using the built-in XGBoost algorithm
estimator = sagemaker.estimator.Estimator(
    image_uri=xgboost_image,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
    sagemaker_session=sagemaker_session
)

# Set hyperparameters for multi-class classification
estimator.set_hyperparameters(
    objective='multi:softmax',  # For multi-class classification
    num_class=3,                # Number of classes
    num_round=50                # Number of boosting rounds
)

# Define the training data channel
train_data = iris_data_uri

# Fit the model on the training data
print("Starting training job...")
estimator.fit({'train': train_data}, wait=True)
print("Training completed.")

# Create a SageMaker model object from the trained estimator
model = estimator.create_model()
endpoint_name = "mlops-iris-endpoint"

# Function to create a new endpoint config
def create_endpoint_config(model, endpoint_config_name, instance_type):
    response = sagemaker_session.sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model.name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
            },
        ],
    )
    return response

# Function to update the endpoint
def update_endpoint(endpoint_name, endpoint_config_name):
    response = sagemaker_session.sagemaker_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    return response

try:
    # Check if the endpoint already exists by listing endpoints via the client
    response = sagemaker_session.sagemaker_client.list_endpoints(MaxResults=100)
    existing_endpoints = [ep['EndpointName'] for ep in response['Endpoints']]
    print(f"Existing endpoints: {existing_endpoints}")

    if endpoint_name in existing_endpoints:
        print(f"Endpoint '{endpoint_name}' already exists. Updating it with the new model.")

        # Create a new endpoint config with a unique name using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

        print(f"Creating new endpoint config: {new_endpoint_config_name}")
        create_endpoint_config(model, new_endpoint_config_name, "ml.m5.large")

        print(f"Updating endpoint '{endpoint_name}' to use new config '{new_endpoint_config_name}'")
        update_endpoint(endpoint_name, new_endpoint_config_name)
        print(f"Updated endpoint: {endpoint_name}")
    else:
        # Create the endpoint if it doesn't exist
        print(f"Creating endpoint: {endpoint_name}")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',
            endpoint_name=endpoint_name
        )
        print(f"Created endpoint: {endpoint_name}")
except ClientError as e:
    print(f"A ClientError occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
