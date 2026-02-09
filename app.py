#!/usr/bin/env python3
import aws_cdk as cdk
from mlops_infrastructure_stack import MLOpsInfrastructureStack

app = cdk.App()

# Create the stack
MLOpsInfrastructureStack(
    app,
    "MLOpsInfrastructureStack",
    # Optional environment specification
    # env=cdk.Environment(account="123456789012", region="us-east-1")
)

app.synth()