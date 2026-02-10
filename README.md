AWS CDK Stack that automatically builds and deploys a Machine Learning model using AWS services. 

Whenever code is pushed to GitHub → run tests → run SageMaker training → deploy ML model automatically.
GitHub → CodePipeline → CodeBuild → CDK Deploy → SageMaker Model Deployment

Your CDK stack deploys a complete ML pipeline:
✔ Pull latest ML code from GitHub 
✔ Build & test it (CI) 
✔ Train a new SageMaker model 
✔ Deploy the trained model to SageMaker Endpoint (CD) 
✔ Automates everything end-to-end This is a true CI/CD + MLOps workflow.
 
**IRIS MULTICLASSIFICATION – FULL MLOps PIPELINE (End‑to‑End Explanation)**
This is a production-grade MLOps pipeline built on AWS, which automates:
Data storage
Model training
Model deployment
Endpoint updates
CI/CD automation via CodePipeline + CodeBuild
Infrastructure-as-Code using AWS CDK
Real-time inference

**What This Project Does?**

This project builds a fully automated MLOps pipeline on AWS for training and deploying a multi‑class classification model (Iris dataset) using SageMaker.
It uses CI/CD pipelines (CodePipeline, CodeBuild) and Infrastructure-as-Code (AWS CDK) to provision cloud resources, train machine learning models, and deploy them to a scalable endpoint exposed for prediction.

You can describe it in one sentence:

A fully automated, cloud-native MLOps pipeline that trains an XGBoost model in SageMaker and deploys it to a real-time inference endpoint using CI/CD and AWS CDK.
