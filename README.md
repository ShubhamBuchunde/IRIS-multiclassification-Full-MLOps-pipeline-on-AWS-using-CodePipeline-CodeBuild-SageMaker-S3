AWS CDK Stack that automatically builds and deploys a Machine Learning model using AWS services. 

Whenever code is pushed to GitHub → run tests → run SageMaker training → deploy ML model automatically.
GitHub → CodePipeline → CodeBuild → CDK Deploy → SageMaker Model Deployment

Your CDK stack deploys a complete ML pipeline:
✔ Pull latest ML code from GitHub 
✔ Build & test it (CI) 
✔ Train a new SageMaker model 
✔ Deploy the trained model to SageMaker Endpoint (CD) 
✔ Automates everything end-to-end This is a true CI/CD + MLOps workflow.
 
IRIS-multiclassification-Full-MLOps-pipeline-on-AWS-using-CodePipeline-CodeBuild-SageMaker-S3
 
