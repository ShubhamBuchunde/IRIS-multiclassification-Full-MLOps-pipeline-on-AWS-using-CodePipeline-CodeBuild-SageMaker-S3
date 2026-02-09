from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as cp_actions,
    aws_codebuild as codebuild,
)
from constructs import Construct


class MLOpsInfrastructureStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # -----------------------------------------------------------
        # 1. EXISTING S3 BUCKET
        # -----------------------------------------------------------
        data_bucket_name = "sagemaker-us-east-1-866824485776"

        data_bucket = s3.Bucket.from_bucket_name(
            self, "DataBucket", data_bucket_name
        )

        # -----------------------------------------------------------
        # 2. CODEBUILD PROJECTS
        # -----------------------------------------------------------

        # Build project
        build_project = codebuild.PipelineProject(
            self,
            "BuildProject",
            build_spec=codebuild.BuildSpec.from_source_filename(
                "buildspecs/build.yml"
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_6_0,
                privileged=False,
            ),
        )

        # Deploy project (SageMaker training + deployment)
        deploy_project = codebuild.PipelineProject(
            self,
            "DeployProject",
            build_spec=codebuild.BuildSpec.from_source_filename(
                "buildspecs/deploy.yml"
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_6_0,
            ),
        )

        # -----------------------------------------------------------
        # 3. IAM PERMISSIONS FOR SAGEMAKER
        # -----------------------------------------------------------

        deploy_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:CreateEndpoint",
                ],
                resources=["*"],
            )
        )

        # iam:PassRole
        sagemaker_execution_role_arn = (
            "arn:aws:iam::866824485776:role/service-role/"
            "AmazonSageMaker-ExecutionRole-20240913T125305"
        )

        deploy_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[sagemaker_execution_role_arn],
            )
        )

        # CloudWatch Logs permissions for SageMaker logs
        deploy_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:DescribeLogStreams",
                    "logs:GetLogEvents",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    "arn:aws:logs:us-east-1:866824485776:log-group:/aws/sagemaker/*"
                ],
            )
        )

        # Bucket read access
        data_bucket.grant_read(build_project)
        data_bucket.grant_read(deploy_project)

        # -----------------------------------------------------------
        # 4. CODEPIPELINE
        # -----------------------------------------------------------

        pipeline = codepipeline.Pipeline(
            self,
            "MLOpsPipeline",
            pipeline_name="MLOpsPipeline",
        )

        source_output = codepipeline.Artifact()
        build_output = codepipeline.Artifact()

        # GitHub CodeStar connection
        connection_arn = (
            "arn:aws:codeconnections:us-east-1:866824485776:connection/"
            "39acefd9-2544-4d08-bdb8-e220e3c0413f"
        )

        github_owner = "manifoldailearning"
        github_repo = "mlops-demo-v3"
        github_branch = "main"

        # Allow pipeline to use connection
        pipeline.add_to_role_policy(
            iam.PolicyStatement(
                actions=["codestar-connections:UseConnection"],
                resources=[connection_arn],
            )
        )

        # Source stage
        pipeline.add_stage(
            stage_name="Source",
            actions=[
                cp_actions.CodeStarConnectionsSourceAction(
                    action_name="SourceCode",
                    connection_arn=connection_arn,
                    owner=github_owner,
                    repo=github_repo,
                    branch=github_branch,
                    output=source_output,
                )
            ]
        )

        # Build stage
        pipeline.add_stage(
            stage_name="Build",
            actions=[
                cp_actions.CodeBuildAction(
                    action_name="BuildAndTest",
                    project=build_project,
                    input=source_output,
                    outputs=[build_output],
                )
            ],
        )

        # Deploy (train + deploy)
        pipeline.add_stage(
            stage_name="Deploy",
            actions=[
                cp_actions.CodeBuildAction(
                    action_name="TrainAndDeployModel",
                    project=deploy_project,
                    input=build_output,
                )
            ],
        )