#!/usr/bin/env python3
import os

import aws_cdk as cdk

from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_iam as iam,
    Duration,
    CfnOutput
)
from constructs import Construct

class DeploymentAwsLambdaCdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create the Lambda function from a Docker image
        fast_api_function = lambda_.DockerImageFunction(
            self, "FastAPIFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=".",  # Path to the directory containing your Dockerfile
                cmd=None,
                entrypoint=None,
                build_args=None,
            ),
            memory_size=512,
            timeout=Duration.seconds(300),
            environment={
                "AWS_LWA_INVOKE_MODE": "RESPONSE_STREAM"
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # Add permission to invoke Bedrock models with response streaming
        fast_api_function.add_to_role_policy(
            iam.PolicyStatement(
                sid="BedrockInvokePolicy",
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModelWithResponseStream"],
                resources=["*"]
            )
        )

        # Create function URL with response streaming
        function_url = fast_api_function.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            invoke_mode=lambda_.InvokeMode.RESPONSE_STREAM
        )
        # Define outputs
        CfnOutput(
            self, "FastAPIFunctionUrl",
            description="Function URL for FastAPI function",
            value=function_url.url
        )

        CfnOutput(
            self, "FastAPIFunctionArn",
            description="FastAPI Lambda Function ARN",
            value=fast_api_function.function_arn
        )


app = cdk.App()
DeploymentAwsLambdaCdkStack(app, "DeploymentAwsLambdaCdkStack",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.

    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */

    env=cdk.Environment(account='245824310656', region='us-east-1'),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
    )

app.synth()
