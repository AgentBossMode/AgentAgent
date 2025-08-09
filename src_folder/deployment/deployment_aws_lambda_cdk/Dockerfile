FROM public.ecr.aws/docker/library/python:3.12.0-slim-bullseye
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /deployment_aws_lambda_cdk
ADD . .
RUN pip install -r requirements2.txt

CMD ["python", "main.py"]