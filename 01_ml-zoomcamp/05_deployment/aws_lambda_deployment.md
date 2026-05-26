# Deploy FastAPI ML Service to AWS Lambda

Uses **AWS Lambda Web Adapter** to run the existing uvicorn/FastAPI app unchanged inside Lambda via a container image.

---

## Prerequisites

- Docker installed
- IAM user created with the following policies attached:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AWSLambdaFullAccess`
  - `IAMFullAccess`

---

## Step 1 — Install and configure AWS CLI

```bash
brew install awscli
aws configure
```

Enter when prompted:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g. `us-east-1`)
- Default output format: `json`

---

## Step 2 — Update the Dockerfile

Two changes are required:

**Add Lambda Web Adapter** — a binary that translates Lambda events into HTTP and forwards them to uvicorn:

```dockerfile
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
```

**Call uvicorn directly from the venv** — Lambda's filesystem is read-only except `/tmp`. `uv run` tries to create a cache at `~/.cache/uv` and crashes. Bypass it by calling uvicorn directly:

```dockerfile
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Final Dockerfile:

```dockerfile
FROM python:3.13-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-dev

COPY . /app/

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 3 — Create ECR repository

```bash
aws ecr create-repository --repository-name churn-service
```

Note the `repositoryUri` from the output:
`<account-id>.dkr.ecr.<region>.amazonaws.com/churn-service`

---

## Step 4 — Build and push the image

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build for Lambda's architecture
# --provenance=false is required — newer Docker BuildKit adds OCI attestations
# that Lambda does not support
docker build --platform linux/amd64 --provenance=false -t churn-service .

# Tag and push
docker tag churn-service:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest
```

---

## Step 5 — Create IAM execution role

```bash
aws iam create-role \
  --role-name churn-service-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name churn-service-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

---

## Step 6 — Create the Lambda function

Wait ~10 seconds after creating the role before running this.

```bash
aws lambda create-function \
  --function-name churn-service \
  --package-type Image \
  --code ImageUri=<account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest \
  --role arn:aws:iam::<account-id>:role/churn-service-lambda-role \
  --memory-size 512 \
  --timeout 30 \
  --environment "Variables={AWS_LWA_PORT=8000}"
```

- `--memory-size 512` — scikit-learn + pandas need more than the 128MB default
- `--timeout 30` — cold start loads model.bin from disk
- `AWS_LWA_PORT=8000` — tells the Lambda Web Adapter which port uvicorn is on

---

## Step 7 — Create a public Function URL

```bash
aws lambda create-function-url-config \
  --function-name churn-service \
  --auth-type NONE
```

Note the `FunctionUrl` from the output.

---

## Step 8 — Grant public access

Since October 2025, Lambda requires **two** separate permission statements for public function URLs:

```bash
# Allow invocation via function URL
aws lambda add-permission \
  --function-name churn-service \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

# Allow function invocation itself
aws lambda add-permission \
  --function-name churn-service \
  --statement-id FunctionURLInvokeAllowPublicAccess \
  --action lambda:InvokeFunction \
  --principal "*" \
  --invoked-via-function-url
```

---

## Step 9 — Test

```bash
curl -X POST https://<function-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerid": "test-1",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 12,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "fiber optic",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic check",
    "monthlycharges": 70.5,
    "totalcharges": 846.0
  }'
```

Expected response: `{"churn_probability": 0.XXXX}`

---

## Updating after code or model changes

```bash
docker build --platform linux/amd64 --provenance=false -t churn-service . && \
docker tag churn-service:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest && \
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest

aws lambda update-function-code \
  --function-name churn-service \
  --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-service:latest
```

---

## Teardown

```bash
aws lambda delete-function --function-name churn-service
aws ecr delete-repository --repository-name churn-service --force
aws iam detach-role-policy \
  --role-name churn-service-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name churn-service-lambda-role
```
