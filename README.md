# CFPB Consumer Complaint Analytics Pipeline

A live, end-to-end AWS data engineering pipeline built on real Consumer Financial Protection Bureau (CFPB) complaint data — from raw ingestion through NLP/GenAI analysis to a publicly hosted interactive dashboard.

**Live dashboard:** https://dkd0nbsq74aem.cloudfront.net

---

## What this is

Consumers file complaints about banks, credit bureaus, and lenders directly with the CFPB, a U.S. federal agency. The CFPB publishes this data publicly. This project ingests that data, cleans and warehouses it, layers on machine learning and generative AI analysis, and exposes it through a live public API and dashboard.

It was built to demonstrate hands-on, production-style experience across the AWS data and AI stack — including the debugging that comes with real systems, not just the "happy path."

## Architecture

```
S3 (raw CSV)
   |
Glue (crawl + ETL to Parquet)
   |
Redshift Serverless (data warehouse)
   |         \
Lambda        Lambda + Amazon Comprehend (sentiment)
(REST API,        |
 parallel     Amazon Bedrock (Nova Micro)
 queries)     (on-demand company summaries)
   |
API Gateway (public REST endpoint, CORS-enabled)
   |
Dashboard (React, single HTML file)
   |
S3 static hosting + CloudFront (public HTTPS)
```

**Services used:** S3, Glue, Redshift Serverless, Lambda, API Gateway, SageMaker (notebook orchestration), Amazon Comprehend, Amazon Bedrock, CloudFront.

## What the dashboard shows

1. **Most-Named Companies** — top companies by complaint volume
2. **Disposition Timeliness** — how often companies respond on time
3. **Leading Issues Filed** — most common complaint categories
4. **Complaints Over Time** — filing trend by month
5. **Narrative Sentiment** — NLP sentiment analysis (Amazon Comprehend) on a sample of complaint narratives
6. **Ask about a company** — a live AI agent: type a company name, and Amazon Bedrock (Nova Micro) generates a real-time plain-English summary of that company's complaint patterns, grounded in the actual narrative text pulled from Redshift

## Repo structure

```
lambda/
  analytics_api.py     # Lambda handler: dashboard data API + Bedrock summarization
dashboard/
  index.html            # Self-contained React dashboard (CDN React, plain-SVG charts)
notebook/
  (SageMaker notebook work: Comprehend sentiment batch processing)
```

## Engineering notes / things that broke and got fixed

Real systems don't work on the first try. A few of the issues hit and resolved along the way:

- **Lambda timeout on cold Redshift queries** — running 5 queries sequentially could exceed both the Lambda timeout and API Gateway's hard 29s limit. Fixed by parallelizing queries with `ThreadPoolExecutor`.
- **IAM permission gaps** — Lambda's execution role needed explicit `GRANT SELECT` on Redshift tables (IAM-to-database role mapping), plus explicit IAM policies for `redshift-serverless:GetCredentials`, Comprehend, and Bedrock.
- **CORS misconfiguration** — API Gateway needs CORS explicitly configured (not just returned from the Lambda) or browsers block the preflight `OPTIONS` request entirely.
- **SageMaker notebook disk space** — default EBS volume too small for heavy ML libraries; pivoted to Amazon Comprehend (managed NLP service) instead of self-hosting a transformer model, which was faster, cheaper, and arguably a better architectural choice.
- **Redshift Serverless cost guardrails** — configured daily RPU-hour usage limits to prevent runaway cost during iterative testing.

## Cost

Built to stay near $0: S3, Lambda, API Gateway, and Comprehend all fall within AWS free-tier limits at this scale. Redshift Serverless usage is capped with a daily RPU-hour limit. Bedrock (Nova Micro) is pay-per-token with no free tier, but costs fractions of a cent per summary call.

## Data source

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — public, anonymized complaint records.

## Document Intelligence (Textract)

The pipeline also supports unstructured document inputs. Users can upload a scanned complaint letter, screenshot, or PDF, and the system automatically:

1. Extracts text using **Amazon Textract**
2. Runs sentiment analysis on the extracted text via **Amazon Comprehend**
3. Generates a concise AI summary of the complaint using **Amazon Bedrock** (Nova Micro)
4. Stores the combined result as structured JSON in S3

This extends the pipeline beyond structured CSV/API data to handle real-world document inputs — the kind financial institutions actually receive alongside typed complaints (scanned letters, screenshots of statements, etc.).

**Architecture:**

```
Document upload → S3 (incoming/) → Lambda trigger → Textract extraction
→ Comprehend sentiment analysis → Bedrock AI summary → S3 (results/) as JSON
```

Code: [`textract-lambda/lambda_function.py`](./textract-lambda/lambda_function.py)


## Future Work

**Infrastructure as Code** — Migrate manually provisioned resources (S3, Glue, Redshift Serverless, Lambda, API Gateway) to Terraform or AWS CDK for repeatable, version-controlled deployments.

**Fuzzy Matching** — Add fuzzy/entity matching to normalize company names in complaint data, improving aggregation accuracy across misspellings and naming variants.

**API Authentication** — Secure the API Gateway endpoint with API keys or Cognito-based auth instead of open public access.

**CI/CD** — Automate Glue job deployment, Lambda updates, and dashboard publishing via GitHub Actions on push to `main`.

**Expanded Sentiment Analysis** — Move beyond Comprehends baseline sentiment scoring toward a fine-tuned transformer model for more nuanced complaint-tone classification (e.g., urgency, frustration level).
