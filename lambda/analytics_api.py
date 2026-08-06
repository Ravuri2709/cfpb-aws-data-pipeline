"""
analytics_api.py

Lambda function behind API Gateway that queries Redshift Serverless via the
Redshift Data API (no persistent DB connection needed - perfect for Lambda)
and returns JSON for the dashboard to chart.

UPDATED: runs all 5 queries in PARALLEL (ThreadPoolExecutor) instead of
sequentially. Sequential execution could take up to 5 x 30s = 150s worst
case, blowing past both the Lambda timeout and API Gateway's hard 29s
integration limit. Running in parallel brings total time down to roughly
the duration of the single slowest query.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

redshift_data = boto3.client("redshift-data", region_name="us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

WORKGROUP = os.environ.get("WORKGROUP_NAME", "default-workgroup")
DATABASE = os.environ.get("DATABASE_NAME", "dev")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

# Max seconds to wait for any single query to finish polling.
# Keep this comfortably under API Gateway's 29s hard limit.
MAX_POLL_SECONDS = 20
POLL_INTERVAL_SECONDS = 0.5

QUERIES = {
    "top_companies": """
        SELECT company, COUNT(*) AS complaint_count
        FROM complaints
        GROUP BY company
        ORDER BY complaint_count DESC
        LIMIT 8
    """,
    "top_issues": """
        SELECT issue, COUNT(*) AS issue_count
        FROM complaints
        GROUP BY issue
        ORDER BY issue_count DESC
        LIMIT 8
    """,
    "monthly_trend": """
        SELECT TO_CHAR(date_received, 'YYYY-MM') AS month, COUNT(*) AS complaints
        FROM complaints
        GROUP BY 1
        ORDER BY 1
    """,
    "response_timeliness": """
        SELECT timely_response, COUNT(*) AS count
        FROM complaints
        GROUP BY timely_response
    """,
    "summary_stats": """
        SELECT
            COUNT(*) AS total_complaints,
            COUNT(DISTINCT company) AS unique_companies,
            COUNT(DISTINCT state) AS states_represented
        FROM complaints
    """,
    "sentiment_breakdown": """
        SELECT sentiment, COUNT(*) AS count
        FROM complaint_sentiment
        GROUP BY sentiment
        ORDER BY count DESC
    """,
}


def run_query(sql: str) -> list:
    resp = redshift_data.execute_statement(
        WorkgroupName=WORKGROUP,
        Database=DATABASE,
        Sql=sql,
    )
    query_id = resp["Id"]

    status = {}
    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        status = redshift_data.describe_statement(Id=query_id)
        if status["Status"] in ("FINISHED", "FAILED", "ABORTED"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if status.get("Status") != "FINISHED":
        raise RuntimeError(
            f"Query failed or timed out: {status.get('Error', status.get('Status', 'unknown'))}"
        )

    result = redshift_data.get_statement_result(Id=query_id)
    columns = [c["name"] for c in result["ColumnMetadata"]]

    rows = []
    for record in result["Records"]:
        row = {}
        for col, field in zip(columns, record):
            row[col] = next(iter(field.values()))
        rows.append(row)
    return rows


def summarize_company(company: str) -> str:
    sql = f"""
        SELECT narrative
        FROM complaints
        WHERE company = '{company.replace("'", "''")}'
        LIMIT 15
    """
    rows = run_query(sql)
    narratives = [r["narrative"] for r in rows if r.get("narrative")]

    if not narratives:
        return f"No complaint narratives found for \"{company}\"."

    prompt = (
        f"Summarize the most common issues in these customer complaints filed "
        f"against \"{company}\" in 3-4 sentences, in plain business language:\n\n"
        + "\n\n".join(narratives)
    )

    resp = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 400},
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"]


def lambda_handler(event, context):
    params = (event or {}).get("queryStringParameters") or {}
    company = params.get("company")

    if company:
        try:
            summary = summarize_company(company)
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"company": company, "summary": summary}),
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": str(e)}),
            }

    data = {}
    errors = {}

    # Run all named queries concurrently instead of one after another.
    with ThreadPoolExecutor(max_workers=len(QUERIES)) as executor:
        future_to_name = {
            executor.submit(run_query, sql): name for name, sql in QUERIES.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                data[name] = future.result()
            except Exception as e:
                errors[name] = str(e)

    if errors and not data:
        # Every query failed - return a real error instead of a blank 200.
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": errors}),
        }

    body = {"data": data}
    if errors:
        # Partial success - still return what we have, plus which ones failed.
        body["errors"] = errors

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }
