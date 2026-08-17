"""
analytics_api.py (== the Lambda's `lambda_function.py`, function name `Pavan__2506`)

Lambda function behind API Gateway (HTTP API) that:
  - queries Redshift Serverless via the Redshift Data API and returns JSON
    for the dashboard to chart (GET, no params)
  - generates a Bedrock-powered company summary (GET ?company=<name>)
  - accepts a document upload and drops it in S3 for the Textract Lambda
    to pick up (POST {"filename": "...", "file_base64": "..."})
  - polls S3 results/ for the processed Textract/Comprehend/Bedrock JSON
    (GET ?check=<filename>)
  - pages through top_companies / top_issues (GET ?more=<name>&offset=N)

Security/reliability notes (fixed after code review):
  - Every request (except CORS preflight) must send a correct `x-api-key`
    header - this is an HTTP API, which (unlike REST APIs) has no built-in
    API-key/usage-plan feature, so the check is enforced in code.
  - All Redshift Data API calls use `Parameters` (named bind params)
    instead of string-interpolated SQL.
  - Uploaded filenames are sanitized before being used as an S3 key
    (previously used the raw client-supplied filename directly, which
    allowed path traversal / overwriting arbitrary objects in the bucket).
  - Raw exception text is never returned to the caller. Full detail is
    logged to CloudWatch; the client gets a short, generic, categorized
    message plus a request id.
  - Required config (workgroup/database/upload bucket/api key) is
    validated at cold start and fails loudly instead of silently falling
    back to a possibly-wrong default.
"""

import base64
import hmac
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configuration -----------------------------------------------------
# No hardcoded region: Lambda always sets AWS_REGION in its execution
# environment, so boto3 picks it up automatically instead of pinning a
# region string in code (which silently breaks if ever deployed elsewhere).
redshift_data = boto3.client("redshift-data")
bedrock = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")


def _required_env(name: str) -> str:
    """Fail loudly at cold start if required config is missing, instead of
    silently using a possibly-wrong default."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in the Lambda's Configuration > Environment variables."
        )
    return value


WORKGROUP = _required_env("WORKGROUP_NAME")
DATABASE = _required_env("DATABASE_NAME")
UPLOAD_BUCKET = _required_env("UPLOAD_BUCKET")
# Shared secret the caller must send as the `x-api-key` header.
API_SECRET_KEY = _required_env("API_SECRET_KEY")
# Reasonable default - not the kind of "wrong workgroup" mistake that
# should hard-fail the whole function, so this one keeps a fallback.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

MAX_POLL_SECONDS = 20
POLL_INTERVAL_SECONDS = 0.5
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8MB
DEFAULT_PAGE_SIZE = 8
MAX_PAGE_SIZE = 50
MAX_OFFSET = 10_000

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(filename: str) -> str:
    """Strip any path components and unsafe characters before this is used
    to build an S3 key, so a filename like '../../results/other-user.json'
    can't be used to read/overwrite objects outside the intended prefix."""
    base = os.path.basename(filename or "")
    base = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "upload"
    return base[:200]


FIXED_QUERIES = {
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

# limit/offset here are always ints we've clamped ourselves (never raw user
# strings), so it's safe to format them directly into the SQL - only
# free-text values (like `company`) go through bind Parameters.
PAGINATED_QUERIES = {
    "top_companies": """
        SELECT company, COUNT(*) AS complaint_count
        FROM complaints
        GROUP BY company
        ORDER BY complaint_count DESC
        LIMIT {limit} OFFSET {offset}
    """,
    "top_issues": """
        SELECT issue, COUNT(*) AS issue_count
        FROM complaints
        GROUP BY issue
        ORDER BY issue_count DESC
        LIMIT {limit} OFFSET {offset}
    """,
}


def _clamp_int(raw, default, minimum, maximum):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


class QueryError(Exception):
    """A query failed or timed out. Safe to log, NOT safe to return to the
    client as-is (may contain internal details)."""


def run_query(sql: str, parameters: list = None) -> list:
    kwargs = {"WorkgroupName": WORKGROUP, "Database": DATABASE, "Sql": sql}
    if parameters:
        kwargs["Parameters"] = parameters

    resp = redshift_data.execute_statement(**kwargs)
    query_id = resp["Id"]

    status = {}
    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        status = redshift_data.describe_statement(Id=query_id)
        if status["Status"] in ("FINISHED", "FAILED", "ABORTED"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if status.get("Status") != "FINISHED":
        raise QueryError(
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


def summarize_company(company: str, limit: int) -> str:
    sql = """
        SELECT narrative
        FROM complaints
        WHERE company = :company
        LIMIT :narrative_limit
    """
    parameters = [
        {"name": "company", "value": company},
        {"name": "narrative_limit", "value": str(limit)},
    ]
    rows = run_query(sql, parameters=parameters)
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


def handle_upload(event, request_id):
    """Accepts {"filename": "...", "file_base64": "..."} and writes it to
    S3 incoming/, which triggers the Textract Lambda automatically."""
    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw)

        filename = _sanitize_filename(body.get("filename"))
        file_b64 = body.get("file_base64")

        if not body.get("filename") or not file_b64:
            return _error_response(400, "missing_fields", request_id)

        # strip a data URI prefix if present, e.g. "data:image/png;base64,...."
        if "," in file_b64 and file_b64.strip().startswith("data:"):
            file_b64 = file_b64.split(",", 1)[1]

        try:
            file_bytes = base64.b64decode(file_b64, validate=True)
        except Exception:
            return _error_response(400, "invalid_file_data", request_id)

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return _error_response(413, "file_too_large", request_id)

        s3.put_object(
            Bucket=UPLOAD_BUCKET,
            Key=f"incoming/{filename}",
            Body=file_bytes,
        )

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"status": "uploaded", "filename": filename}),
        }
    except Exception:
        logger.exception("Upload failed [request_id=%s]", request_id)
        return _error_response(502, "upload_failed", request_id)


def handle_check(filename, request_id):
    """Polls S3 results/ for the processed Textract/Comprehend/Bedrock JSON."""
    safe_filename = _sanitize_filename(filename)
    result_key = f"results/{safe_filename}.json"
    try:
        obj = s3.get_object(Bucket=UPLOAD_BUCKET, Key=result_key)
        result = json.loads(obj["Body"].read())
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"status": "ready", "result": result}),
        }
    except s3.exceptions.NoSuchKey:
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"status": "processing"}),
        }
    except Exception:
        logger.exception("Check failed [request_id=%s filename=%r]", request_id, safe_filename)
        return _error_response(502, "check_failed", request_id)


def _error_response(status_code: int, category: str, request_id: str):
    """Return a sanitized error to the client. The real exception detail
    was already logged server-side under `request_id` - never echo raw
    exception text back to the caller."""
    body = {"error": category, "request_id": request_id}
    return {"statusCode": status_code, "headers": CORS_HEADERS, "body": json.dumps(body)}


def _is_authorized(event) -> bool:
    headers = (event or {}).get("headers") or {}
    supplied = headers.get("x-api-key") or headers.get("X-Api-Key") or ""
    return hmac.compare_digest(supplied, API_SECRET_KEY)


def _get_http_method(event) -> str:
    """This API is an API Gateway HTTP API using payload format 2.0, where
    the method lives at event.requestContext.http.method - there is NO
    top-level event["httpMethod"] (that only exists in the older REST-API
    / payload-format-1.0 event shape). Reading the wrong field silently
    defaulted every request - including real OPTIONS preflight requests -
    to "GET", which broke CORS as soon as an auth check was added (the
    misread "GET" preflight had no x-api-key header and got rejected)."""
    event = event or {}
    if "httpMethod" in event:  # format 1.0 / REST API
        return event["httpMethod"]
    return ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET")


def lambda_handler(event, context):
    request_id = uuid.uuid4().hex[:12]
    method = _get_http_method(event)

    # CORS preflight - browsers don't send custom headers on OPTIONS, so
    # this must bypass the API-key check or the dashboard's fetch() calls
    # would fail before ever sending the real request.
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    if not _is_authorized(event):
        logger.warning("Rejected unauthorized request [request_id=%s method=%s]", request_id, method)
        return _error_response(401, "unauthorized", request_id)

    if method == "POST":
        return handle_upload(event, request_id)

    params = (event or {}).get("queryStringParameters") or {}

    check_filename = params.get("check")
    if check_filename:
        return handle_check(check_filename, request_id)

    company = params.get("company")
    if company:
        narrative_limit = _clamp_int(params.get("limit"), 15, 1, MAX_PAGE_SIZE)
        try:
            summary = summarize_company(company, narrative_limit)
            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps({"company": company, "summary": summary}),
            }
        except Exception:
            logger.exception("summarize_company failed [request_id=%s company=%r]", request_id, company)
            return _error_response(502, "summary_unavailable", request_id)

    more = params.get("more")
    if more:
        if more not in PAGINATED_QUERIES:
            return _error_response(400, "unknown_dataset", request_id)
        limit = _clamp_int(params.get("limit"), DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE)
        offset = _clamp_int(params.get("offset"), 0, 0, MAX_OFFSET)
        sql = PAGINATED_QUERIES[more].format(limit=limit, offset=offset)
        try:
            rows = run_query(sql)
        except Exception:
            logger.exception("paginated query failed [request_id=%s dataset=%s]", request_id, more)
            return _error_response(502, "query_failed", request_id)
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"rows": rows, "limit": limit, "offset": offset}, default=str),
        }

    # Full dashboard fetch (first page of everything)
    data = {}
    errors = {}

    all_queries = dict(FIXED_QUERIES)
    for name, template in PAGINATED_QUERIES.items():
        all_queries[name] = template.format(limit=DEFAULT_PAGE_SIZE, offset=0)

    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        future_to_name = {
            executor.submit(run_query, sql): name for name, sql in all_queries.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                data[name] = future.result()
            except Exception:
                logger.exception("query failed [request_id=%s dataset=%s]", request_id, name)
                errors[name] = "query_failed"

    if errors and not data:
        return _error_response(502, "all_queries_failed", request_id)

    body = {"data": data}
    if errors:
        body["errors"] = errors
        body["request_id"] = request_id

    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(body, default=str)}
