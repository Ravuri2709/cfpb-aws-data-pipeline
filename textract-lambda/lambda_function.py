import json
import boto3
import os

s3 = boto3.client('s3')
textract = boto3.client('textract')
comprehend = boto3.client('comprehend')
bedrock = boto3.client('bedrock-runtime')

RESULTS_PREFIX = "results/"

def lambda_handler(event, context):
    try:
        # Get the uploaded file's bucket and key from the S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        # Skip if this event is for something in results/ (avoid loops)
        if key.startswith(RESULTS_PREFIX):
            return {"statusCode": 200, "body": "Skipped results file"}

        filename = os.path.basename(key)

        # 1. Extract text using Textract
        textract_response = textract.detect_document_text(
            Document={'S3Object': {'Bucket': bucket, 'Name': key}}
        )
        extracted_lines = [
            block['Text'] for block in textract_response['Blocks']
            if block['BlockType'] == 'LINE'
        ]
        extracted_text = "\n".join(extracted_lines)

        if not extracted_text.strip():
            result = {
                "filename": filename,
                "error": "No text could be extracted from this document."
            }
            write_result(bucket, filename, result)
            return {"statusCode": 200, "body": json.dumps(result)}

        # 2. Sentiment analysis using Comprehend (max 5000 bytes per call)
        comprehend_text = extracted_text[:4900]
        sentiment_response = comprehend.detect_sentiment(
            Text=comprehend_text,
            LanguageCode='en'
        )
        sentiment = sentiment_response['Sentiment']
        sentiment_scores = sentiment_response['SentimentScore']

        # 3. Generate a summary using Bedrock (Nova Micro)
        prompt_text = (
            "You are summarizing a customer complaint document. "
            "In 2-3 sentences, summarize the key issue described below:\n\n"
            f"{extracted_text[:3000]}"
        )
        bedrock_response = bedrock.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": prompt_text}]}
                ]
            }),
            contentType="application/json",
            accept="application/json"
        )
        bedrock_body = json.loads(bedrock_response['body'].read())
        summary = bedrock_body['output']['message']['content'][0]['text']

        # 4. Assemble and store the result
        result = {
            "filename": filename,
            "extracted_text": extracted_text,
            "sentiment": sentiment,
            "sentiment_scores": sentiment_scores,
            "summary": summary
        }
        write_result(bucket, filename, result)

        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as e:
        error_result = {"error": str(e)}
        try:
            write_result(bucket, filename, error_result)
        except Exception:
            pass
        return {"statusCode": 500, "body": json.dumps(error_result)}


def write_result(bucket, filename, result):
    result_key = f"{RESULTS_PREFIX}{filename}.json"
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(result, indent=2),
        ContentType="application/json"
    )
