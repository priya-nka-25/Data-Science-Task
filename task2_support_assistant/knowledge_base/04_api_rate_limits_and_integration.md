# Developer API Rate Limits & Webhook Protocols

## Section 4.1: API Quotas and Token Bucket Rate Limits
All API endpoints enforce rate limits managed by a distributed token bucket algorithm. Standard accounts are capped at 60 requests per minute and 10,000 requests per day. Business accounts receive 300 requests per minute and 100,000 requests per day. Enterprise Dedicated accounts enjoy custom throughput up to 2,000 requests per minute with burst capability up to 5,000 requests. When limits are exceeded, the API responds with HTTP status `429 Too Many Requests` containing standard `Retry-After` and `X-RateLimit-Reset` response headers.

## Section 4.2: Webhook Delivery and Exponential Backoff Retries
Webhook events are signed with HMAC-SHA256 signatures in the `X-Hub-Signature-256` header. The receiver server must return a `2xx` HTTP status code within 5 seconds. If delivery fails or times out, our dispatcher implements an exponential backoff retry schedule: retry at 1 minute, 5 minutes, 30 minutes, 2 hours, and 12 hours. If all 5 attempts fail, the endpoint is flagged as degraded and an administrative webhook failure alert is triggered.

