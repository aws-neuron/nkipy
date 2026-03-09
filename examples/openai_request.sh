curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

curl -X POST http://localhost:8000/sleep
curl -X POST http://localhost:8000/wake_up


curl -X POST http://localhost:8100/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

curl -X POST http://localhost:8100/sleep
curl -X POST http://localhost:8100/wake_up