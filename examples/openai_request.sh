curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

curl -X POST http://localhost:8000/sleep
curl -X POST http://localhost:8000/wake_up

curl -X POST http://localhost:8001/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

curl -X POST http://localhost:8002/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

curl -X POST http://localhost:8001/sleep
curl -X POST http://localhost:8001/wake_up


# peer-to-peer weight transfer within instances
curl -X POST http://localhost:8001/wake_up \
  -H "Content-Type: application/json" \
  -d '{"peer_url": "http://localhost:8002"}'

# peer-to-peer weight transfer across instances
curl -X POST http://localhost:8001/wake_up \
  -H "Content-Type: application/json" \
  -d '{"peer_url": "http://172.31.44.131:8000"}'