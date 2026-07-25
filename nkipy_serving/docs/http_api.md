# HTTP API

`nkipy-serving` exposes three HTTP API groups:

- OpenAI-compatible generation routes such as `/v1/completions` and `/v1/chat/completions`
- Native SGLang-style generation route `/generate`
- Control and utility routes for health, tokenization, cache flush, and weight reload

The current runtime is generation-only. Unsupported task routes such as embeddings, reranking, and cross-encoder scoring return explicit `501` responses.

## Endpoint Summary

| Group | Method | Path | Notes |
|------|--------|------|-------|
| Health | `GET` | `/health` | Liveness only |
| Health | `GET` | `/ready` | Readiness after runtime warmup |
| Health | `GET` | `/health_generate` | Tiny generate probe |
| Info | `GET` | `/get_server_info` | Full runtime info |
| Info | `GET` | `/server_info` | Alias of `/get_server_info` |
| Info | `GET` | `/version` | Server version |
| Info | `GET` | `/get_model_info` | Current model metadata |
| Info | `GET` | `/tokenizer_info` | Tokenizer class, vocab size, max context |
| Tokenizer | `POST` | `/v1/tokenize` | OpenAI-style utility route |
| Tokenizer | `POST` | `/v1/detokenize` | OpenAI-style utility route |
| Native | `POST` | `/generate` | Native generate API |
| Control | `POST` | `/abort_request` | Abort by request id |
| Control | `POST` | `/pause_generation` | Pause scheduler generation loop |
| Control | `POST` | `/continue_generation` | Resume paused generation |
| Control | `POST` | `/flush_cache` | Flush KV/prefix/request state |
| Control | `POST` | `/reload_weights_from_disk` | In-place same-shape reload |
| OpenAI | `POST` | `/v1/completions` | OpenAI completions |
| OpenAI | `POST` | `/v1/chat/completions` | OpenAI chat completions |
| OpenAI | `GET` | `/v1/models` | Model list |
| OpenAI | `GET` | `/v1/models/{model}` | Single model lookup |

## OpenAI-Compatible Generation

Supported OpenAI routes:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/models/{model}`

Example:

```bash
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":4,"temperature":0.0}' | jq .
```

```bash
curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":true}'
```

## Native Generate API

`POST /generate` is the native rollout-oriented API.

Supported inputs:

- `prompt`, `text`, or `prompts`
- pretokenized `input_ids`
- top-level sampling fields such as `max_new_tokens`, `temperature`, `top_k`, `top_p`, `min_p`
- nested `sampling_params` for the same sampling fields
- `n` for server-side duplication per logical input
- `return_logprob`, `logprob_start_len`, `top_logprobs_num`

Rules:

- Specify exactly one of text input or `input_ids`
- `text` may be a single string or `list[str]`
- `input_ids` may be `list[int]` or `list[list[int]]`
- streaming is only supported for a single generate request

Example: single-request form

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"The capital of France is","max_new_tokens":2}' | jq .
```

Example: batched token-id generation with sampled-token logprobs

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'content-type: application/json' \
  -d '{
        "input_ids":[[1,2,3],[4,5,6]],
        "max_new_tokens":1,
        "n":2,
        "return_logprob":true,
        "top_logprobs_num":0
      }' | jq .
```

Response shape:

```json
{
  "results": [
    {
      "request_id": null,
      "index": 0,
      "sample_index": 0,
      "text": "...",
      "prompt_ids": [1, 2, 3],
      "completion_ids": [42],
      "output_ids": [1, 2, 3, 42],
      "meta_info": {
        "finish_reason": "length",
        "output_token_logprobs": [[-0.3, 42, null]]
      }
    }
  ],
  "batch_size": 2,
  "n": 2
}
```

Logprob notes:

- `return_logprob=true` enables native logprob payloads
- `top_logprobs_num=0` returns sampled-token logprobs only
- prompt-side input logprobs require raw-logits backends today

## Tokenizer Utility Routes

Example tokenize:

```bash
curl -s http://127.0.0.1:30000/v1/tokenize \
  -H 'content-type: application/json' \
  -d '{"prompt":["Hello world","Hello"]}' | jq .
```

Example detokenize:

```bash
curl -s http://127.0.0.1:30000/v1/detokenize \
  -H 'content-type: application/json' \
  -d '{"tokens":[[9707,1879],[9707]],"skip_special_tokens":false}' | jq .
```

Example tokenizer info:

```bash
curl -s http://127.0.0.1:30000/tokenizer_info | jq .
```

## Control Routes

### Flush Cache

```bash
curl -s http://127.0.0.1:30000/flush_cache \
  -H 'content-type: application/json' \
  -d '{"abort_all_requests":true}' | jq .
```

### In-Place Weight Reload

Current reload is:

- in-place
- same-architecture and same-shape only
- KV/prefix/request state is flushed after reload
- compiled kernels are reused

```bash
curl -s http://127.0.0.1:30000/reload_weights_from_disk \
  -H 'content-type: application/json' \
  -d '{"model_path":"/path/to/local/snapshot","abort_all_requests":true}' | jq .
```

## Unsupported Routes

The following currently return `501` on the generation-only runtime:

- `/v1/embeddings`
- `/pooling`
- `/classify`
- `/v1/classify`
- `/rerank`
- `/v1/rerank`
- `/v2/rerank`
- `/v1/score`

`/v1/score` is reserved for cross-encoder scoring semantics (not supported by generation models).
