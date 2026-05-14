# LLM Serving Scheduler Design

## 1. Goal

In [FAST_MODEL_SWITCH_DEMO.md](FAST_MODEL_SWITCH_DEMO.md), we document the system design for fast on-demand model serving scaling up. This document designs the LLM serving scheduler that orchestrates that capability. The scheduler must:

- Support second-level scaling up and down of serving models according to their workloads
- Support different hardware platforms (GPU, Trainium)
- Support different models with heterogeneous resource requirements
- Support adding new models at runtime without downtime

## 2. Assumptions

- The scheduler talks to all hardware platforms via the same HTTP endpoints (`/wake_up`, `/sleep`, `/completions`, `/health`).
- The scheduler has a global view of all hardware resources in the cluster, including health status and platform type.
- Requests for each model are aggregated in a per-model queue before reaching the scheduler.
- Each machine runs 100+ standby engines; only one can be active at a time.
- At least one active engine exists per model to serve as a P2P weight sender.

## 3. Architecture Overview

The scheduler separates into two planes:

- **Control Plane** — All decision-making: what to scale, when, and where. Stateless and horizontally scalable. Can be upgraded/restarted without disrupting active inference.
- **Data Plane** — Instances running standby engine pools. Only responds to commands from the control plane. P2P weight transfers are data-plane-internal (the control plane triggers them but does not participate in bulk data movement).

```
                              ┌───────────────────────────────┐
                              │           Users               │
                              └───────────────┬───────────────┘
                                              │ requests
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CONTROL PLANE                                          │
│                                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────┐    ┌──────────────────┐  │
│  │   Model Registry     │     │   Autoscaling        │    │  Global Resource │  │
│  │                      │────▶│   Controller         │◀──▶│  Manager         │  │
│  │ • model configs      │     │                      │    │                  │  │
│  │ • resource profiles  │     │ • scaling policies   │    │ • machine health │  │
│  │ • deployment specs   │     │ • cooldown logic     │    │ • capacity map   │  │
│  │ • runtime add/remove │     │ • preemption rules   │    │ • platform types │  │
│  └──────────────────────┘     └──────────┬───────────┘    └──────────────────┘  │
│                                          │ scale decisions                      │
│                                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │              Per-Model Request Distributor (one per model)               │   │
│  │                                                                          │   │
│  │  ┌─────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │   │
│  │  │ Ingress │  │ Load Monitor │  │   Routing    │  │ Response Collector│  │   │
│  │  │ Queue   │─▶│ (QPS, queue  │─▶│ (round-robin │─▶│ (gather & return) │  │   │
│  │  │         │  │  depth, P99) │  │  / adaptive) │  │                   │  │   │
│  │  └─────────┘  └──────────────┘  └──────────────┘  └───────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└──────────────────────────────────────────────────────────────┬──────────────────┘
                                                               │ /wake_up, /sleep
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA PLANE                                            │
│                                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ Instance A           │  │ Instance B           │  │ Instance C           │   │
│  │  Standby Pool (100+) │  │  Standby Pool (100+) │  │  Standby Pool (100+) │   │
│  │  ┌────┐ ┌────┐       │  │  ┌────┐ ┌────┐       │  │  ┌────┐ ┌────┐       │   │
│  │  │ M_A│ │ M_B│ ...   │  │  │ M_A│ │ M_C│ ...   │  │  │ M_B│ │ M_D│ ...   │   │
│  │  └────┘ └────┘       │  │  └────┘ └────┘       │  │  └────┘ └────┘       │   │
│  │  Active: M_A         │  │  Active: M_C         │  │  Active: M_B         │   │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘   │
│                                                                                 │
│  P2P RDMA weight transfers happen directly between instances (data plane only)  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4. Control Plane Components

### 4.1 Model Registry

The source of truth for model metadata. Supports runtime registration without disrupting serving.

**Responsibilities:**
- Store model deployment configurations (TP degree, compiled NEFF paths, resource profile)
- Map models to compatible hardware platforms
- Define resource profiles: how much host memory, how many NeuronCores, which instance types
- Accept runtime model registration/deregistration

**Interface:**
```
POST   /models/register       — register a new model + deployment config
DELETE /models/{model_id}     — deregister a model (triggers graceful drain)
GET    /models                — list all registered models and their status
GET    /models/{model_id}     — get config and current allocation for a model
```

### 4.2 Global Resource Manager

A read-only view of cluster capacity. It observes and reports; it does NOT make decisions.

**Responsibilities:**
- Track machine health via periodic `/health` polling
- Maintain a capacity map: which instances exist, their platform type, available memory
- Track the current model→instance mapping (which engines are active/sleeping where)
- Detect and report faulty machines (the Autoscaling Controller decides what to do)

**State it maintains:**
```
Instance ID  │ Platform   │ Health  │ Active Model │ Standby Models
─────────────┼────────────┼─────────┼──────────────┼─────────────────
i-001        │ trn1.32xl  │ healthy │ model_A      │ [B, C, D, ...]
i-002        │ trn1.32xl  │ healthy │ model_C      │ [A, B, E, ...]
i-003        │ p5.48xl    │ degraded│ model_F      │ [G, H, ...]
```

### 4.3 Autoscaling Controller

The brain of the scheduler. The only component that mutates the model→instance mapping.

**Responsibilities:**
- Receive load metrics from Per-Model Request Distributors
- Decide when to scale up (wake) or scale down (sleep) engines
- Select which instance to wake/sleep on (sender selection, placement)
- Handle preemption when cluster is fully utilized
- Enforce cooldown periods to prevent thrashing

**Decision inputs and outputs:**
```
Inputs:                              Outputs:
┌────────────────────────┐           ┌────────────────────────────┐
│ Per-model load metrics │           │ wake_up(instance, model,   │
│ Per-model active count │──────────▶│          sender_instance)  │
│ Available idle engines │           │ sleep(instance, model)     │
│ Machine health status  │           │ preempt(instance, victim,  │
│ Model priority levels  │           │          beneficiary)      │
└────────────────────────┘           └────────────────────────────┘
```

**Scaling policy (three cases, in order of complexity):**

| Case | Trigger | Action | Latency |
|------|---------|--------|---------|
| Scale up from idle | Load exceeds capacity, idle engine exists | `wake_up` sleeping engine | ~5–7s |
| Scale down to idle | Load drops below threshold for cooldown period | `sleep` excess engine | ~2s |
| Preemption | No idle engines, high-priority model needs capacity | `sleep` victim + `wake_up` beneficiary | ~7–9s |

**Preemption strategies (configurable):**
- **LRU** — Sleep the model whose engines last served traffic longest ago
- **Priority-based** — Models have explicit priority tiers; lower-priority models yield first
- **Excess-capacity** — Sleep the model with the most engines relative to its current load

### 4.4 Per-Model Request Distributor

A lightweight, stateless routing proxy. One logical instance per registered model (can share process for low-traffic models).

**Responsibilities:**
- Accept requests from the model's ingress queue
- Route to any active engine for that model
- Report load metrics **up** to the Autoscaling Controller (does NOT make scaling decisions)
- Collect and return responses to callers

**Routing strategies:**
- Round-robin (default, sufficient when engines are homogeneous)
- Least-connections (when request durations vary)
- Adaptive (route away from engines about to sleep)

**Key property:** This component scales independently per model — high-traffic models can have multiple distributor instances; low-traffic models share one.

## 5. Data Plane

The data plane is the set of instances running standby engine pools. Each instance:
- Hosts 100+ sleeping engines across multiple models
- Has exactly one active engine at a time (holding NeuronCores)
- Exposes HTTP endpoints for lifecycle management (`/wake_up`, `/sleep`, `/health`)
- Handles P2P RDMA weight transfers directly with peer instances

The control plane never touches bulk weight data. It only tells the data plane "wake model X on instance B, using instance A as sender." The data plane executes the P2P protocol autonomously.

```
Control Plane command:
  wake_up(instance=B, model=X, sender=A)

Data Plane execution:
  1. Instance B: sleep current active engine (~2s)
  2. Instance B → Instance A: POST /p2p_preconnect (exchange RDMA metadata)
  3. Instance B → Instance A: POST /p2p_push_weights (trigger RDMA write)
  4. Instance A: pushes weights via RDMA (~2s for 70B model)
  5. Instance B: reload kernels, begin serving
```

## 6. Key Flows

### 6.1 Scale Up (Happy Path)

```
1. Request Distributor detects: queue_depth > threshold
2. Distributor reports load metrics to Autoscaling Controller
3. Controller queries Resource Manager: "which instances have a sleeping engine for model X?"
4. Controller selects target instance (closest to an active sender, least loaded)
5. Controller selects sender instance (active engine of model X, least loaded)
6. Controller issues: wake_up(target, model_X, sender)
7. Target instance executes P2P wake protocol (~5-7s)
8. Target reports healthy; Controller updates Resource Manager
9. Distributor begins routing to the new engine
```

### 6.2 Scale Down

```
1. Distributor reports: load dropped below threshold
2. Controller waits for cooldown period (e.g., 60s) to confirm sustained low load
3. Controller selects engine to sleep (least recently used, or explicit policy)
4. Controller drains the engine (stop routing new requests, wait for in-flight to complete)
5. Controller issues: sleep(instance, model_X)
6. Instance releases NeuronCores, enters standby (~2s)
7. Controller updates Resource Manager
```

### 6.3 Preemption

```
1. Model Y needs more capacity, but no idle engines exist
2. Controller evaluates preemption candidates across all models:
   - Which models have excess capacity relative to load?
   - Which models have lower priority?
3. Controller selects victim: model Z on instance C
4. Controller drains victim engine (stop routing, wait for in-flight)
5. Controller issues: sleep(instance_C, model_Z)
6. Controller issues: wake_up(instance_C, model_Y, sender_instance)
7. Total preemption latency: drain + sleep + wake (~9-12s)
```

### 6.4 Register New Model at Runtime

```
1. Admin: POST /models/register {model_id, config, resource_profile, standby_count}
2. Model Registry validates config, persists metadata
3. Autoscaling Controller receives event: "new model registered"
4. Controller selects instances based on resource profile + platform match
5. Controller issues: deploy_standby(instances[], model_config)
6. Data plane instances initialize standby engines in background (~15s each)
7. Model becomes routable once first engine reports healthy
8. No user-facing cold start — standby is pre-initialized before traffic arrives
```

**Standby placement: how many instances init a standby engine for a new model?**

The registration request includes a `standby_count` parameter — the number of instances that should initialize a sleeping engine for this model. This determines how many instances can potentially serve the model without cross-model preemption.

Factors that determine `standby_count`:
- **Expected peak concurrency** — If a model may need up to N engines at peak, it needs at least N standby engines across the cluster.
- **Redundancy** — Extra standby engines beyond peak provide fault tolerance (if one instance fails, others can still serve).
- **Host memory budget** — Each sleeping engine consumes ~3 GB host RAM. An instance with 495 GB usable RAM can host ~100 sleeping engines total across all models. The controller must not over-commit.

The controller enforces a cluster-wide memory budget:
```
For each candidate instance:
  available_standby_slots = (usable_RAM - active_engine_RAM - Σ sleeping_engine_RAM) / per_engine_RAM
  if available_standby_slots >= 1:
    deploy standby here
```

**Does starting a standby engine affect the active engine's performance?**

Standby initialization is CPU-bound (Python imports, NEFF compilation) and does NOT touch NeuronCores or device memory. The active engine's inference runs entirely on NeuronCores. Therefore:

- **No inference latency impact** — NeuronCore compute is uncontested.
- **Transient CPU pressure** — During the ~15s initialization window, CPU-bound tasks (tokenization, scheduling) on the active engine may see slight slowdown. In practice this is negligible because inference is NeuronCore-bound, not CPU-bound.
- **Mitigation** — Engines are launched with 2s stagger (see Section 4.3 of FAST_MODEL_SWITCH_DEMO.md) to avoid CPU thundering herd. The controller can also rate-limit concurrent standby deployments per instance.

## 7. Serving Modes

Each registered model operates in one of two modes. The mode is set at registration time and can be changed at runtime.

### 7.1 Always-Active Mode (default)

At least one engine for this model remains active at all times. This engine serves dual purposes:
- Handles inference requests (even at low load, keeping latency minimal)
- Acts as P2P weight sender when scaling up additional engines

```
State machine (always-active):

  ┌──────────────────────────────────────────────────────┐
  │              Always ≥ 1 active engine                 │
  │                                                      │
  │  [1 active] ──scale up──▶ [N active] ──scale down──┐ │
  │       ▲                                             │ │
  │       └─────────────── (keep last one) ─────────────┘ │
  └──────────────────────────────────────────────────────┘
```

**Properties:**
- First request always served immediately (no cold start)
- Scale-up latency: ~5–7s (P2P transfer from the always-active sender)
- Cost: one instance per model is always occupied, even at zero traffic

**Best for:** Production models with SLA requirements, latency-sensitive workloads, models that receive at least sporadic traffic.

### 7.2 Fully-Sleeping Mode

All engines for this model can be put to sleep. When traffic arrives and no active engine exists, the system falls back to disk-based weight loading.

```
State machine (fully-sleeping):

  [all sleeping] ──traffic arrives──▶ [1 active, disk load] ──scale up──▶ [N active, P2P]
        ▲                                        │
        └──────── scale down (all idle) ─────────┘
```

**Properties:**
- Zero resource cost when idle (no instance occupied)
- First-request cold start: ~35–45s (local NVMe) or longer (remote storage)
- Once first engine is active, subsequent scale-ups use fast P2P (~5–7s)
- The controller transitions back to fully-sleeping after sustained inactivity (configurable timeout)

**Best for:** Dev/test models, low-priority batch workloads, models with predictable traffic windows (can pre-warm before expected traffic).

**Disk-based wake flow (fully-sleeping, no active sender):**
```
1. Request arrives for model X; no active engine exists
2. Controller selects instance with sleeping engine for model X
3. Controller issues: wake_up(instance, model_X, source="disk")
4. Instance loads weights from local NVMe (~35s) or remote storage
5. Engine becomes active; begins serving queued requests
6. Subsequent scale-ups use this engine as P2P sender (fast path)
```

### 7.3 Mode Selection in Registration

```
POST /models/register {
  model_id: "llama-3-70b",
  config: { tp_degree: 32, neff_path: "...", ... },
  resource_profile: { host_memory_gb: 3.0, platform: "trn1" },
  standby_count: 10,
  serving_mode: "always_active" | "fully_sleeping",
  sleeping_timeout_s: 300   // only for always_active: go fully sleeping after 5min zero traffic
}
```

The `sleeping_timeout_s` field enables a hybrid behavior: a model starts in always-active mode but transitions to fully-sleeping after prolonged inactivity. This avoids paying for a permanently active engine for rarely-used models while still providing fast P2P scaling during active periods.

## 8. Sender Selection (Sequential)

When waking an engine, the controller must choose which active instance serves as the P2P weight sender. This matters at scale with many potential senders.

**Selection criteria (ordered by priority):**
1. **Health** — Only healthy senders
2. **Load** — Prefer senders with lower current request load (RDMA push has ~3% latency impact)
3. **Recency** — Prefer senders that haven't pushed recently (avoid QP exhaustion / MR re-registration overlap)
4. **Topology** — Prefer senders in the same availability zone (lower network latency)

In the common case (small cluster, few active engines per model), the choice is trivial — there's often only one sender. The selection logic matters at scale when a model has 10+ active engines.

## 9. Design Decisions

Decisions made at this stage to keep the initial implementation simple:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scheduler topology | Single global scheduler | Simplicity. Stateless control plane can be restarted quickly; HA via active-passive failover is sufficient for v1. Federate later if cluster grows beyond one region. |
| Scaling granularity | Sequential (one engine at a time) | Avoids bandwidth contention on sender. APIs are designed to be general — parallel wake can be added later without interface changes. |
| Minimum active guarantee | Two modes (always-active / fully-sleeping) | See Section 7. Covers both latency-sensitive production models and cost-sensitive dev/batch models. |
| Scaling signal | Reactive (queue depth + active count) | The ~5s reallocation latency is below traffic burst timescales. Prediction can be layered on later as an optimization. |

## 10. Open Design Questions

Remaining questions for future iterations:

1. **Preemption policy** — When the cluster is fully utilized, which model yields? Options: pure priority tiers, priority + fairness (guaranteed minimum allocation), or load-proportional. Needs workload characterization data to decide.

2. **Failure handling** — If a `wake_up` fails mid-transfer (sender crash, network partition), should the controller retry on the same instance or failover to a different one? Current implementation returns 503 and cleans up (see FAST_MODEL_SWITCH_DEMO.md §4.11); the controller needs a retry policy.

3. **Scaling signal refinement** — Should the controller use request rate (forward-looking) in addition to queue depth (current state)? Rate-based signals could trigger scale-up slightly earlier.

4. **Standby rebalancing** — As models are added/removed, the standby distribution across instances may become uneven. Should the controller periodically rebalance sleeping engines, or only adjust on registration events?

5. **Multi-instance parallel wake** — When a model needs to scale from 1 to N engines quickly, can the sender push to multiple receivers simultaneously? What is the bandwidth impact on the sender's inference latency? (Deferred — sequential is sufficient for v1.)


## Implementation task

- Implement this control plane as a separate github repo using Python. 
- Add unit tests and integration tests for each major functionalities. 
- For the end-to-end tests, you can use the two available trn1 instances, 172.31.44.131 and 172.31.40.200. 
- Document encountered issues and solutions.