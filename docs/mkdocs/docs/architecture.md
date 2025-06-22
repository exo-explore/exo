# Architecture Overview

- EXO is a distributed AI inference framework that unifies heterogeneous devices (Mac, Linux, Raspberry Pi, etc.) into a single cluster.
- Devices connect peer-to-peer (no master-worker), and models are partitioned across available memory.
- The architecture supports multiple inference engines (MLX, tinygrad, etc.).
- Device discovery is automatic, and the system is designed for minimal configuration.


# EXO High-Level Architecture

```mermaid
flowchart TD
    User[User/API/Web UI]
    subgraph Cluster
      Device1[Device 1]
      Device2[Device 2]
      Device3[Device 3]
    end
    Model[AI Model Partitioned]
    User -->|Request| Device1
    User -->|Request| Device2
    User -->|Request| Device3
    Device1 -->|Distributed Inference| Model
    Device2 -->|Distributed Inference| Model
    Device3 -->|Distributed Inference| Model
```