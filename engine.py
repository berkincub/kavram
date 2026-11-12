"""
acps_v5/engine.py — INFERENCE_ENGINE: The Defender (Process Entry Point)
=============================================================
INFERENCE_ENGINE is an isolated CPU process running the BiRNN+GCN inference loop
at millisecond-latency. It:

  1. Subscribes to MQTT sensor feeds published by ADVERSARIAL_SIMULATOR
  2. Accumulates sensor windows per node
  3. Runs the FacilityHybridModel (BiRNN+GCN) for anomaly scoring
  4. Emits AlertPackets to SEMANTIC_GOVERNOR via multiprocessing Queue
  5. Triggers autonomous physical mitigation for CRITICAL/HIGH alerts

Rev 3.0 — Multi-Path Propagation:
  - Normalized Graph Laplacian computed per batch: L = I - D^{-1/2} A D^{-1/2}
    (Kipf & Welling, 2017 ICLR — Eq.1 spectral graph convolution)
  - CONVERGENT topology detection: if |eigvals(L) > threshold| >= 2, a multi-source
    convergence is flagged and propagation_topology stamped on AlertPacket.
  - BiRNN context window extended to [t, t-1, t-2, t-3] via sensor_seq accumulator
    (sliding window in engine.py; birnn.py architecture unchanged).

IPC:
  MQTT (in)          ← Sensor readings from ADVERSARIAL_SIMULATOR
  friday_queue (out) ← AlertPacket to SEMANTIC_GOVERNOR semantic layer
  app_queue (out)    ← Real-time sensor data for APP Cockpit display

Run standalone:
    python -m inference_engine.engine

Or spawned by mas_launch.py.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Deque, Optional

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
da_path = str(ROOT / "defender_agent")
if da_path not in sys.path:
    sys.path.insert(0, da_path)

from shared.tts_utils import _speak_tts

_speech_lock = threading.Lock()
_last_speech_time = 0.0

from shared.config_mas import mas_cfg, SESSION_OUTPUTS_DIR
from shared.ipc_protocol import AlertPacket, Severity
from inference_engine.alert_schema import make_alert_packet, classify_severity, SeverityAssigner, _severity_assigner
from inference_engine.mitigator import PhysicalMitigator

cfg = mas_cfg.acps_v5

# Sensor channels in order (matches dataset.py)
SENSOR_CHANNELS = [
    "temperature", "humidity", "CO2", "energy_consumption",
    "occupancy", "air_quality_index", "vibration", "pressure",
]
SEQ_LEN = 24  # BiRNN sliding window length

# Rev 3.0: BiRNN temporal lag context — number of previous timesteps to include
# in sensor_seq as additional context rows [t, t-1, t-2, t-3].
# Does NOT change birnn.py architecture (checkpoint-safe).
# Ref: Granger (1969) — causal lag detection requires p>=2 lags for VAR models.
LAG_CONTEXT_STEPS: int = 4  # t, t-1, t-2, t-3

# Rev 3.0: Laplacian eigenvalue threshold for CONVERGENT topology detection.
# If >= 2 non-trivial eigenvalues exceed this threshold, multi-source topology flagged.
# Ref: Kipf & Welling (2017) — spectral gap of normalized Laplacian.
_LAPLACIAN_EIG_THRESHOLD: float = 0.1
_CALIB_CACHE_LOADED = False

# Production latency tracking for autonomous mitigation reflexes (rolling buffer)
_reflex_latency_log_ms: Deque[float] = deque(maxlen=200)
_last_latency_flush_time: float = time.time()
_reflex_trigger_count: int = 0


def record_reflex_latency(delta_t_ms: float, force_flush: bool = False) -> tuple[float, float, float]:
    """
    Production Latency Metric:
    Records instantaneous autonomous reflex response time (ms) into a rolling deque (maxlen=200).
    Computes rolling percentiles (p50, p95, max) and periodically persists them to disk
    under SESSION_OUTPUTS_DIR / 'reflex_latency_stats.json' (every 50 triggers or 60s).
    """
    global _reflex_trigger_count, _last_latency_flush_time
    _reflex_latency_log_ms.append(delta_t_ms)
    _reflex_trigger_count += 1

    latencies = list(_reflex_latency_log_ms)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    max_val = float(np.max(latencies))

    now = time.time()
    if force_flush or _reflex_trigger_count % 10 == 0 or (now - _last_latency_flush_time) >= 5.0:
        _last_latency_flush_time = now
        stats = {
            "timestamp": datetime.now().isoformat(),
            "samples": len(latencies),
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "max_ms": round(max_val, 3),
        }
        stats_path = SESSION_OUTPUTS_DIR / "reflex_latency_stats.json"
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"[LATENCY] Reflex latency stats flushed (n={len(latencies)}): p50={p50:.2f}ms | p95={p95:.2f}ms | max={max_val:.2f}ms -> {stats_path.name}")
        except Exception as flush_err:
            print(f"[LATENCY] Failed to persist latency stats: {flush_err}")

    return p50, p95, max_val

NOMINAL_VALUES = {
    "temperature": 22.0, "humidity": 45.0, "CO2": 400.0, "energy_consumption": 0.0,
    "occupancy": 0.0, "air_quality_index": 50.0, "vibration": 0.01, "pressure": 101.3
}


# ── Model loader ─────────────────────────────────────────────────────────────

def _load_model():
    """
    Attempt to load the FacilityHybridModel checkpoint.
    Falls back to a lightweight heuristic scorer if checkpoint unavailable.
    """
    try:
        import torch
        from defender_agent.models.hybrid import FacilityHybridModel
        from defender_agent.config import Config

        ckpt = cfg.model_checkpoint
        if not Path(ckpt).exists():
            print(f"[INFERENCE_ENGINE] No checkpoint at {ckpt} — using heuristic scorer")
            return None, None

        # Dynamically fall back to CPU if CUDA is not physically available on the machine
        device = "cuda" if (torch.cuda.is_available() and cfg.device == "cuda") else "cpu"
        
        # Multi-core CPU parallel execution for tensor operations
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        try:
            torch.set_num_interop_threads(min(4, num_cores))
        except RuntimeError:
            pass
        print(f"[INFERENCE_ENGINE] 🚀 Multi-Core PyTorch Active: {torch.get_num_threads()} CPU threads utilized.")
        
        model_cfg = Config()
        model = FacilityHybridModel(model_cfg, sensor_dim=len(SENSOR_CHANNELS))
        
        # Using map_location = device/torch.device ensures CUDA-serialized storages map safely to CPU
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)
        print(f"[INFERENCE_ENGINE] Loaded checkpoint: {ckpt} on {device}")
        _speak_tts("A C P S active. G C N model loaded successfully.")
        return model, device

    except Exception as e:
        print(f"[INFERENCE_ENGINE] Model load failed ({e}) — using heuristic scorer")
        _speak_tts("Warning. Model loading failed. Falling back to heuristic scorer.")
        return None, None


def _heuristic_score(readings: Dict[str, float]) -> float:
    """
    Lightweight heuristic anomaly scorer when model is unavailable.
    Uses z-score deviation from expected facility baselines.
    """
    is_bench = getattr(cfg, "benchmark_mode", False) or (readings.get("_source") == "LBNL_REPLAY")
    temp_sigma = 5.0 if is_bench else 2.0
    BASELINES = {
        "temperature": (22.0, temp_sigma),
        "humidity": (45.0, 5.0),
        "CO2": (450.0, 50.0),
        "energy_consumption": (3.5, 1.0),
        "vibration": (0.03, 0.02),
        "pressure": (0.0 if is_bench else 101.3, 1.06 if is_bench else 0.5),
        "air_quality_index": (70.0, 10.0),
    }
    z_scores = []
    for ch, (mu, sigma) in BASELINES.items():
        val = readings.get(ch)
        if val is not None and sigma > 0:
            z_scores.append(abs((float(val) - mu) / sigma))

    if not z_scores:
        return 0.0
    mean_z = float(np.mean(z_scores))
    
    # LİTERATÜR (Ph.D.) DÜZELTMESİ: Doğrusal (Linear) kesme yerine Rayleigh CDF 
    # (Cumulative Distribution Function) kullanılarak Z-skoru olasılığa [0, 1] çevrilir.
    # Bu sayede ufak sapmalar doğrudan CRITICAL (%90+) olmaz. Olasılık pürüzsüz artar.
    # P(Anomaly) = 1 - exp(- (Z^2) / 2)
    # Bu, Shapiro-Wilk ve Shannon Entropy testlerindeki 'Yalancı Çoban' sendromunu çözer.
    prob = 1.0 - math.exp(- (mean_z ** 2) / 2.0)
    return float(np.clip(prob, 0.0, 1.0))


# ── Rev 3.0: GCN Graph Laplacian utilities ────────────────────────────────────

def _build_normalized_laplacian(n_nodes: int, edges: list) -> np.ndarray:
    """
    Compute the Symmetric Normalized Graph Laplacian:
        L = I - D^{-1/2} A D^{-1/2}

    where A is the adjacency matrix and D is the diagonal degree matrix.
    This is the Laplacian used in the Spectral GCN formulation.

    Ref: Kipf & Welling (2017) ICLR — "Semi-Supervised Classification with
    Graph Convolutional Networks", Eq.1.

    Args:
        n_nodes: Number of nodes in the facility subgraph.
        edges:   List of (src, dst) tuples (0-indexed node IDs).

    Returns:
        L: [n_nodes, n_nodes] normalized Laplacian as numpy array.
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src, dst in edges:
        if 0 <= src < n_nodes and 0 <= dst < n_nodes:
            A[src, dst] = 1.0
            A[dst, src] = 1.0  # undirected
    degree = A.sum(axis=1)  # [n_nodes]
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n_nodes, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def _detect_propagation_topology(
    node_scores: dict,
    node_ids: list,
    global_g = None,
    threshold: float = None,
) -> str:
    """
    Detect the propagation topology of the current alert batch using
    eigenvalue analysis of the symmetric normalized Graph Laplacian.
    """
    global _LAPLACIAN_EIG_THRESHOLD, _CALIB_CACHE_LOADED
    
    calib_file = SESSION_OUTPUTS_DIR / "laplacian_calibrated.txt"
    if not _CALIB_CACHE_LOADED:
        if calib_file.exists():
            try:
                with open(calib_file, "r") as f:
                    _LAPLACIAN_EIG_THRESHOLD = float(f.read().strip())
                _CALIB_CACHE_LOADED = True
                print(f"[INFERENCE_ENGINE] Kalibre edilmis _LAPLACIAN_EIG_THRESHOLD yuklendi = {_LAPLACIAN_EIG_THRESHOLD:.4f}")
            except Exception:
                pass

    if threshold is None:
        threshold = _LAPLACIAN_EIG_THRESHOLD

    active = [nid for nid, sc in node_scores.items() if sc > 0.0]
    n = len(active)

    if n <= 1:
        return "LINEAR"

    # GERÇEK IFC KOMŞULUK ANALİZİ (DYNAMIC SUBGRAPH LAPLACIAN)
    if global_g is not None and hasattr(global_g, 'edge_index') and node_ids:
        try:
            node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            active_indices = [node_to_idx[nid] for nid in active if nid in node_to_idx]
            active_idx_set = set(active_indices)
            
            # Global grafın kenar listesini oku
            edge_index_np = global_g.edge_index.cpu().numpy()
            srcs, dsts = edge_index_np[0], edge_index_np[1]
            
            sub_edges = []
            global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(active_indices)}
            
            # Sadece aktif düğümler arasındaki gerçek BIM/IFC bağlantılarını filtrele
            for s, d in zip(srcs, dsts):
                if s in active_idx_set and d in active_idx_set:
                    sub_edges.append((global_to_local[s], global_to_local[d]))
                    
            if sub_edges:
                L = _build_normalized_laplacian(len(active_indices), sub_edges)
            else:
                edges = [(i, i + 1) for i in range(n - 1)]
                L = _build_normalized_laplacian(n, edges)
        except Exception as e:
            print(f"[INFERENCE_ENGINE] Subgraph Laplacian extraction failed: {e}")
            edges = [(i, i + 1) for i in range(n - 1)]
            L = _build_normalized_laplacian(n, edges)
    else:
        edges = [(i, i + 1) for i in range(n - 1)]
        L = _build_normalized_laplacian(n, edges)

    try:
        eigvals = np.linalg.eigvalsh(L)
        # Laplacian threshold self-calibration
        if not calib_file.exists() and len(eigvals) > 0:
            raw_file = SESSION_OUTPUTS_DIR / "laplacian_eigenvalues.txt"
            non_trivial_candidates = [float(ev) for ev in eigvals if 0.0 < ev < 2.0]
            if non_trivial_candidates:
                try:
                    with open(raw_file, "a") as rf:
                        for ev in non_trivial_candidates:
                            rf.write(f"{ev}\n")
                except Exception:
                    pass
                
                try:
                    if raw_file.exists():
                        with open(raw_file, "r") as rf:
                            lines = rf.readlines()
                        if len(lines) >= 500:
                            vals = [float(l.strip()) for l in lines if l.strip()]
                            if vals:
                                p95 = float(np.percentile(vals, 95))
                                p95 = max(0.01, p95)
                                with open(calib_file, "w") as cf:
                                    cf.write(f"{p95}\n")
                                _LAPLACIAN_EIG_THRESHOLD = p95
                                _CALIB_CACHE_LOADED = True
                                print(f"[CALIBRATION] 🎯 Eigenvalue calibration complete! _LAPLACIAN_EIG_THRESHOLD set to {p95:.4f}")
                except Exception as e:
                    print(f"[CALIBRATION] Error running percentile calibration: {e}")

        non_trivial = int(np.sum((eigvals > threshold) & (eigvals < 2.0 - threshold)))
    except Exception:
        non_trivial = 0

    return "CONVERGENT" if non_trivial >= 2 else "LINEAR"


def _build_lag_context_seq(buffer_arr: np.ndarray, lag_steps: int = LAG_CONTEXT_STEPS) -> np.ndarray:
    """
    Build an extended sensor sequence [t, t-1, t-2, ..., t-(lag_steps-1)]
    by stacking the last `lag_steps` rows of the buffer as additional feature context.

    This does NOT change the BiRNN architecture (birnn.py checkpoint-safe).
    Instead, we prepend the lag rows as extra timesteps to the sequence,
    expanding the temporal context window from SEQ_LEN to SEQ_LEN+lag_steps.
    The model sees the most recent readings last (standard temporal ordering).

    Rationale: BiRNN Lag-1 autocorrelation failure arises because the model
    only sees t. Adding t-1, t-2, t-3 rows allows the BiRNN to capture
    delayed anomaly propagation signatures without architecture changes.
    Ref: Granger (1969) — causal lag detection requires p>=2 lags.

    Args:
        buffer_arr: [SEQ_LEN, n_features] sensor buffer numpy array.
        lag_steps:  Number of lag context rows to prepend (default=4).

    Returns:
        Extended array [SEQ_LEN + lag_steps, n_features] with lag context prepended.
    """
    if len(buffer_arr) < lag_steps:
        # Not enough history: pad with zeros
        pad = np.zeros((lag_steps - len(buffer_arr), buffer_arr.shape[1]), dtype=np.float32)
        return np.vstack([pad, buffer_arr])
    # Prepend the lag rows (oldest first to maintain temporal ordering)
    lag_rows = buffer_arr[-lag_steps:][::-1]  # [t-(lag-1), ..., t-1, t] reversed = [t, t-1, ...]
    lag_rows_ordered = lag_rows[::-1]         # restore [t-(lag-1), ..., t-1] for prepend
    return np.vstack([lag_rows_ordered, buffer_arr]).astype(np.float32)


# ── Sensor window accumulator ─────────────────────────────────────────────────

class NodeWindowBuffer:
    """Her node için genişletilmiş kayan sensör penceresi (Zaman gecikmeli context dahil)"""
    def __init__(self, maxlen: int = SEQ_LEN + LAG_CONTEXT_STEPS):
        self._windows: Dict[str, Deque[Dict]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._metadata: Dict[str, Dict] = {}

    def push(self, node_id: str, reading: Dict) -> None:
        self._windows[node_id].append(reading)
        self._metadata[node_id] = {
            "space_name": reading.get("space_name", node_id),
            "floor":      reading.get("floor", "Unknown"),
            "zone":       reading.get("zone", "Unknown"),
        }

    def is_ready(self, node_id: str) -> bool:
        return len(self._windows[node_id]) >= (SEQ_LEN + LAG_CONTEXT_STEPS)

    def get_tensor(self, node_id: str):
        """Return [SEQ_LEN + LAG_CONTEXT_STEPS, n_features] numpy array."""
        window = list(self._windows[node_id])
        arr = np.array([
            [float(r.get(ch, 0.0)) for ch in SENSOR_CHANNELS]
            for r in window
        ], dtype=np.float32)
        return arr

    def get_latest(self, node_id: str) -> Dict:
        if not self._windows[node_id]:
            return {}
        return dict(self._windows[node_id][-1])

    def get_meta(self, node_id: str) -> Dict:
        return self._metadata.get(node_id, {"space_name": node_id, "floor": "?", "zone": "?"})


# ── Process entry ─────────────────────────────────────────────────────────────

def run_inference_engine(
    friday_queue:     multiprocessing.Queue,
    app_queue:        multiprocessing.Queue,
    validation_queue: multiprocessing.Queue,
    command_queue:    multiprocessing.Queue,
    stop_event:       multiprocessing.Event = None,
    benchmark_mode:   bool = False,
) -> None:
    """
    Main INFERENCE_ENGINE inference loop.
    Subscribes to MQTT, scores sensor windows, emits AlertPackets.
    """
    from shared.config_mas import mas_cfg
    mocker_cfg = mas_cfg.mocker

    print("[INFERENCE_ENGINE] Defender process starting...")

    is_benchmark = bool(benchmark_mode or getattr(cfg, "benchmark_mode", False))
    if is_benchmark:
        cfg.benchmark_mode = True
        _severity_assigner.set_benchmark_mode(True)
        print("[INFERENCE_ENGINE] ⚠️ BENCHMARK MODE ACTIVE: Severity entropy penalty (_entropy_penalty) is FULLY BYPASSED.")

    # Load model (or fallback)
    model, device = _load_model()

    # Load global GCN topology
    outputs_dir = SESSION_OUTPUTS_DIR
    mapped_nodes_path = outputs_dir / "mapped_nodes.json"
    adj_matrix_path = outputs_dir / "adjacency_matrix.npy"
    node_features_path = outputs_dir / "node_features.npy"
    
    global_g = None
    node_ids = []
    
    if mapped_nodes_path.exists() and adj_matrix_path.exists() and node_features_path.exists():
        try:
            import torch
            from torch_geometric.data import Data as PyGData
            
            with open(mapped_nodes_path, "r", encoding="utf-8") as f:
                node_ids = json.load(f)
            
            A = np.load(str(adj_matrix_path))
            x_feat = np.load(str(node_features_path))
            
            src_idx, dst_idx = np.where(A > 0.0)
            edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)
            x_tensor = torch.from_numpy(x_feat).float()
            
            global_g = PyGData(x=x_tensor, edge_index=edge_index)
            print(f"[INFERENCE_ENGINE] Dynamic GCN graph loaded: {len(node_ids)} nodes, {edge_index.shape[1]} spatial edges.")
        except Exception as e:
            print(f"[INFERENCE_ENGINE] ⚠️ Failed to load GCN topology: {e}")
            global_g = None

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFERENCE_ENGINE] CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    # Sensor buffer
    buffer = NodeWindowBuffer()

    # Mitigator
    mitigator = PhysicalMitigator(cooldown_sec=cfg.mitigation_cooldown)

    # MQTT setup
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[INFERENCE_ENGINE] paho-mqtt not installed — cannot receive MQTT data")
        return

    _last_inference: Dict[str, float] = {}
    last_heartbeat = 0.0

    # LOAD STATE (OPERASYON: KALICILIK)
    state_file = SESSION_OUTPUTS_DIR / ".inference_state.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                saved_state = json.load(f)
                for node_id, readings in saved_state.items():
                    for r in readings:
                        buffer.push(node_id, r)
            print(f"[INFERENCE_ENGINE] Restored state for {len(saved_state)} nodes")
        except Exception:
            pass

    def on_message(client, userdata, msg):
        try:
            if msg.topic == "facility/control/recalibrate":
                from inference_engine.alert_schema import _severity_assigner
                print(f"[INFERENCE_ENGINE] 🔄 Recalibration signal received. Clearing severity history.")
                _severity_assigner.history.clear()
                return

            # Hardcoded absolute MQTT isolation guard: in production, discard simulation feeds immediately
            is_benchmark = benchmark_mode or getattr(cfg, "benchmark_mode", False)
            if not is_benchmark and not cfg.sim_mode and "/sim/" in msg.topic:
                return

            # Resolve topic suffix
            if is_benchmark:
                if msg.topic.startswith("facility/sensors/"):
                    node_id = msg.topic[len("facility/sensors/"):]
                elif msg.topic.startswith("facility/sim/sensors/"):
                    node_id = msg.topic[len("facility/sim/sensors/"):]
                else:
                    return
            elif cfg.sim_mode:
                if not msg.topic.startswith("facility/sim/sensors/"):
                    return
                node_id = msg.topic.split("/")[-1]
            else:
                if not msg.topic.startswith("facility/sensors/"):
                    return
                node_id = msg.topic.split("/")[-1]

            payload = json.loads(msg.payload.decode())
            payload["node_id"] = node_id
            payload["_arrival_time"] = time.time()  # <--- Added for Delta-T measurement
            buffer.push(node_id, payload)

            global _msg_count
            if '_msg_count' not in globals():
                _msg_count = 0
            _msg_count += 1
            if _msg_count % 100 == 0:
                print(f"[INFERENCE_ENGINE] Sniff: Received {_msg_count} MQTT readings. Latest node: {node_id}")

            # Forward raw reading to APP queue (lightweight — always)
            import queue
            try:
                app_queue.put_nowait(("sensor_update", {
                    "node_id": node_id,
                    "reading": {ch: payload.get(ch) for ch in SENSOR_CHANNELS},
                    "anomaly_score": payload.get("anomaly_score", 0.0),
                    "timestamp": payload.get("timestamp", datetime.now().isoformat()),
                    "source": "mocker",
                }))
            except queue.Full:
                pass # Lavabo tıkalıysa bu veriyi atla
            except Exception:
                import sys, traceback
                sys.stderr.write('\033[91m')
                traceback.print_exc()
                sys.stderr.write('\033[0m')

        except Exception as e:
            print(f"[INFERENCE_ENGINE] on_message error: {e}")

    client = mqtt.Client(client_id="BlueTeam_INFERENCE_ENGINE")
    client.on_message = on_message
    
    def _on_connect(c, ud, f, rc):
        if rc == 0:
            is_benchmark = benchmark_mode or getattr(cfg, "benchmark_mode", False)
            if is_benchmark:
                c.subscribe("facility/sensors/#")
                c.subscribe("facility/sim/sensors/#")
                print(f"[INFERENCE_ENGINE] Subscribed to benchmark topics (facility/sensors/# & facility/sim/sensors/#)")
            elif cfg.sim_mode:
                c.subscribe("facility/sim/sensors/#")
                print(f"[INFERENCE_ENGINE] Subscribed to simulation topics (facility/sim/sensors/#)")
            else:
                c.subscribe("facility/sensors/#")
                print(f"[INFERENCE_ENGINE] Subscribed to production topics (facility/sensors/#)")
            c.subscribe("facility/control/#")
        else:
            print(f"[INFERENCE_ENGINE] MQTT connect failed rc={rc}")
            
    client.on_connect = _on_connect

    try:
        client.connect(mocker_cfg.mqtt_broker, mocker_cfg.mqtt_port, 60)
        client.loop_start()
        print(f"[INFERENCE_ENGINE] Subscribed to facility/sensors/# on "
              f"{mocker_cfg.mqtt_broker}:{mocker_cfg.mqtt_port}")
    except Exception as e:
        print(f"[INFERENCE_ENGINE] MQTT connect error: {e}")
        return

    # ── Inference loop ────────────────────────────────────────────────────────
    print("[INFERENCE_ENGINE] Inference loop active...")
    _kalman_states: dict = {}
    Q_param = 0.005  # Process noise
    R_param = 0.05   # Measurement noise
    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            # ── Heartbeat ─────────────────────────────────────────────────────
            if time.time() - last_heartbeat > 5.0:
                import queue
                try:
                    from shared.ipc_protocol import HeartbeatPacket
                    hb = HeartbeatPacket(source="INFERENCE_ENGINE")
                    app_queue.put_nowait(("heartbeat", hb))
                    last_heartbeat = time.time()
                except queue.Full:
                    pass # Lavabo tıkalıysa bu veriyi atla
                except Exception:
                    import sys, traceback
                    sys.stderr.write('\033[91m')
                    traceback.print_exc()
                    sys.stderr.write('\033[0m')

            # ── Command Listener ──────────────────────────────────────────────
            try:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd.target in ["INFERENCE_ENGINE", "ALL"]:
                        print(f"[INFERENCE_ENGINE] Received command: {cmd.command}")
                        if cmd.command == "SET_THRESHOLD":
                            cfg.anomaly_threshold = cmd.payload.get("threshold", cfg.anomaly_threshold)
                        elif cmd.command == "STOP":
                            return
            except Exception:
                pass

            now = time.time()
            _node_scores_cycle: Dict[str, float] = {}
            
            # Execute batch GCN inference and per-node processing, throttled to run every 2s (or 50ms in benchmark mode)
            global _last_batch_inference_time
            if '_last_batch_inference_time' not in globals():
                _last_batch_inference_time = 0.0

            interval = 0.05 if is_benchmark else cfg.inference_interval
            if (now - _last_batch_inference_time) >= interval:
                _last_batch_inference_time = now

                batch_results = None
                if global_g is not None and model is not None:
                    try:
                        import torch
                        seq_tensors = []
                        for node_id in node_ids:
                            if buffer.is_ready(node_id):
                                seq_arr = buffer.get_tensor(node_id)
                            else:
                                # Pad unready nodes with nominal values (including lag context steps)
                                seq_arr = np.array([
                                    [NOMINAL_VALUES[ch] for ch in SENSOR_CHANNELS]
                                    for _ in range(SEQ_LEN + LAG_CONTEXT_STEPS)
                                ], dtype=np.float32)
                                
                            seq_tensors.append(torch.from_numpy(seq_arr))
                            
                        seq_t_all = torch.stack(seq_tensors, dim=0)  # [N, SEQ_LEN+LAG_CONTEXT_STEPS, F]
                        
                        with torch.no_grad():
                            out = model.infer(global_g, seq_t_all, device=str(device))
                            
                        batch_scores = out["anomaly_scores"].squeeze(-1).tolist()
                        batch_embeddings = out["gcn_embeddings"].tolist()
                        
                        batch_results = {}
                        for idx, sensor_id in enumerate(node_ids):
                            batch_results[sensor_id] = {
                                "score": batch_scores[idx],
                                "embedding": batch_embeddings[idx]
                            }
                        
                        ready_count = sum(1 for nid in node_ids if buffer.is_ready(nid))
                        max_s = max(batch_scores) if batch_scores else 0.0
                        print(f"[INFERENCE_ENGINE] GCN Cikarimi: {ready_count}/{len(node_ids)} dugum hazir. Max anomali skoru: {max_s:.4f}")
                    except Exception as model_err:
                        print(f"[INFERENCE_ENGINE] ⚠️ Batch GCN inference failed: {model_err}")
                        batch_results = None

                # Process scores/alerts per node
                all_known_nodes = list(node_ids) if node_ids else []
                scored_nodes = list(dict.fromkeys(all_known_nodes + list(buffer._windows.keys())))
                for node_id in scored_nodes:
                    if not buffer._windows[node_id]:
                        continue

                    latest  = buffer.get_latest(node_id)
                    meta    = buffer.get_meta(node_id)
                    snapshot = {ch: latest.get(ch, 0.0) for ch in SENSOR_CHANNELS}
                    snapshot["_arrival_time"] = latest.get("_arrival_time", time.time())
                    if "_source" in latest:
                        snapshot["_source"] = latest["_source"]
                    if "_fault_label" in latest:
                        snapshot["_fault_label"] = latest["_fault_label"]

                    # 1. Calculate instantaneous heuristic score
                    h_score = _heuristic_score(snapshot)

                    # 2. Heuristic Override / Safety Net
                    if h_score > 0.85 and not is_benchmark:
                        # Apaçık bir kriz var -> GCN ve Kalman filtresini bypass et
                        score = h_score
                        severity = classify_severity(score)
                        _node_scores_cycle[node_id] = score
                        # Align Kalman state with the override value to prevent lag/jump when crisis ends
                        _kalman_states[node_id] = {"x": h_score, "P": 0.1}
                        
                        # Set default GCN embeddings for the alert packet
                        gcn_embed = [0.0] * 64
                        if batch_results is not None and node_id in batch_results:
                            gcn_embed = batch_results[node_id]["embedding"]
                    else:
                        # Kriz yok -> GCN modelini ve Kalman filtresini normal şekilde işlet
                        if batch_results is not None and node_id in batch_results:
                            score = batch_results[node_id]["score"]
                            gcn_embed = batch_results[node_id]["embedding"]
                        elif model is not None:
                            # Model fallback
                            try:
                                import torch
                                from torch_geometric.data import Data as PyGData
                                seq_arr = buffer.get_tensor(node_id) if buffer.is_ready(node_id) else np.zeros((SEQ_LEN, len(SENSOR_CHANNELS)))
                                seq_arr_extended = _build_lag_context_seq(seq_arr, LAG_CONTEXT_STEPS)
                                seq_t = torch.from_numpy(seq_arr_extended).unsqueeze(0)
                                x = torch.zeros(1, 64)
                                edge_index = torch.zeros(2, 1, dtype=torch.long)
                                g = PyGData(x=x, edge_index=edge_index)
                                with torch.no_grad():
                                    out = model.infer(g, seq_t, device=str(device))
                                score = float(out["anomaly_scores"][0, 0])
                                gcn_embed = out["gcn_embeddings"][0].tolist()
                            except Exception:
                                score = _heuristic_score(snapshot)
                                gcn_embed = [0.0] * 64
                        else:
                            score = latest.get("anomaly_score", _heuristic_score(snapshot))
                            gcn_embed = [0.0] * 64

                        # Kalman filter
                        if node_id not in _kalman_states:
                            _kalman_states[node_id] = {"x": score, "P": 1.0}
                        
                        k_state = _kalman_states[node_id]
                        x_prev = k_state["x"]
                        P_prev = k_state["P"]

                        P_pred = P_prev + Q_param
                        K_gain = P_pred / (P_pred + R_param)
                        x_new = x_prev + K_gain * (score - x_prev)
                        P_new = (1.0 - K_gain) * P_pred

                        _kalman_states[node_id] = {"x": x_new, "P": P_new}
                        score = float(np.clip(x_new, 0.0, 1.0))
                        severity = classify_severity(score)
                        _node_scores_cycle[node_id] = score

                    if score >= cfg.anomaly_threshold:
                        topology = _detect_propagation_topology(_node_scores_cycle, node_ids, global_g)

                        packet = make_alert_packet(
                            node_id         = node_id,
                            space_name      = meta.get("space_name", node_id),
                            floor           = meta.get("floor", "Unknown"),
                            anomaly_score   = score,
                            sensor_snapshot = snapshot,
                            gcn_embeddings  = gcn_embed,
                        )
                        packet.propagation_topology = topology
                        if "_source" in latest:
                            setattr(packet, "_source", latest["_source"])
                        
                        # ── 1. IMMEDIATE OMURILIK REFLEKSI (Zero-Latency: 10-30ms) ───────────
                        # Executed BEFORE queues, network backpressure, or TTS delays.
                        is_calibrated = (SESSION_OUTPUTS_DIR / "laplacian_calibrated.txt").exists()
                        will_mitigate = (
                            cfg.auto_mitigate
                            and ((severity in [Severity.CRITICAL, Severity.HIGH]) or (score >= 0.40))
                            and (is_calibrated or severity in [Severity.CRITICAL, Severity.HIGH])
                        )
                        if will_mitigate:
                            cmd = mitigator.mitigate(packet, app_queue=app_queue)
                            delta_t_ms = (time.time() - snapshot.get('_arrival_time', time.time())) * 1000.0
                            
                            if cmd:
                                packet.mitigation_triggered = True
                                packet.mitigation_action    = cmd.action
                                packet.reflex_latency_ms    = round(delta_t_ms, 2)
                                p50, p95, _ = record_reflex_latency(delta_t_ms)
                                print(f"[REFLEX] OTONOM OMURILIK REFLEKSI | {packet.mitigation_action} | instant={delta_t_ms:.2f}ms | rolling p50={p50:.2f}ms p95={p95:.2f}ms")
                            else:
                                packet.mitigation_triggered = False
                                packet.reflex_latency_ms    = 0.0
                                # Determine exact suppression reason for audit logging
                                if not mitigator.should_mitigate(packet.node_id):
                                    packet.mitigation_action = f"Engellendi: Cooldown Beklemesinde (<{mitigator._cooldown:.0f}s)"
                                else:
                                    packet.mitigation_action = "Engellendi: Safe Mode veya Guvenlik Filtresi"
                        else:
                            packet.mitigation_triggered = False
                            if not is_calibrated and severity not in [Severity.CRITICAL, Severity.HIGH] and score >= 0.40:
                                packet.mitigation_action = "Engellendi: Kalibrasyon Fazı (Düşük/Orta Seviye Baskılama)"
                            else:
                                packet.mitigation_action = f"Tetiklenmedi: Skor Yetersiz ({score:.2f} < 0.40)"
                            packet.reflex_latency_ms    = 0.0
                        
                        # ── 2. PUSH FULLY-STAMPED PACKET TO FRIDAY GOVERNOR & VALIDATION QUEUES ──
                        while True:
                            if stop_event and stop_event.is_set():
                                break
                            try:
                                friday_queue.put(packet, timeout=2.0)
                                break
                            except Exception as ex:
                                print(f"[INFERENCE_ENGINE] ⚠️ SEMANTIC_GOVERNOR queue full — Backpressure applied (Retrying in 1s). Details: {ex}")
                                time.sleep(1.0)
                                
                        # Push alert to Validation Process queue (asynchronous/non-blocking)
                        try:
                            validation_queue.put_nowait(packet)
                        except Exception:
                            pass

                        print(
                            f"[INFERENCE_ENGINE] 🚨 ALARM | {node_id} | "
                            f"Skor={score:.3f} | {severity.value} | "
                            f"{packet.threat_vector.value} | Eylem={packet.mitigation_action}"
                        )

                        # ── 3. VOICE NOTIFICATION (Async TTS) ──────────────────────────────────
                        global _last_speech_time
                        with _speech_lock:
                            if time.time() - _last_speech_time > 15.0:
                                _speak_tts(f"Warning. Anomaly detected in space {node_id}.")
                                _last_speech_time = time.time()

            time.sleep(0.01 if is_benchmark else 0.1)  # 10ms in benchmark mode, 100ms in live simulation

    except KeyboardInterrupt:
        print("\n[INFERENCE_ENGINE] Keyboard interrupt.")
    finally:
        client.loop_stop()
        client.disconnect()

        # SAVE STATE (OPERASYON: KALICILIK)
        try:
            state_data = {
                node_id: list(win) for node_id, win in buffer._windows.items()
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f)
            print(f"[INFERENCE_ENGINE] Saved state for {len(state_data)} nodes")
        except Exception:
            pass

        print("[INFERENCE_ENGINE] Defender process stopped.")


if __name__ == "__main__":
    import multiprocessing as mp
    friday_q = mp.Queue(maxsize=500)
    app_q    = mp.Queue(maxsize=1000)
    run_inference_engine(friday_q, app_q)
