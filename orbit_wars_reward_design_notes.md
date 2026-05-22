# Orbit Wars — Graph-Native Reward Tasarım Notları

Bu doküman, `orbit_wars_graph_metrics_rl.md`'de özetlenen graph-native RL tasarımının
**reward sistemi** ve **policy ağı** boyutunda yapılan son optimizasyonları, kalan
açık konuları ve sıradaki iş için planlanan **action-level credit assignment**
tasarımını kapsar.

> Diğer dokümanlar:
> - `orbit_wars_graph_metrics_rl.md` — graph metrik tanımları ve mantığı
> - `orbit_wars_physics_inspired_metrics.md` — fizik temelli alternatif metrikler

---

## 1. Tamamlanmış İyileştirmeler

### 1.1 Policy Ağı Optimizasyonları (`model_graph_pyg.py`, `model_graph_pyg_edgeonly.py`)

| # | İyileştirme | Mekanizma |
|---|-------------|-----------|
| 1 | ETA grid step `0.5 → 1.0` | Tensor boyutu ½, FLOPs ~%50 ↓ |
| 2 | Batched ETA / node / edge metric helpers | B sample loop kaldırıldı, vectorized |
| 3 | ETA cache | Step başına 1× hesap (önce 2×) |
| 4 | `BilinearEdgeHead` modülü | `MLP(4d → d)` yerine `Linear(d → score)` (FLOPs 4× ↓) |
| 5 | Encoder input `LayerNorm` | Heterojen feature ölçeklerini dengeler |
| 6 | Shared `encoder.edge_proj` | Ayrı `edge_pair_proj` parametresi kaldırıldı |

**Doğrulama**: per-sample vs batched bit-exact eşit (`max diff = 0.0`).
**Test**: 4/4 PyG test geçiyor; 154 PPO/model testi geçiyor.

### 1.2 Reward Sistemi (`rl/rewards.py`, `rl/rollout.py`)

| # | İyileştirme | Etki |
|---|-------------|------|
| 1 | `compute_graph_metrics_summary` helper (Data alloc yok) | Step başına 2× → 1× metric pass |
| 2 | `prev_graph_summaries` cache (rollout) | `Observation` saklamak yerine `dict(node_sum, edge_sum)` |
| 3 | Redundant kanallar (influence_balance, strategic_potential) **default 0** | Triple-counting eliminasyonu |
| 4 | **Potential-based shaping**: `r = γ·φ(s') − φ(s)` | Ng et al. 1999 invariance |
| 5 | `RunningMeanStd` per-player normalization | Heterojen scale farklarını dengeliyor |
| 6 | Edge metric delta channel (`flow_opportunity`) | Yeni directed sinyal |

**Yeni `RewardConfig` alanları**:
```python
graph_potential_discount: float = 0.99
graph_reward_normalize: bool = False
graph_reward_normalize_warmup: int = 64
graph_edge_efficiency_signal: str = "flow_opportunity"
```

**Yeni testler** (`tests/test_graph_native_reward.py` — 11 adet):
- Summary shape & finiteness
- `compute_graph_native_reward` summary dict ve Observation eşit sonuç
- Potential discount formülü (γ=0.5, identical state → r = (γ-1)·φ)
- Edge weight 0 → edge delta etkisiz
- Edge weight aktif → doğru delta
- Redundant channels default 0
- `RunningMeanStd` warmup, variance, normalize

**Toplam test**: 264 passed.

### 1.3 Yan İş: Rollout Uyumluluğu

PygPlanetGraphPolicy ve PygEdgeNativePolicy `policy.planet_dim` (raw 34) ve
`policy.f_max` attribute'lerini expose etmiyordu — graph_native rollout için zorunlu.
Düzeltildi:

```python
self.planet_dim = PLANET_FEATURE_DIM      # 34, build_observation için
self.encoder_planet_dim = PYG_PLANET_FEATURE_DIM  # 42, encoder internal
self.f_max = F_MAX_DEFAULT                # 64
```

End-to-end smoke test: 30 step × 2 agent → 60 transition, NaN/Inf yok, normalization
çalışıyor (`reward range: [-2.02, 1.00]`).

---

## 2. Açık Kalan Konular

Önceliğe göre sıralı:

### 🔴 A. Action-Level Credit Assignment (sıradaki ana iş — bkz. §3)

Step-level graph_native ödülü şu an aynı turn'deki tüm transition'lara **eşit**
dağıtılıyor. NOOP yapan ile A→B launch yapan oyuncu aynı `r`'yi alıyor. Bu en büyük
eksik. Tasarım §3'te.

### 🔴 G. Calibration: Terminal vs Shaping

- Terminal: `±1.0`
- Shaping (smoke test): `mean(|r|) × n_transition ≈ 0.31 × 60 ≈ 18.6` per oyuncu
- **Risk**: shaping cumulative değer terminal sinyalini ezebilir → policy terminal'i
  öğrenemez

**Çözüm önerisi**:
- `reward_scale=0.001` → `0.0001` indir, **veya**
- Terminal'i `±10`'a çıkar, **veya**
- Shaping'i `tanh(r/k)` ile clip et (bkz. §C)

### 🟡 C. Reward Outlier / Clipping

`RunningMeanStd` std ile normalize ediyor ama tek seferlik büyük sıçrama
(büyük neutral planet capture → `friendly_influence` ~100 birim) hâlâ baskın.

**Çözüm**: ±N·std clipping (`r = clip(r, -3·std, +3·std)`) eklenebilir.
PPO'nun advantage normalize'i bunu kısmen çözse de step-level outlier ağırlık
güncellemesinde noise yaratıyor.

### 🟡 B. Loss Aversion / Asimetri

Graph-native'de `friendly_influence_weight = 1.0` ve `enemy_influence_weight = -1.0`
**simetrik**. `current` preset'inde `denial_value_weight ≠ capture_value_weight`
ile loss aversion var, graph-native'de yok.

**Etkisi**: Policy gezegen kaybetmeyi önleme önceliğini öğrenemiyor olabilir.

### 🟡 H. Invalid Action / Beta Overflow Bypass

`graph_native` preset'te `invalid_action_penalty = 0.0` ve `ship_overflow_weight = 0.0`.
`current` preset'te ikisi de aktif. Policy gereksiz invalid action / aşırı ship
request yapma cezası almıyor.

**Çözüm**: graph_native preset'ine ekle (`invalid_action_penalty = 0.01`,
`ship_overflow_weight = 0.5`).

### 🟢 F. Multi-Edge Signal

`graph_edge_efficiency_signal` tek seçim — `"flow_opportunity"` aktif.
Birden fazla edge channel paralel kullanılamıyor (örn.
`flow_opportunity + 0.5·same_owner_expansion`).

**Çözüm**: `Dict[str, float]` desteği ekle.

### 🟢 E. Channel Curriculum / Annealing

Tüm graph channel weight'leri sabit. Training phase'ine göre dinamik weight yok:
- **Early**: `production_potential` öncelikli (ekonomi kurma)
- **Mid**: `vulnerability`, `supportability` (tactical pozisyonlama)
- **Late**: `friendly - enemy` (sonu götürme)

Linear/cosine schedule ile çözülebilir, ama büyük iş.

### 🟢 D. N-step Potential Shaping

Mevcut: `r = γ·φ(s_{t+1}) − φ(s_t)` (1-step delta).
Edge metrikleri (özellikle `flow_opportunity`) volatil — büyük fleet hareketleri ile
sıçrayıp düşüyor. N-step `r = γⁿ·φ(s_{t+n}) − φ(s_t)` daha az gürültülü olabilir.

**Karmaşıklık**: rollout buffer'a "delayed potential window" eklemek gerek.

### 🟢 J. Legacy Field Cleanup

Hâlâ kalan deprecated alanlar:
- `eligibility_trace_horizon`, `eligibility_trace_decay`, `eligibility_trace_reverse`
- `production_reinforcement_weight`, `production_reinforcement_decay_window`
- `noop_trace_weight`

Sadece dökümantasyon/karmaşıklık temizliği — fonksiyonel etki yok.

### 🟢 K. Pre-existing Rollout Bugs

- `PygEdgeNativePolicy.f_max` artık var ama `policy.act` ve `policy.act_log_prob` ile
  rollout entegrasyonu **derinden test edilmedi**.
- Smoke test'te `PygPlanetGraphPolicy` çalıştı ama gerçek seçilmesi gerekenleri
  doğrulamadık (action distribution sanity).

### 🟢 I. Test Coverage Eksikleri

Mevcut yeni testler reward formülünü doğruluyor, ama:
- **Action attribution invariant** yok: graph_native'de fleet capture aynı
  transition'a credit alıyor mu?
- **Cum reward over episode** benchmarkı yok: shaping toplam terminal'i ezmiyor mu?
- **Distinguishability test** yok: NOOP ile good action arasında reward farkı olmalı

---

## 3. Sıradaki İş: Per-Node Action-Level Credit (Tasarım Tartışması)

### 3.1 Çekirdek Fikir

> "Her aksiyon, hangi node'ların metriklerinde değişikliğe sebep olduysa oradan
> ödül alır. Launch `x → y` yapıldıysa, x ve y gezegenindeki metrik değişimlerini
> bu aksiyona yansıt."

Formüller:
```
r(launch(x → y)) = w_src_node · Δφ_node[x]
                 + w_tgt_node · Δφ_node[y]
                 + w_edge      · Δφ_edge[x, y]

Δφ_node[i] = γ · curr_node_metrics[i] − prev_node_metrics[i]   (per-planet, weighted)
Δφ_edge[i,j] = γ · curr_edge_metric[i,j] − prev_edge_metric[i,j]
```

### 3.2 Avantajlar

1. **Local credit** — global broadcast yok
2. **NOOP ile launch ayrışır** (NOOP=0, launch'lar ≠ 0)
3. **FleetTracker ile temiz separation**:
   - Graph delta = anlık pozisyon kalitesi
   - FleetTracker = uzun vadeli capture/kill outcome
4. **Multi-launch in same turn** doğal destekleniyor
5. **Source over-commit otomatik cezalanır** (x'ten gemi gönderince x'in
   `vulnerability` artar → negatif source-credit)

### 3.3 İncelikler & Karar Noktaları

#### İncelik 1: Residual Reward (Atfedilmemiş Kısım)

Her turn metrikler **opponent action + production + fleet movement** nedeniyle de
değişir. Bu kısım nereye gider?

| Seçenek | Davranış | Karar |
|---------|----------|-------|
| **A) Drop** | Sadece launch'a atfedilen credit verilir | ✅ Önerilen |
| B) NOOP'a ver | Residual → NOOP transition'larına | Belirsiz multi-NOOP |
| C) Tüm transition'lara eşit dağıt | Mevcut step-shaping davranışı | Bilgi kaybı yok ama zayıf attribution |

**Karar gerekçesi**: Critic zaten residual'ı value function olarak öğrenir.
Shaping aksiyon-spesifik sinyal vermek için var. NOOP=0 sorun olursa B'ye geçeriz.

#### İncelik 2: Multi-Launch → Same Target

İki launch aynı turn'de `y`'yi hedeflerse `Δφ_node[y]` nasıl bölünür?

| Seçenek | Karar |
|---------|-------|
| **Eşit böl** (`Δφ[y] / k`) | ✅ Önerilen |
| Ship-weighted | Daha doğru ama karmaşık |
| Full credit (overcount) | ❌ Yanlış, double-counts |

> Env kuralı: bir oyuncu turn başına farklı `src` kullanmak zorunda → source-side
> bölme problemi yok, sadece target-side var.

#### İncelik 3: Non-Local Effects

Launch `x→y` aslında **z**'nin metriklerini de etkiler (kernel `exp(-eta/τ)`). Ama
τ=10 turn → en büyük etki kesinlikle x ve y'de, non-local etki tipik <%20.

**Karar**: v1'de görmezden gel.

#### İncelik 4: Source vs Target Asimetri

Source (`x`) tarafında `vulnerability` artıyor → launch negatif source credit alır.
Doğru ama aşırı agresif olursa policy hiç launch yapmaz.

**Karar**: Ayrı weight'ler:
```python
graph_src_node_weight: float = 0.5
graph_tgt_node_weight: float = 1.0
```

#### İncelik 5: Multi-Player Same Target

İki **farklı oyuncu** aynı turn'de `y`'ye launch atarsa:
- Her oyuncu kendi metrik uzayında (kendi gemiler = friendly)
- Δφ matrisleri **farklı** (asymmetric ownership)
- **Bölme sorunu yok**: her oyuncu kendi metric uzayında credit'liyor

### 3.4 Önerilen Final Tasarım (Pseudocode)

```python
def credit(action, prev_metrics, curr_metrics, *, cfg, k_targets_y):
    """Per-action reward."""
    γ = cfg.graph_potential_discount

    if action.is_noop:
        return 0.0  # residual atılır (mod A)

    x, y = action.src, action.tgt
    delta_node = γ * curr_metrics.node - prev_metrics.node     # (P, 8)
    per_planet = (delta_node * node_weights).sum(-1)           # (P,)

    # Multi-launch target sharing
    r_node = (cfg.graph_src_node_weight * per_planet[x]
              + cfg.graph_tgt_node_weight * per_planet[y] / max(1, k_targets_y))

    delta_edge = γ * curr_metrics.edge[x, y] - prev_metrics.edge[x, y]
    r_edge = cfg.graph_edge_efficiency_weight * delta_edge

    return (r_node + r_edge) * time_factor * cfg.reward_scale
```

### 3.5 Implementation Plan

| Adım | Açıklama | LOC tahmini |
|------|----------|-------------|
| 1 | `compute_graph_metrics_summary` → full matrix döndür (sum yerine) | ~20 |
| 2 | Yeni `compute_action_credited_rewards(prev, curr, actions, cfg, turn) -> list[float]` | ~80 |
| 3 | `RewardConfig` yeni alanlar (`graph_src_node_weight`, `graph_tgt_node_weight`, `graph_action_attribution: bool`) | ~10 |
| 4 | `rollout.py`: her transition için kendi action'ını eşle, attributed reward set et | ~60 |
| 5 | Multi-launch target counting helper | ~15 |
| 6 | Backward compat: `cfg.graph_action_attribution = False` ise eski global mod | ~10 |
| 7 | Unit testler | ~120 |
| **Toplam** | | **~315 LOC** |

### 3.6 Karar Verilmesi Gerekenler (kodlamadan önce)

1. ✅ **Residual handling**: A (drop)
2. ✅ **Multi-launch target**: eşit böl
3. ✅ **Source/target asimetri**: ayrı weight (default `0.5` / `1.0`)
4. ⏳ **Backward compat flag**: `cfg.graph_action_attribution: bool = True` mu?
5. ✅ **N-step / 1-step**: 1-step (v1)

### 3.7 Riskler

- **Source-side over-penalty**: w_src çok yüksek olursa policy launch yapmaz.
  Calibration gerek (`w_src=0.5` başlangıç).
- **Multi-launch attribution**: aynı turn'de >2 launch nadir, ama happy path'i
  test etmemiz şart.
- **Launch kuyrukta gecikirse**: Action turn t'de yapıldı ama metrik delta turn
  t+1'de görünüyor → buffer sıralamasında dikkat.
- **NOOP=0 dezavantaj**: Pasif savunma rolündeki oyuncu hiç sinyal almaz.
  Yan etki olabilir; eğer ortaya çıkarsa residual mode B'ye geçmek için flag
  hazır olmalı.

---

## 4. Sıradaki Adımlar

1. **Karar 4** (backward compat flag) onaylandıktan sonra §3.5 implementation planına başla
2. Action-level credit eklendikten sonra §2'deki **C (clipping)** ve **G (calibration)**
   küçük PR'lar halinde gelir
3. **B (asymmetry)** ve **H (invalid action penalty)** isteğe bağlı tuning olarak ek
4. **I (test coverage)** action-level credit ile birlikte gelmeli — yeni invariant'lar
   eklemek için doğal yer

---

## 5. Referans Komutlar

```bash
# Tüm test suite (264 passed)
orbit_wars_venv/bin/python -m pytest tests/ -q

# Sadece graph_native reward testleri (11 passed)
orbit_wars_venv/bin/python -m pytest tests/test_graph_native_reward.py -v

# Smoke: end-to-end graph_native rollout
orbit_wars_venv/bin/python - <<'PY'
import sys; sys.path.insert(0, '.')
from rl.rollout import collect_episode
from rl.buffer import RolloutBuffer
from rl.rewards import build_reward_config
from model_graph_pyg import PygPlanetGraphPolicy

policy = PygPlanetGraphPolicy(p_max=44, d_model=32, n_layers=2)
buf = RolloutBuffer()
cfg = build_reward_config("graph_native")
collect_episode(policy, buf, max_steps=30, num_agents=2, reward_cfg=cfg, seed=42)
print(f"transitions: {len(buf.data)}")
PY
```
