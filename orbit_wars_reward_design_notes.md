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
self.encoder_planet_dim = PYG_PLANET_FEATURE_DIM  # 43, encoder internal
self.f_max = F_MAX_DEFAULT                # 64
```

End-to-end smoke test: 30 step × 2 agent → 60 transition, NaN/Inf yok, normalization
çalışıyor (`reward range: [-2.02, 1.00]`).

---

## 2. Açık Kalan Konular

Önceliğe göre sıralı:

### ✅ A. Action-Level Credit Assignment — counterfactual arrival credit ile çözüldü

Eski sorun: step-level graph_native ödülü aynı turn'deki tüm transition'lara
**eşit** dağıtılıyordu; NOOP ile A→B launch aynı `r`'yi alıyordu.

Güncel çözüm: launch transition'ına **counterfactual arrival credit** delayed
reward olarak yazılıyor (`actual_arrival - no_launch_counterfactual`). Detay
ve formüller `docs/arrival_credit_design.md` ve aşağıdaki §3 özetinde.

Step-level potential shaping artık arrival credit aktifken otomatik kapatılıyor
(`enable_arrival_credit=True` ⇒ step `r=0`), yani çift kredi sorunu kalmadı.
Doğrulayan regression testleri: `tests/test_arrival_credit.py::test_rollout_arrival_credit_disables_step_shaping`
ve `..._off_keeps_step_shaping`.

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

## 3. Güncel Reward Sistemi: Counterfactual Arrival Credit (Özet)

> **Detaylı tasarım, formüller, CLI bayrakları, double-counting analizi ve
> AI handoff özeti** için tek kaynak doğruluğu (single source of truth):
> [`docs/arrival_credit_design.md`](docs/arrival_credit_design.md).
>
> Bu bölüm yalnızca o dokümandaki kararların kısa Türkçe özetidir. Bir
> uyumsuzluk olursa `docs/arrival_credit_design.md` esas alınır.

### Çekirdek karar

```
No-launch direct reward = 0

Launch delayed reward =
    metrics(actual_arrival_branch)
  - metrics(no_launch_counterfactual_branch)
```

Kötü launch'lar farktan negatif credit alır → ayrı NOOP bonusuna gerek yok.

### Aktif katmanlar

Step shaping ve arrival credit **mutually exclusive** — `enable_arrival_credit=True`
iken step-level shaping otomatik kapanır (`rl/rollout.py` `graph_native` branch'inde
`shaping_active = not cfg.enable_arrival_credit` guard'ı). Aynı metric swing
iki kez kredi olarak gradient'e girmez.

1. **Counterfactual arrival credit** *(yeni default; arrival credit on)* —
   fleet vardığında launch transition'ına `buffer.add_delayed_reward` ile
   `<channel_weights, M_actual − M_no_launch>` yazılır. Step reward = 0.
2. **Graph-native potential shaping** *(legacy; arrival credit off)* —
   `r_t = γ φ(s_{t+1}) − φ(s_t)` her adıma yazılır.
3. **Flux metrics** — her iki modda da geçerli; arrival credit modunda
   counterfactual'ın 13-D metric vektörünün son 4 kanalı olarak yaşar.

Terminal reward (±1) ve `invalid_action_penalty` her iki modda etkilenmez.

### Kritik kurallar (kısa)

- Counterfactual'da `tgt.ships -= arriving`, `src.ships += arriving`.
- `tgt.ships < 0` → `tgt.owner = non_self`, `ships = abs(...)`. Negatif ship
  count metric input'a bırakılmaz; `friendly_*` kaynakları doğal kapanır.
- `arrival_credit_counterfactual_horizon > 0` ise etkilenen self-owned target/source
  için horizon içindeki **soonest enemy arrival** resolve edilir
  (kurtarma değerlerini ölçmek için).
- Production kanalları için `friendly_production_potential` kullanılır;
  ham `production_potential` ownership-blind.

### Önerilen başlangıç ayarı

```bash
--reward-system graph_native \
--enable-flux-metrics \
--enable-arrival-credit \
--arrival-credit-counterfactual \
--arrival-credit-counterfactual-scope full_graph \
--arrival-credit-counterfactual-horizon 5 \
--arrival-credit-mode exp_decay \
--arrival-credit-tau 10.0 \
--arrival-credit-weight 1.0
```

`arrival_credit_weight` artık `1.0` ile başlanabilir — step shaping kapalı
olduğu için çift kredi yok. PPO `valL` variance fazla olursa `0.5`'e düşür.

### Kod karşılığı

| Dosya | Rol |
|---|---|
| `rl/rewards.py` | `RewardConfig` graph weights, counterfactual flags |
| `rl/arrival_credit.py` | `ArrivalCreditTracker`, counterfactual obs, credit hesabı |
| `rl/rollout.py` | launch snapshot, arrival'da actual/counterfactual metric matrix |
| `rl/flux.py` | transit fleet flux metrics |
| `tests/test_arrival_credit.py` | 41 test — unit + integration smoke + shaping-off regression |

---

## 4. Sıradaki Adımlar

1. Counterfactual arrival-credit'i training'de `arrival_credit_weight=1.0`
   ile dene (step shaping kapalı; çift kredi yok).
2. `credits_emitted` / `credits_total` diagnostic counters'ı training log'a ekle.
3. Terminal vs delayed credit kalibrasyonu: counterfactual credit'in
   cumulative magnitude'u terminal sinyali (±1) ezerse `arrival_credit_weight`
   ya da `reward_scale` indir.
4. No-launch özel reward ekleme; önce kötü launch penalty sinyalinin yeterli
   olup olmadığını gözle.
5. Gerekirse sadece threatened-source defensive hold credit tasarla.

---

## 5. Referans Komutlar

```bash
# Tüm test suite (332 passed)
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
