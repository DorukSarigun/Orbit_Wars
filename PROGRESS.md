# Orbit Wars — Proje Durumu

Bu doküman, [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
yarışması için geliştirilen ajan altyapısının şu ana kadarki durumunu özetler.

> **Hedef:** Kazanan bir RL agent. Yaklaşımımız: ağır geometrik/strateji
> hesaplarını deterministik kodla çözmek, modeli yalnızca yüksek seviyeli
> kararla (kimden kime ne kadar gemi?) ilgilenmeye zorlamak.

---

## 1. Mimari Genel Bakış

```
                ┌───────────────────────────────────────────────┐
                │              Kaggle env observation           │
                │  planets / fleets / comets / angular_velocity │
                └────────────────────────┬──────────────────────┘
                                         │
                            ┌────────────▼────────────┐
                            │   feature_extraction    │  raw rows -> Planet / Fleet
                            │   (polar koordinatlar,  │  (sun-centered, type flags)
                            │    type sınıflandırması)│
                            └────────────┬────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
       ┌─────────────┐           ┌──────────────┐           ┌──────────────┐
       │ navigation  │           │  playground  │           │ obs_builder  │
       │             │           │              │           │              │
       │ intercept,  │           │ matplotlib   │           │ padded numpy │
       │ collision,  │           │ render +     │           │ tensors,     │
       │ launch,     │           │ interactive  │           │ masks,       │
       │ reachable,  │           │ launch UI    │           │ edge bias    │
       │ predict     │           │              │           │              │
       └──────┬──────┘           └──────────────┘           └──────┬───────┘
              │                                                    │
              │                                                    ▼
              │                                           ┌────────────────┐
              │                                           │     model      │
              │                                           │ Set Transformer│
              │                                           │  + masked      │
              │                                           │  pointer policy│
              │                                           └────────┬───────┘
              │                                                    │
              └────────────────────────┬───────────────────────────┘
                                       ▼
                              env action [from_id, angle, ships]
                              (compute_launch açıyı çözer,
                               agent yalnızca src/tgt/ship_bin seçer)
```

---

## 2. Modüller

### 2.1 `feature_extraction.py` — Veri sınıfları
[`@/home/doruk/Documents/Kaggle/Orbit_Wars/feature_extraction.py`]

- **`Planet` dataclass**
  - Ham `[id, owner, x, y, radius, ships, production]` satırını zenginleştirir.
  - **Polar koordinatlar** (`r_sun`, `theta`) güneş merkezli; oyunun rotasyonel
    simetrisini yakalar, RL feature'ları için doğal.
  - Tip sınıflandırması: `is_static`, `is_rotating`, `is_comet` (env'in oyun
    başında karar verdiği yerleşim mantığını birebir yansıtır).
  - `position_at(steps_ahead, omega)` — gelecekteki konumu öngörür (rotating
    için açısal interpolasyon, comet için path interpolation).
  - `ships_at(steps_ahead)` — saldırı planlaması için kaba ship projection.
  - `is_mine`, `is_enemy`, `is_neutral` ownership helper'ları.

- **`Fleet` dataclass**
  - Ham `[id, owner, x, y, angle, from_planet_id, ships]` → uçuşta filo nesnesi.
  - `speed = fleet_speed(ships)` — env'in formülünü birebir kopyalar:
    `v(s) = 1 + (v_max − 1) · (ln s / ln 1000)^{1.5}`.
  - `position_at(t)` — doğrusal kinematik; intercept ve hedef tahmini için.

- **`parse_planets(obs)`**, **`parse_fleets(obs)`** — env çıktısını listeye
  çeviren convenience helper'lar.

### 2.2 `navigation.py` — Geometri & launch çözücüleri
[`@/home/doruk/Documents/Kaggle/Orbit_Wars/navigation.py`]

Bu modül "agent açıyı düşünmek zorunda kalmasın" felsefesiyle yazıldı.

| Fonksiyon | İşi |
| --- | --- |
| `segment_min_dist_to_point` | Doğru parçası ↔ nokta minimum mesafe. |
| `crosses_sun` | Filo segmenti güneşten geçer mi? (margin ile). |
| `find_intercepts` | **Lead-shot intercept çözücü.** Statik hedef için kapalı form, rotating/comet için coarse scan + bisection (birden çok "lap" kökü destekler). |
| `path_collision` | Filo doğrusal yolunda başka gezegene çarpıyor mu? Env'in tur sırasını birebir yansıtır: (a) filo hareketi vs gezegen başlangıç pozisyonu, (b) gezegen sweep'i vs filo bitiş pozisyonu. Bu, debugging sürecinde ortaya çıkan kritik bir bug fix'tir. |
| `compute_launch` | **Üst seviye API.** `(src, tgt, ships)` üzerinden geçerli `(angle, eta, intercept)` üretir. Sun ve gezegen çarpışmalarını eler, en erken çözümü döner. |
| `compute_launch_at` | Gecikmeli fırlatma varyantı: "k tur sonra fırlatırsam?" |
| `earliest_reachable` | Belirli `(src, tgt)` için fizibıl olduğu **en erken k**'yı bulur — agent'ın "şu an gönderemiyorum, kaç tur beklerim?" sorusuna cevap. |
| `reachability_scan` | Çoklu hedef üzerinde `earliest_reachable` koşar, total_time'a göre sıralar. "Ne saldırayım?" oracle'ı. |
| `predict_fleet_target` | **Filonun nereye varacağını çıkarır.** Env observation'ında açıkça hedef ID'si yok; deterministik ileri simülasyon ile çözeriz. |
| `predict_all_fleet_targets` | Toplu tahmin. |
| `incoming_threats` | Düşman filolarının kendi gezegenlerime varış tahminleri, ETA'ya göre sıralı; savunma planlamasının çekirdeği. |

### 2.3 `playground.py` — Görselleştirme & interaktif kontrol
[`@/home/doruk/Documents/Kaggle/Orbit_Wars/playground.py`]

- `Playground` sınıfı: env'i sarar, `step`, `launch(src_id, tgt_id, ships)`,
  status, history yönetir.
- `render_board(obs, ...)` matplotlib renderer:
  - Güneş + güvenlik halkası, gezegen yörüngeleri (rotating için), comet path.
  - Filo ikonları (sahibe göre renkli üçgen) + hareket vektörü oku.
  - Launch tarihi: kaynak, intercept noktası, ETA.
  - Tahmini fleet target'ları (kesikli çizgi + halka), gelen tehditler.
- Tablo helper'ları: `planet_table`, `fleet_table`, `reach_table`,
  `fleet_targets`, `threats`.

### 2.4 `obs_builder.py` — RL observation pipeline
[`@/home/doruk/Documents/Kaggle/Orbit_Wars/obs_builder.py`]

Ham env observation'ını **sabit şekilli numpy tensörlerine** çevirir.
RL ağı için tasarlandı; permutation invariant.

- **Çıktı: `Observation` dataclass**
  - `planet_feats: (P_max, 15)`
  - `planet_mask: (P_max,)`, `source_mask: (P_max,)` (kim launch yapabilir)
  - `fleet_feats: (F_max, 11)`, `fleet_mask: (F_max,)`
  - `global_feats: (12,)` — CLS-style global istatistik
  - `edge_bias: (T, T, 5)` — pairwise (dist, dx, dy, sin Δφ, cos Δφ); son iki kanal güneşe göre polar açı farkı ile rotasyondan bağımsız orbital ilişki sinyali
  - `planet_ids`, `fleet_ids` — env id eşlemesi (action decoding için)

- **Planet özellikleri (15-d):** ownership one-hot, ships/prod/radius normalize,
  polar (`r_sun`, `cos/sin θ`), tip flag'leri, ω·dt, log_ships,
  ships-to-capture.
- **Fleet özellikleri (11-d):** ownership, ships, log_ships, hız, heading
  (cos/sin), pozisyon, **`predict_fleet_target` ETA prior'u**.
- **Global (12-d):** turn, ω, planet/fleet sayıları, ship/prod toplamları,
  incoming threat ölçütleri.

- **`decode_action(...)`** — model'in seçtiği `(src_slot, tgt_slot, ship_bin)`
  üçlüsünü env aksiyonuna çevirir; açıyı `compute_launch` deterministik bulur.

### 2.5 `model.py` — Set Transformer policy
[`@/home/doruk/Documents/Kaggle/Orbit_Wars/model.py`]

#### Mimari kararı: neden Set Transformer?
- **Permutation invariance** otomatik (positional encoding kullanmıyoruz).
- **Edge bias** sayesinde mesafe/yön bilgisini sıfırdan öğrenmek zorunda değil.
- **Heterojen tipler** (planet vs fleet vs global) tek diziye, type embedding ile.
- PyG/DGL bağımlılığı yok; sadece `torch`.

#### Token akışı

```
planet feats (P, 15) ─ planet_proj ─┐
fleet  feats (F, 11) ─ fleet_proj  ─┼─ concat ─→ (B, T, d_model) ─ + type_embed
global feats   (12,) ─ global_proj ─┘
                                          │
                                          ▼
              n_layers × TransformerBlock with BiasedMHA:
                  scores = QK^T / √d  +  edge_proj(edge_bias)
                  (per-head, padded keys -inf maskelenir)
                                          │
                                          ▼
                            LayerNorm → tokens (B, T, d_model)
              ┌────────────┬─────────────┬────────────┬────────────┐
              ▼            ▼             ▼            ▼            ▼
          src_head    tgt_query/key   ship_head   value_head   (NO-OP param)
        (B, P) + 1   (B, P, P)        (B, P, k)   (B,)        (1,)
```

#### Aksiyon faktorizasyonu

```
src_slot ~ Categorical(src_logits)              # P_max + 1 (son slot = NO-OP)
tgt_slot ~ Categorical(tgt_logits[src_slot])    # self-target ve padded maskeli
ship_bin ~ Categorical(ship_logits[src_slot])   # 5 ayrık fraksiyon (10/25/50/75/100%)
```

**Açıyı agent SEÇMİYOR** — `compute_launch(src, tgt, ships)` sun ve çarpışma
kontrollü olarak çözüyor. Bu, action space'i continuous angle yerine discrete
pointer'a indirir; sample efficiency'yi büyük ölçüde artırır.

#### Maskeleme garantileri (testlerle doğrulanmış)
- Padded gezegenlere src/tgt olasılığı sızmaz.
- Yasadışı kaynak (rakip / boş gezegen) softmax öncesi `-inf` ile elenir.
- Self-target (`src == tgt`) maskeli.
- NaN-safe: tamamen maskeli row'lar `nan_to_num` ile sıfırlanır.

### 2.6 Notebook'lar
- [`@/home/doruk/Documents/Kaggle/Orbit_Wars/orbit-wars-data/notebooks/notebooks/launch_playground.ipynb`]
  — Manuel oynat: `Playground` ile launch yap, render et, debug et.
- [`@/home/doruk/Documents/Kaggle/Orbit_Wars/orbit-wars-data/notebooks/notebooks/orbit_wars.strateji.ipynb`]
  — Planet class & navigation API'si üzerine deneyler.
- [`@/home/doruk/Documents/Kaggle/Orbit_Wars/orbit-wars-data/notebooks/notebooks/rl_observation_and_model.ipynb`]
  — RL pipeline'ının uçtan uca açıklaması (feature tabloları, edge bias
  heatmap'leri, mask sanity check'leri, forward pass shape'leri, full rollout).

### 2.7 Dokümantasyon
- [`@/home/doruk/Documents/Kaggle/Orbit_Wars/docs/math.md`] — oyunun matematiksel
  modeli (intercept denklemi, fleet speed, sun avoidance, path collision)
  + 5 üretilmiş figür.
- [`@/home/doruk/Documents/Kaggle/Orbit_Wars/TODO.md`] — sıradaki işler.

---

## 3. Test Kapsamı

[`@/home/doruk/Documents/Kaggle/Orbit_Wars/tests/`]

| Dosya | Test sayısı | Kapsam |
| --- | --- | --- |
| `test_navigation.py` | 51 | Geometri helper'ları, intercept solver, path collision (rotating + sweep), compute_launch, gecikmeli launch, earliest_reachable, fleet target prediction, threats, reported bug regresyonları. |
| `test_obs_builder.py` | 12 | Tensor şekilleri, dtype'lar, mask semantiği, owner one-hot, edge bias geometrisi, threat aggregation, action decoding. |
| `test_model.py` | 8 | Forward shape'leri, NaN-safe logits, padded/yasadışı slot maskeleme, self-target maskeleme, action sampling, deterministic argmax, gradient flow. |
| **Toplam** | **71** | Tamamı `< 0.5s`'de geçer. |

```bash
pytest tests/ -v
```

---

## 4. Kritik Tasarım Kararları

1. **Polar koordinatlar.** Güneş merkezli `(r, θ)` ham `(x, y)`'e tercih edildi;
   oyunun rotasyonel simetrisini açığa çıkarır, RL feature uzayında öğrenmesi
   daha kolay.

2. **Açıyı agent değil compute_launch seçer.** Aksiyon uzayı discrete
   `(src, tgt, ship_bin)` üçlüsüne indirgendi. Sample efficiency için kritik
   karar.

3. **Deterministik prior'lar feature olarak.** `predict_fleet_target` ve
   `earliest_reachable` çıktıları observation'a giriyor. Domain knowledge
   gradient descent'e bedava şortcut.

4. **Set Transformer + edge bias > GNN.** Permutation invariance bedava
   geliyor; mesafe bilgisi attention bias olarak veriliyor; PyG bağımlılığı
   yok; tek `torch.nn` ile maintainable.

5. **NO-OP slot.** Bazı turlarda hiç fırlatmamak en iyi seçim olabilir
   (gemi biriktir, üretim arttır). Policy'ye bunu ifade etmesi için ekstra
   slot verildi, böylece "yapay launch" zorunluluğu yok.

6. **Path collision env tur sırasıyla.** Bug fix sonrasında: env önce filoyu
   hareket ettirir SONRA gezegenleri döndürür. Çarpışma testi her iki adımı
   da kontrol eder (filo segmenti vs gezegen başlangıç + gezegen sweep vs filo
   bitiş). Daha önce çakılan yörüngeler artık doğru tespit ediliyor.

---

## 5. Repo İskeleti

```
Orbit_Wars/
├── feature_extraction.py        # Planet / Fleet veri sınıfları
├── navigation.py                # intercept, collision, launch, prediction
├── playground.py                # matplotlib render + interactive controller
├── obs_builder.py               # RL observation pipeline (numpy tensorler)
├── model.py                     # Set Transformer policy + value
├── tests/
│   ├── test_navigation.py       # 51 test
│   ├── test_obs_builder.py      # 12 test
│   └── test_model.py            # 8 test
├── docs/
│   ├── math.md                  # oyunun matematiksel modeli
│   └── generate_figures.py      # 5 figür üreteci
├── orbit-wars-data/notebooks/notebooks/
│   ├── launch_playground.ipynb  # manuel oyna
│   ├── orbit_wars.strateji.ipynb
│   └── rl_observation_and_model.ipynb  # RL pipeline tour
├── TODO.md
├── PROGRESS.md                  # bu dosya
└── play.ipynb                   # serbest scratch notebook
```

---

## 6. Sıradaki Adımlar

Detayı `TODO.md` içinde; özetle:

### Yakın vade (RL eğitim ayağı)
- [ ] **`train_ppo.py`** — vectorized env wrapper, GAE, entropy bonus,
      learning rate schedule, gradient clipping.
- [ ] **Reward shaping** — kazanma sinyali çok seyrek; per-step
      `Δ(my_ships + my_prod·planets) − Δ(opp_total)` shaping.
- [ ] **Self-play opponent pool** — baseline `nearest sniper` + son N checkpoint.
- [ ] **Sanity rollout metrikleri** — NO-OP oranı, başarısız `compute_launch`
      oranı, action latency, win-rate vs baseline.

### Strateji geliştirmeleri
- [ ] `min_ships_to_capture` (binary search).
- [ ] Kombineli saldırı senkronizasyonu (iki kaynaktan aynı turda varış).
- [ ] Defense module: garrison ihtiyacı hesabı `incoming − my_ships − production`.
- [ ] Tangent-around-sun waypoint planlaması.

### Profiling & operasyonel
- [ ] `reachability_scan` cache (1 turn'da birden çok kez çağrı).
- [ ] Submission paketi: `main.py` + minimal bağımlılık.
- [ ] Kaggle replay/log pipeline (debug için).

---

## 7. Hızlı Başlangıç

```bash
# Bağımlılıklar
pip install -e ./orbit-wars-data/kaggle-environments
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy matplotlib pandas pytest

# Testleri koştur
pytest tests/ -v

# Notebook'u aç
jupyter notebook orbit-wars-data/notebooks/notebooks/rl_observation_and_model.ipynb
```

```python
# Programatik kullanım
from kaggle_environments import make
from obs_builder import build_observation
from model import OrbitWarsPolicy, pack_observations

env = make('orbit_wars', debug=False)
env.reset(2)
env.step([[], []])  # world generation

obs0 = env.state[0].observation
omega = obs0.get('angular_velocity', 0.0)
o = build_observation(obs0, player=0, angular_velocity=omega)

net = OrbitWarsPolicy()
out = net(**pack_observations([o]))
action = net.act(out, deterministic=False)
```
