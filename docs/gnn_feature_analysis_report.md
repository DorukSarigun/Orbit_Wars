# Orbit Wars GNN — Derinlemesine Mimari ve Feature Analizi Raporu

> Kapsam: `pyg_planet_graph` modelinin (`model_graph_pyg.py`) encoder/policy mimarisi,
> node ve edge feature pipeline'ları, GINEConv mesaj geçişinin sonuçları, tespit edilen
> zayıflıklar ve **somut iyileştirme önerileri**.
>
> Kaynak dosyalar: `model_graph_pyg.py`, `obs_builder.py`, `rl/flux.py`, `rl/held_production.py`.
> İlgili referans notebook: `docs/node_metrics_complete_reference.ipynb`.

---

## 1. Mimari özeti

### 1.1 Pipeline ve boyutlar

```
env_obs
  └─ obs_builder.build_observation
        planet_feats (P_max=44, 34)   global_feats (14)
        edge_bias (T, T, 6)           fleet routing arrays
  └─ model_graph_pyg (deterministik, grad yok)
        _estimated_intercept_eta_matrix   → ETA (n,n)
        _graph_node_metrics               → 10 (+4 flux) kanal
        _graph_edge_metric_matrix         → 9 (+2 flux) kanal
  └─ observation_to_pyg_data
        x         = concat(planet_feats[34], node_metrics[10]) = 44   (flux: 48)
        edge_attr = concat(edge_bias[6], edge_metrics[9])      = 15   (flux: 17)
        edge_index = tam bağlı yönlü graf  (n·(n−1) kenar)
        global    = (1, 14)
  └─ PygPlanetGraphEncoder  (GINEConv × 3, d_model=128)
        node_tokens (n, 128)   graph_summary (1, 128)
  └─ PygPlanetGraphPolicy heads
        src ~ Categorical    tgt ~ Categorical(src)
        ships ~ Beta(α,β)    value ~ scalar    aux_owner ~ 5-way
```

| Katman | Boyut | Not |
|--------|-------|-----|
| Node input | **44** (= 34 + 10) | flux açıksa 48 |
| Edge input | **15** (= 6 + 9) | flux açıksa 17 |
| Global input | **14** | mean/max pool ile birleşir |
| `d_model` | **128** | |
| `n_layers` | **3** GINEConv | |
| Graf yapısı | **tam bağlı (FC)** yönlü | `n·(n−1)` kenar |

### 1.2 Encoder iç yapısı (`PygPlanetGraphEncoder`)

```
node_proj   : LayerNorm(44) → Linear(44→128) → GELU → LayerNorm      (use_input_embed=True)
edge_proj   : LayerNorm(15) → Linear(15→128) → GELU → LayerNorm
global_proj : LayerNorm(14) → Linear(14→128) → GELU → LayerNorm

for i in 0..2:                       # 3 katman
    msg = GINEConv(MLP)(x, edge_index, edge_attr)   # MLP: Linear→GELU→Linear
    x   = LayerNorm(x + dropout(msg))                # residual + norm

mean_pool = global_mean_pool(x)
max_pool  = global_max_pool(x)
graph_summary = Linear(3·128 → 128)( cat[mean_pool, max_pool, global_tok] )
```

**GINEConv mesaj kuralı** (PyG):

\[
x_i' = \mathrm{MLP}\Big( (1+\epsilon)\,x_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}(x_j + e_{ij}) \Big)
\]

- Komşuluk **tam bağlı** olduğu için her düğüm diğer **tüm** düğümlerden mesaj alır.
- Toplama (`sum`) agregasyonu kullanılır → düğüm sayısıyla ölçek büyür (aşağıda kritik nokta).
- `edge_proj` her durumda `d_model`'e projekte eder çünkü aksiyon head'leri `d_model` genişliğinde edge token'ı üzerinde skorlar.

### 1.3 Policy head'leri

| Head | Girdi | Çıktı | Mimari |
|------|-------|-------|--------|
| `src_head` | `[node_tok, graph_summary]` (2·128) | kaynak logit | MLP 256→128→1 |
| `noop_head` | aynı | no-op logit | MLP 256→128→1 |
| `pair_score_head` | src,tgt,edge,global token | hedef logit | `BilinearEdgeHead` |
| `ship_head` | aynı | Beta (α,β) | `BilinearEdgeHead` out=2 |
| `value_head` | `graph_summary` | V(s) | MLP 128→128→1 |
| `aux_owner_head` | `node_tok` | 5-way owner | Linear 128→5 (auxiliary) |

`BilinearEdgeHead`: `fuse(W_src·s + W_tgt·t + W_edge·e + W_global·g)` — 4·d_model concat MLP yerine paralel projeksiyon toplamı (aynı parametre, çok daha az FLOP).

`SinkhornEdgeHead` tanımlı ama bu policy'de kullanılmıyor (transport-plan bias için opsiyonel).

---

## 2. Node feature derinlemesine analiz

Node girdisi iki bloktan oluşur: **ham gözlem (34)** + **graph node metrikleri (10)**.

### 2.1 Ham gözlem bloğu (obs_builder, 34 kanal)

| Grup | Idx | Kanallar | Değerlendirme |
|------|-----|----------|---------------|
| Ownership | 0–4 | self / enemy0-2 / neutral one-hot | Sağlam; 4 düşman slotu 1v1'de israf |
| Ekonomi | 5,6,15 | ships, production, log_ships | `ships/100` büyük gezegende **doygunluğa** yakın (200+ gemi → 2.0+) |
| Geometri | 7–10 | radius, r_sun, cos/sin θ | İyi; rotasyon-değişmez |
| Yörünge | 11–14 | static/rotating/comet, omega | İyi |
| Incoming | 16–25 | self/enemy bucket + total | **Çok zengin**; ETA-bucketlı tehdit |
| Tehdit özeti | 26–33 | min_eta, soonest/largest, net_threat, capture_value, defense_deficit | Türev sinyaller; bazıları redundant |

**Güçlü yanlar**
- Incoming-fleet ETA bucket'ları (16–23) zamansal tehdit bilgisini iyi taşıyor.
- `capture_value` ve `defense_deficit` zaten ekonomik/savunma sinyali veriyor.

**Zayıflıklar**
1. **Ölçek doygunluğu:** `ships/100` ve `production/5` lineer normalize; büyük gezegenlerde >1 değerler GELU/LayerNorm öncesi dengesiz. `log_ships` var ama `ships` ham hali de ayrıca giriyor.
2. **Ownership israfı:** 1v1 eğitimde `enemy_0..2` 3 slot ama tek düşman var → 2 kanal hep 0.
3. **Mutlak vs göreli yok:** Gezegenin **kendi imparatorluğuma göre** konumu yok (örn. "sınır gezegeni miyim, iç bölge mi"). Sadece mutlak polar koordinat var.
4. **Zamansal birikim yok:** "Bu gezegen kaç turdur benim", "ne kadar üretim biriktirdi" gibi tarihçe yok (Markov varsayımı, ama held_prod kısmen telafi ediyor).

### 2.2 Graph node metrik bloğu (10 kanal)

Tam tanımlar `docs/node_metrics_complete_reference.ipynb` §5'te. Özet değerlendirme:

| # | Kanal | Sorun / Not |
|---|-------|-------------|
| 0 | friendly_influence | Sağlam (kernel-ağırlıklı dost güç) |
| 1 | enemy_influence | Sağlam |
| 2 | **influence_balance** | **Redundant**: `0 − 1`. Encoder zaten öğrenebilir. |
| 3 | friendly_support | Diagonal-sıfır; **kendi garrison'unu saymıyor** → ping-pong artefaktı |
| 4 | supportability | `support/(enemy+incoming+1)`, clamp/10 |
| 5 | vulnerability | `enemy_inf/(ships+inc_friendly+1)`, clamp/10 |
| 6 | production_potential | Ownership-blind diffusion |
| 7 | friendly_production_potential | Ownership-aware; **capture sinyali burada** |
| 8 | **strategic_potential** | **Redundant**: `prod + friendly − enemy` lineer combo |
| 9 | held_discounted_production | Garrison-bağımlı; ping-pong'un asıl kaynağı |

**Kritik bulgular**
- **2 ve 8 redundant** (lineer kombinasyonlar). GNN'nin ilk Linear katmanı bunları zaten üretebilir; girdi olarak vermek **kapasiteyi şişiriyor ve gradyan gürültüsü ekliyor**. Reward'da zaten ağırlıkları 0.
- **Kanal ölçekleri tutarsız:** influence `/100`, production `/25`, supportability/vulnerability `[0,1]`, held_prod `/500`. LayerNorm girişte var ama **kanal-içi** clamp'lar bilgi kaybı yaratıyor (örn. supportability 10'da satüre).
- **3 (friendly_support) tanımı** kendi gemisini saymadığı için "gemiyi başka gezegene yığ → komşunun support'u artar" yapay sinyali doğuruyor (ping-pong analizinde gösterildi).

---

## 3. Edge feature derinlemesine analiz

Edge girdisi: **edge_bias (6)** + **graph edge metrikleri (9)** = 15.

### 3.1 edge_bias bloğu (obs_builder, 6 kanal)

| Idx | Kanal | Not |
|-----|-------|-----|
| 0 | distance/100 | ETA fallback'i |
| 1,2 | dx, dy / 100 | Yönsel |
| 3,4 | sin/cos(Δφ) | Güneş etrafı açı farkı (yörünge-değişmez) |
| 5 | crosses_sun | **Binary** — güneş engeli |

**Zayıflık:** `crosses_sun` binary; gerçekte güneşe teğet geçişin maliyeti süreklidir. Ayrıca dx/dy mutlak çerçevede — rotasyon-değişmezlik sin/cos Δφ'de var ama dx/dy onu bozuyor.

### 3.2 Graph edge metrik bloğu (9 kanal)

| # | Kanal | Sorun / Not |
|---|-------|-------------|
| 0 | eta_norm | **Anlık** intercept ETA; rotasyon dinamiğini tek karede donduruyor |
| 1 | movement_resistance | `eta_norm·(1+crosses_sun)` — eta ile **korelasyonlu** |
| 2 | same_owner | İyi yapısal sinyal |
| 3 | src_is_self | Düğüm feature'ının kenara kopyası (redundant) |
| 4 | tgt_is_self | Aynı — düğümde zaten var |
| 5 | tgt_is_enemy | Aynı |
| 6 | tgt_is_neutral | Aynı |
| 7 | strategic_gradient | **Redundant kaynağa dayanıyor** (strategic_potential farkı) |
| 8 | flow_opportunity | `relu(Δstrategic)/(eta+1)` — yine strategic'e bağlı |

**Kritik bulgular**
1. **3–6 (src/tgt ownership) düğümde zaten mevcut.** GINEConv mesajda `x_j + e_ij` topladığı için hedef ownership'i komşu düğümden zaten geliyor; kenara tekrar koymak **redundant**. Yalnızca `same_owner` (2) gerçek ilişkisel bilgi.
2. **7–8 strategic_potential'a dayalı** — strategic'in kendisi redundant combo. Yani edge tarafında da türev-üstüne-türev sinyal var.
3. **Feasibility yok:** "Bu kaynaktan bu hedefe **yeterli gemiyle** ulaşıp ele geçirebilir miyim" gibi aksiyona-yakın bir kenar sinyali yok. `flow_opportunity` strateji gradyanı veriyor ama gemi bütçesi/ele geçirme eşiği yok.
4. **Zaman dinamiği yok:** Dönen gezegenlerde ETA tura göre değişir; sadece **t anındaki** intercept var. "ETA'nın değişim yönü" (kapanıyor mu açılıyor mu) yok.

---

## 4. GINEConv mesaj geçişinin yapısal sonuçları

1. **Tam bağlı graf + sum agregasyon.** Her düğüm `n−1` komşudan `ReLU(x_j + e_ij)` toplar. Gezegen sayısı 5–25 arası değiştiği için **mesaj büyüklüğü n ile ölçekleniyor**. `LayerNorm` bunu kısmen düzeltir ama erken katmanda dağılım kayması olur. → **mean agregasyon veya derece normalizasyonu** daha kararlı olurdu.
2. **Edge ile node aynı uzaya projekte ediliyor** (`edge_proj → d_model`); GINEConv `edge_dim=d_model` ile çalışıyor. Bu doğru ama **edge_proj hem conv'da hem aksiyon head'lerinde paylaşılıyor** — temsil tutarlı ama edge bilgisinin aksiyon-spesifik kısmı conv tarafından "ezilik" olabilir.
3. **3 katman + FC graf** → 3 hop yeterli (graf çapı 1, çünkü FC). Aslında **tek hop bile tüm grafı görüyor**; 3 katman daha çok non-linearite/derinlik için. Daha derin katman muhtemelen marjinal.
4. **Residual ilk katmanda yok** (`use_input_embed=False` yolunda). Default `True` yolunda residual var; iyi.
5. **graph_summary = mean ⊕ max ⊕ global** → value head buradan besleniyor. Max-pool tek baskın gezegene duyarlı; mean tüm grafı yumuşatıyor. İyi bir kombinasyon ama **owner-ayrıştırılmış pooling yok** (benim gezegenlerimin ortalaması vs düşmanınki ayrı havuzlanmıyor).

---

## 5. Tespit edilen zayıflıklar (özet)

| # | Zayıflık | Kategori | Etki |
|---|----------|----------|------|
| W1 | `influence_balance` (2) ve `strategic_potential` (8) redundant lineer combo | Node | Kapasite israfı, gürültü |
| W2 | Edge `src/tgt is_self/enemy/neutral` (3–6) düğümde zaten var | Edge | Redundant, FLOP israfı |
| W3 | Edge `strategic_gradient`/`flow_opportunity` redundant kaynağa dayalı | Edge | Türev-üstü-türev |
| W4 | `friendly_support` kendi garrison'u saymıyor → ping-pong artefaktı | Node | **Reward exploit** |
| W5 | Kanal ölçekleri tutarsız + clamp doygunluğu | Node/Edge | Bilgi kaybı |
| W6 | `ships/100` doygunluk; ham + log birlikte | Node | Dengesiz ölçek |
| W7 | Owner-göreli konum / sınır-gezegen feature yok | Node | Stratejik körlük |
| W8 | Edge feasibility (ele geçirme eşiği) yok | Edge | Aksiyon-uzak sinyal |
| W9 | ETA tek-kare; zaman dinamiği (kapanıyor/açılıyor) yok | Edge | Rotasyon körlüğü |
| W10 | GINE sum agregasyon n ile ölçekleniyor | Encoder | Dağılım kayması |
| W11 | Owner-ayrıştırılmış pooling yok | Encoder | Value head zayıf özet |
| W12 | 1v1'de enemy_0..2 slot israfı | Node | Küçük |

---

## 6. İyileştirme önerileri (öncelik sıralı)

### Öncelik A — Düşük risk, hızlı kazanım

**A1. Redundant kanalları çıkar (W1, W2, W3).**
- Node'dan `influence_balance` ve `strategic_potential` çıkar → 10 → **8 kanal** (node input 44 → 42).
- Edge'den `src_is_self`, `tgt_is_self`, `tgt_is_enemy`, `tgt_is_neutral` çıkar → `same_owner` kalsın. `strategic_gradient`/`flow_opportunity` ya çıkar ya da raw influence farkına dayandır.
- Beklenen etki: aynı veya daha iyi performans, daha az parametre, daha temiz gradyan. **Ablation için ideal ilk adım.**

**A2. Ölçek normalizasyonunu düzelt (W5, W6).**
- Tüm graph metriklerini `[0,1]` veya `[-1,1]`'e oturt; clamp yerine `tanh`/`log1p` sıkıştırma kullan (supportability/vulnerability'de 10'da satüre olmasın).
- `ships` için sadece `log_ships` tut veya `tanh(ships/200)` kullan; ham `ships/100`'ü kaldır.

**A3. Owner-ayrıştırılmış pooling (W11).**
- `graph_summary`'ye `mean_pool(self_nodes)`, `mean_pool(enemy_nodes)` ayrı havuzları ekle. Value head "benim toplam gücüm vs düşmanın" sinyalini doğrudan görür → **kazanma tahmini iyileşir** (şu an valL ~1.0–1.8, win=0).

### Öncelik B — Orta risk, yapısal kazanım

**B1. Owner-göreli konum feature'ları (W7).**
- Her gezegene ekle: `r / (benim_centroid'e uzaklık)`, `is_frontier` (en yakın düşman gezegene ETA < eşik), `controlled_neighbors_ratio`.
- Sınır gezegenlerini tanımak savunma/genişleme kararını netleştirir.

**B2. Edge feasibility kanalı (W8).**
- Yeni edge kanalı: `capture_feasible = relu(src_ships − (tgt_ships + tgt_prod·eta)) / norm` — "bu kaynaktan bu hedefi ele geçirmeye yetecek gemi var mı". Aksiyona doğrudan bağlı, `flow_opportunity`'den çok daha bilgilendirici.

**B3. ETA zaman dinamiği (W9).**
- `d(eta)/dt` işareti: dönen gezegen yaklaşıyor mu uzaklaşıyor mu. İki ardışık-tur ETA farkı veya analitik türev. Rotasyon timing'ini öğrenmeyi kolaylaştırır.

### Öncelik C — Yüksek risk / deneysel

**C1. GINE agregasyon değişikliği (W10).**
- `GINEConv` yerine derece-normalize edilmiş bir conv (örn. `GENConv` softmax aggr, veya `TransformerConv` edge-aware attention). FC graf + attention, "tek baskın hedefe yığılma" eğilimini de azaltabilir.
- Alternatif: `SinkhornEdgeHead` (kodda zaten var) `pair_score_head` yerine devreye alınarak transport-plan bias eklenebilir → fleet-spam exploitini doğal olarak bastırır.

**C2. Ownership one-hot sıkıştırma (W12).**
- 1v1 için `enemy_0..2` → tek `is_enemy` kanalı. Çok-oyunculuda geri aç.

---

## 7. Önerilen deney planı

| Adım | Değişiklik | Ölçüt | Karşılaştırma |
|------|-----------|-------|---------------|
| E0 | Baseline (mevcut) | win rate, valL, len | — |
| E1 | A1: redundant kanal çıkar | aynı win, ↓param | E0 |
| E2 | A2 + A3: ölçek + owner pooling | ↑valL, ↑win | E1 |
| E3 | B1 + B2: owner-göreli + feasibility | ↑win, ↓ping-pong | E2 |
| E4 | C1: attention conv veya Sinkhorn | ↑win, ↓fleet-spam | E3 |

**Notlar**
- Her feature değişikliği `PYG_PLANET_FEATURE_DIM` / `PYG_EDGE_FEATURE_DIM` sabitlerini ve eski checkpoint uyumluluğunu kırar → yeni `run_*` dizini gerekir.
- A1–A3 reward'dan **bağımsız**; ping-pong exploit'i tek başına çözmezler (o reward tarafında, `docs/node_metrics_complete_reference.ipynb` §10). Ama feature temizliği reward debugging'i kolaylaştırır.
- Feature ablation'larını **scripted opponent sabit** tutarak çalıştır ki win-rate karşılaştırılabilir olsun.

---

## 8. Hızlı referans — dosya/sembol haritası

| Sembol | Dosya | Satır (yaklaşık) |
|--------|-------|------------------|
| `_graph_node_metrics` | `model_graph_pyg.py` | 249 |
| `_graph_edge_metric_matrix` | `model_graph_pyg.py` | 338 |
| `_estimated_intercept_eta_matrix` | `model_graph_pyg.py` | 157 |
| `PygPlanetGraphEncoder` | `model_graph_pyg.py` | 991 |
| `BilinearEdgeHead` | `model_graph_pyg.py` | 858 |
| `SinkhornEdgeHead` | `model_graph_pyg.py` | 904 |
| `PygPlanetGraphPolicy` | `model_graph_pyg.py` | 1127 |
| `observation_to_pyg_data` | `model_graph_pyg.py` | 472 |
| `_planet_features` | `obs_builder.py` | 142 |
| `compute_node_flux_features` | `rl/flux.py` | 149 |
| `held_discounted_production_value` | `rl/held_production.py` | 108 |

> Bu rapor analiz amaçlıdır; kod değişikliği içermez. Uygulama için Öncelik A'dan başlanması önerilir.
