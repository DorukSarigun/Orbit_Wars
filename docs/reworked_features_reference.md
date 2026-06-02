# Reworked Node Features — Reference

`Orbit_Wars/reworked_features.py` içinde tasarladığımız **yeni, bağımsız** node
feature implementasyonunun referansı. Bu modül legacy metrik kodundan
(`model_graph_pyg._graph_node_metrics`, `rl/held_production.py`, `rl/rewards.py`)
**hiçbir şey import etmez**; her metrik tek başına, vektörel (NumPy + opsiyonel
torch) saf bir fonksiyondur.

Tasarım felsefesi: node feature'larını **ham, yerel ve fiziksel** tutmak;
ilişkisel/karmaşık örüntüleri GNN'in message-passing ile kendisinin öğrenmesine
bırakmak. El ile çarpım (ör. `production × centrality`) üretmiyoruz — onları
ayrı kanal verip GNN'e birleştirme özgürlüğü bırakıyoruz.

Şu ana kadar eklenenler iki gruba ayrılır: **node feature** (gezegen başına
GNN'e giren kanal) ve **iç yardımcı** (başka bir feature içinde kullanılan,
tek başına feature olarak çıkarılmayan fonksiyon).

### Node feature'lar (gezegen başına GNN'e girer) — **potansiyel / garnizon**

| # | Fonksiyon | Ne ölçer | Sahiplikten bağımsız mı? |
|---|-----------|----------|--------------------------|
| 2 | `discounted_production_potential` | Gezegenin ekonomik değeri | Evet |
| 3 | `closeness_centrality` | Geometrik merkezilik | Evet |
| 4 | `standard_node_features` | Ham gözlem kanalları (10 kolon) | Hayır (sahiplik içerir) |
| 6 | `potential_threat` | Hız-duyarlı *potansiyel* düşman baskısı | Hayır (düşman maskesi) |
| 7 | `potential_support` | Hız-duyarlı *potansiyel* dost takviye kapasitesi | Hayır (dost maskesi) |
| 8 | `friendly_compactness` | Bölgemin compactlığı (iç bölge vs. frontier) | Hayır (dost maskesi) |

### Edge feature'lar (yönlü kenar başına GNN'e girer) — **fiili / uçan filo**

| # | Fonksiyon | Ne ölçer |
|---|-----------|----------|
| 9 | `edge_features` | Kenar üzerinde *uçan* filoların ETA-ağırlıklı dost/düşman akışı (+ statik mesafe / güneş) |

> **Potansiyel vs. fiili ayrımı:** node metrikleri (#6, #7) garnizondan
> *fırlatılabilecek* kuvveti ölçer (potansiyel); `edge_features` ise *zaten
> fırlatılmış*, kenar üzerinde uçan filoları ölçer (fiili). İkisi birbirini
> tamamlar.

### İç yardımcılar (feature DEĞİL)

| # | Fonksiyon | Ne yapar | Nerede kullanılır |
|---|-----------|----------|-------------------|
| 1 | `geometric_horizon_factor` | İndirimli kalan tur sayısı | `discounted_production_potential` içinde |
| 5 | `fleet_speed` | Filo hızı (gemi sayısının fonksiyonu) | `potential_threat`/`potential_support` ETA hesabında |
| — | `inflight_flux_matrices` | Yönlü uçan-filo akış matrisleri | `edge_features` içinde |

> **Not:** `fleet_speed` bilerek gezegen başına bir feature **olarak çıkarılmaz**.
> Hız, gezegenin sabit bir özelliği değil; gönderilen **filonun** büyüklüğüne
> bağlıdır. Dolayısıyla yalnızca tehdit ETA'sını hesaplarken (kaynak gezegenin
> garnizonu üzerinden) anlam kazanır.

---

## 1. `geometric_horizon_factor(remaining_turns, gamma)` — yardımcı

Kalan turları, geometrik indirimle "efektif tur sayısına" çevirir.

$$
H(R) = \sum_{k=0}^{R-1} \gamma^{k} = \frac{1 - \gamma^{R}}{1 - \gamma}
$$

- $R = \max(0,\ \text{max\_turns} - \text{turn})$ — kalan tur.
- **Yorum:** her tur $\gamma$ ile indirildiğinde geriye kalan "değerli tur" sayısı.
- **Özellikler:** $H(0) = 0$ (oyun bitti); $R \to \infty$ iken $\frac{1}{1-\gamma}$'ya
  doygunlaşır. `gamma = 1` özel durumda lineer: $H(R) = R$.
- Varsayılan `DEFAULT_GAMMA = 0.99` → doygunluk = $\frac{1}{1-0.99} = 100$ efektif tur.

Bu, tek başına bir feature değil; aşağıdaki ekonomik metriğin çarpanıdır.

---

## 2. `discounted_production_potential(production, remaining_turns)` — ekonomik değer

Bir gezegenin **şu andan oyun sonuna kadar** üreteceği indirimli toplam üretim.
Yani gezegenin **içsel ekonomik değeri**.

$$
V_g = \frac{\text{prod}_g \cdot H(R)}{\text{norm}},
\qquad
H(R) = \frac{1 - \gamma^{R}}{1 - \gamma}
$$

**Hesap adımları:**
1. $H(R)$ hesapla (madde 1).
2. Gezegenin tur başı üretimi $\text{prod}_g$ ile çarp → toplam indirimli üretim.
3. `norm` ile ölçekle ($V$ kabaca $[0,1]$ aralığına otursun).

**Özellikler:**
- **Sahiplikten bağımsız.** Gezegenin kendi değerini yansıtır, kimde olduğunu
  değil — sahiplik/hayatta kalma ayrı kanallarda tutulur. (Bu sayede
  ping-pong'u besleyen "kendi gezegenime gönderince değer artıyor" artefaktı
  bu kanalda *oluşmaz*.)
- **Oyun sonuna doğru azalır.** $R \to 0$ iken $H(R) \to 0$, dolayısıyla
  $V \to 0$ — son turlarda gezegen ele geçirmek neredeyse değersiz.
- **Lineer değil, indirimli.** Yakın gelecek baskın; çok uzun ufuklar
  $\frac{\text{prod}}{(1-\gamma)\cdot\text{norm}}$'a doygunlaşır.

Varsayılanlar: `gamma = 0.99`, `norm = 500` → tur başı ~5 üreten bir gezegen
uzun ufukta $\frac{5 \cdot 100}{500} = 1.0$ civarına oturur.

---

## 3. `closeness_centrality(distance_matrix)` — geometrik merkezilik

Gezegenin, **tam bağlı** (fully-connected) gezegen grafında diğer tüm
gezegenlere ortalama yakınlığı.

$$
C(g) = \frac{N_g - 1}{\sum_{h \neq g} d(g, h)}
$$

- $d(g,h)$ = normalize ikili mesafe (edge weight, ör. $\lVert \Delta pos \rVert / \text{BOARD\_SIZE}$).
- $N_g$ = $g$'den erişilebilir geçerli gezegen sayısı.
- **Yorum:** tüm gezegenlere ortalama mesafesi kısa olan gezegen merkezîdir →
  yüksek değer (lojistik/kuvvet projeksiyonu için iyi üs).

**Hesap adımları:**
1. Köşegeni dışla ($d(g,g)$ sayılmaz).
2. Geçerli (pad olmayan) komşulara olan mesafeleri topla: $\sum_{h\neq g} d(g,h)$.
3. Geçerli komşu sayısına böl (payda $+\varepsilon$ ile sıfır koruması).

**Özellikler:**
- **Sahiplik- ve üretim-bağımsız** — yalnızca takımyıldız geometrisine bağlı saf
  yapısal sinyal. GNN'i `production × centrality` türü çarpımları kendi
  öğrenmesi için serbest bırakır.
- `valid_mask` ile pad gezegenler hem toplamlardan düşülür hem de 0 değer alır.
- Çıktı $[0,1]$'e bağlı **değil**; encoder'daki LayerNorm ölçeği halleder.
  Board-normalize mesafelerle tipik aralık ~$[0.7, 3]$.

> **Not (kapsam):** Centrality **tüm** gezegenler üzerinden bakar (sadece bana ait
> olanlar değil) — yapısal/geometrik bir özellik olduğu için.

---

## 4. `standard_node_features(...)` — ham gözlem bloğu (10 kolon)

Observation'dan doğrudan çıkan, **ham** özellikler. Türetilmiş hiçbir
agregasyon yok; GNN bunları yukarıdaki metriklerle kendi birleştirir.

Kolon düzeni (`STD_IDX_*`):

| idx | kanal | hesap |
|-----|-------|-------|
| 0 | `is_self` | `owner == player` |
| 1–3 | `enemy_0..2` | düşman one-hot, slot = `(owner - player - 1) mod 4` |
| 4 | `is_neutral` | `owner == -1` |
| 5 | `log_ships` | `log(ships + 1) / log(1000)` |
| 6 | `production` | `production / 5` (ham, sadece ölçek) |
| 7 | `radius` | `radius / 5` |
| 8 | `is_rotating` | `rotating > 0.5` |
| 9 | `is_comet` | `comet > 0.5` |

**Notlar:**
- **Sahiplik 4 slot** (self + 3 düşman + neutral): çok oyunculu senaryolarda
  düşmanları ayırt etmek için gerekli; israf değil. Enemy slot ataması legacy
  `obs_builder` ile **birebir aynı** konvansiyonu kullanır.
- `log_ships`: gemi sayısı geniş aralıkta (1–1000+) değiştiği için log-sıkıştırma.
- `production` **ham** hali de burada (madde 2'deki indirimli değerden ayrı) —
  GNN başka bir feature ile birleştirip farklı bir şey bulabilir.

---

## 5. `fleet_speed(ships)` — filo hızı (yardımcı)

Oyun motorunun (`feature_extraction.fleet_speed`, `orbit_wars.py`) hız
formülünün **birebir** vektörel kopyası.

$$
v(s) = 1 + (v_{\max} - 1)\left(\frac{\ln s}{\ln 1000}\right)^{1.5}
$$

- Edge case'ler motorla aynı: $s \le 0 \Rightarrow 0$, $\;s = 1 \Rightarrow 1$.
- $v_{\max} = 6$ (varsayılan). $s$'de monoton artan; $s \approx 1000$'de doygun.
- **Kritik gözlem:** filo ne kadar büyükse o kadar hızlı → aynı mesafeyi daha az
  turda alır → **efektif mesafesi kısalır**. Büyük düşman garnizonlarını
  orantısız tehlikeli yapan tam da budur.

Doğrulama (motorla 6 hane uyumlu):

| ships | speed |
|-------|-------|
| 1 | 1.00 |
| 5 | 1.56 |
| 50 | 3.13 |
| 200 | 4.36 |
| 1000 | 6.00 |

---

## 6. `potential_threat(distance_matrix, owner, ships, player=...)` — hız-duyarlı potansiyel tehdit

Her hedef gezegen $g$ üzerindeki **düşman baskısı**. Efektif mesafe, kaynak
$h$'den tam güçle çıkan bir filonun **varış süresidir (ETA, tur)**:

$$
\text{eta}(h \to g) = \frac{d(h, g)}{v(s_h)},
\qquad
\text{threat}(g) = \sum_{h \in \text{düşman}} \frac{s_h}{\text{eta}(h \to g) + \varepsilon}
$$

Hız yasasını yerine koyunca:

$$
\text{threat}(g) = \sum_{h \in \text{düşman}} \frac{s_h \cdot v(s_h)}{d(h,g) + \varepsilon\, v(s_h)}
$$

Yani her düşman gezegeni **gemi kütlesi × varış hızı**, ham mesafeyle indirimli
olarak katkı yapar. İstenen iki etki tek formülde:

- **daha çok gemi → daha çok kuvvet** ($s_h$ çarpanı), ve
- **daha çok gemi → daha hızlı filo → kısa ETA** ($v(s_h)$ çarpanı).

**Hesap adımları:**
1. Her kaynağın hızı $v(s_h)$ (madde 5).
2. ETA matrisi: `eta[h,g] = raw_dist[h,g] / v(s_h)` (satır = kaynak $h$).
   Normalize mesafe verilirse `board_size` ile ham birime çevrilir → ETA tur cinsinden.
3. Ağırlık `1 / (eta + eta_eps)`; köşegen ($h=g$) ve nötr/kendi gezegenler dışlanır.
4. Düşman kaynaklar üzerinden topla: `threat[g] = Σ_h enemy_mask · s_h · weight`.

**Tasarım notları:**
- **Girdi konvansiyonu** `closeness_centrality` ile aynı: normalize mesafe
  matrisi (`edge_bias[...,0]`). `board_size` ham birime çevirir; ham mesafe
  verirsen `board_size = 1.0`.
- `eta_eps ≈ 1 tur` sıfır-mesafe patlamasını engeller.
- Legacy intercept kernel'inden **daha sadık**: o, hızı sabit referans (~50
  gemi) alıyordu; bu, her kaynağın **gerçek garnizonundan** hızı türetir.
- Maksimum-tehdit varsayımı: düşman tüm garnizonunu tek filoda gönderir.

**Doğrulama** — eşit mesafede (0.5 ≈ 50 birim) iki düşman:

| gezegen | gemi | hız | ETA | tehdit |
|---------|------|-----|-----|--------|
| enemy1 | 20 | 2.43 | 20.6 tur | 0.93 |
| enemy2 | 200 (10×) | 4.36 | 11.5 tur | 16.04 |

**10× gemi → ~17.3× tehdit.** Fazladan ~1.73× tam olarak hız etkisi
(ETA 20.6 → 11.5 tur). İki etki `s_h · v(s_h) / d` içinde çarpılır.

### Alternatif ağırlık çekirdeği (opsiyonel)
`1/(eta + eps)` yerine **üstel ufuk** kullanılabilir:

$$
\text{threat}(g) = \sum_{h} s_h \cdot e^{-\,\text{eta}(h \to g) / \tau}
$$

$\tau$ ≈ tehdit ufku (ör. 15 tur). Tehdidi sınırlar ve "yalnızca $\tau$ tur içinde
varabilen tehditler önemli" yorumunu netleştirir. Şu an `1/(eta+eps)` (mesafe
gravity modeliyle tutarlı) kullanılıyor.

---

## 7. `potential_support(distance_matrix, owner, ships, player=...)` — potansiyel dost takviye

`potential_threat`'in **simetriği**: gezegenim $g$ saldırıya uğrarsa, **diğer**
gezegenlerimden ne kadar hızlı ve güçlü takviye gelebilir?

$$
\text{eta}(h \to g) = \frac{d(h,g)}{v(s_h)}, \qquad
\text{support}(g) = \sum_{\substack{h \in \text{benim} \\ h \neq g}}
    \frac{s_h}{\text{eta}(h \to g) + \varepsilon}
$$

Aynı hız-duyarlı gravity çekirdeği (`_speed_weighted_pressure`), tek fark
**dost maskesi** (`owner == player`).

**Tasarım notları:**
- Köşegen dışlanır → $g$'nin **kendi garnizonu sayılmaz**. O, yerinde anlık
  savunmadır ve zaten `standard_node_features`'taki `log_ships` kanalında var.
  Bu metrik yalnızca **başka yerden gelebilecek takviyedir**.
- Aynı çifte etki: büyük dost garnizonu hem çok gemi taşır hem hızlı yetişir.
- `potential_threat` ile birlikte verildiğinde GNN **net savunulabilirliği**
  (support − threat) kendisi çıkarabilir; farkı elle kodlamayız.
- Düşman gezegenlerde de hesaplanır ama oradaki değer anlamsızdır ("oraya ne
  kadar takviye yollarım"); sahiplik one-hot'u GNN'in bunu kapılaması (gate)
  için yeterli sinyali verir. Aynısı `potential_threat` için de geçerli.

**Doğrulama** — ön cephedeki gezegenim (5 gemi), yakında 150-gemilik üs:

```
potential_threat[front]  = 6.48
potential_support[front] = 18.53   (150-gemilik üsten hızlı takviye)
net (support - threat)   = +12.05  -> savunulabilir
```

---

## 8. `friendly_compactness(distance_matrix, owner, player=...)` — bölge compactlığı

Bölgemin **per-node** compactlığı: `closeness_centrality`'nin **sadece kendi
gezegenlerim** üzerinde değerlendirilmiş hali (dost alt-graf).

```
friendly_compactness(g) = (N_benim - 1) / sum_{h: benim, h != g} d(g, h)
```

**Yorum:**
- **Yüksek** → `g` diğer gezegenlerimle çevrili: derin **iç bölge**, kısa iç
  takviye hatları, kolay tutulur.
- **Düşük** → `g` dost gezegenlerden uzak: açık **frontier** / kanat gezegeni.
- Tek başına gezegen (başka dost yok) → 0 (maksimum frontier).

**Tasarım notları:**
- Yalnızca kendi gezegenlerim değer alır; düşman/nötr 0 döner (benim bölgemin
  compactlığı onlar için anlamsız; sahiplik one-hot'u GNN'in kapılaması için
  yeterli). `closeness_centrality(D, valid_mask=is_friendly)` üzerine ince bir
  wrapper.
- `potential_threat` ile birlikte verildiğinde GNN "açık frontier gezegenim
  tehdit altında" gibi çıkarımları yapabilir.
- Girdi konvansiyonu `closeness_centrality` ile aynı (normalize mesafe).

**Doğrulama** — 0,1 sıkı küme; 2 uzak frontier; 3 düşman:

```
compactness = [2.00, 2.22, 1.18, 0.00]
              (iç)  (iç)  (frontier) (düşman -> 0)
tek başına kalan dost gezegen -> 0.00
```

---

## 9. `edge_features(distance_matrix, *, fleet_*...)` — uçan filolar (edge)

Node metrikleri (#6, #7) garnizonun **potansiyelini** ölçer; bu fonksiyon
**fiilen uçan** filoları, ait oldukları **yönlü kenar** üzerinde temsil eder.
Bir filo fiziksel olarak kaynak→hedef kenarında durduğu için doğal yeri edge.

Çıktı `(n, n, 4)`; girdi `[i, j]` yönlü kenarı (i → j) tanımlar:

```
[ distance(i,j),         # statik geometri (edge_bias[...,0])
  crosses_sun(i,j),      # statik: i->j düz yolu güneş engelliyor mu (0/1)
  friendly_flux(i->j),   # ETA-ağırlıklı dost gemi akışı / ship_norm
  enemy_flux(i->j) ]     # ETA-ağırlıklı düşman gemi akışı / ship_norm
```

**Flux hesabı** (`inflight_flux_matrices` yardımcısı):
her uçan filo $k$, $s_k$ gemiyle `fleet_src[k]` → `fleet_tgt[k]` uçar, ETA
$e_k$ turda varır ve sahibinin matrisine **yakınlık ağırlığı** kadar katkı yapar:

```
M[src, tgt] += w(s_k, e_k)
```

**Yakınlık ağırlığı** (varsayılan `exp_decay`, $\tau = 10$):

$$
w(s, e) =
\begin{cases}
s & \text{raw} \\
s \cdot \max(0,\ 1 - e/\text{eta\_norm}) & \text{linear} \\
s \cdot e^{-e/\tau} & \text{exp\_decay (varsayılan)}
\end{cases}
$$

Yani varışa yakın filo neredeyse tam kütlesiyle, uzaktaki ağır indirimli sayılır
— tehdit tarafında konuştuğumuz "üstel ufuk" mantığının aynısı.

**ETA tutarlılığı (önemli):** filonun `fleet_eta` değeri navigation tarafından
filonun **gerçek gemi sayısıyla** (yani `fleet_speed(ships)` ile) hesaplanır.
Dolayısıyla "büyük filo → kısa ETA" etkisi burada **otomatik** dahildir;
edge tarafında `fleet_speed`'i tekrar uygulamaya gerek yok (node tarafında elle
uyguluyoruz çünkü orada henüz fırlatılmamış potansiyel kuvveti modelliyoruz).

**GNN ne çıkarır:** flux yönü + hedef sahipliği birleşince —
- düşman flux → bana ait $j$ = **gelmekte olan saldırı**,
- dost flux → bana ait $j$ = **yoldaki takviye** (`potential_support`'un fiili karşılığı).

**Doğrulama** — planet0'a iki uçan filo:

```
fleet 1->0: 30 dost gemi, eta=3   -> edge[1,0] = [0.30, 0, 0.444, 0    ]
fleet 3->0: 40 düşman gemi, eta=8 -> edge[3,0] = [0.40, 0, 0,     0.359]
filosuz kenar (0->2)              -> edge[0,2] = [0.60, 0, 0,     0    ]
```

(0.444 = 30·e^(−3/10)/50, 0.359 = 40·e^(−8/10)/50 — doğrulandı.)

---

## 10. `incoming_enemy_estimate(...)` — allocator savunma talebi

`reworked_allocator.allocate_launches`'a verilen **`incoming_enemy`** dizisini
(her gezegene gelecek düşman gemi sayısı) metriklerden türetir. Garnizonla
doğrudan kıyaslanabilmesi için **ham gemi birimi** üretir (potential_threat gibi
gravity-ağırlıklı değil).

İki gemi-birimi bileşeni:

1. **Uçan (kesin).** Her gezegene *yola çıkmış* düşman gemi kütlesi — ham
   düşman flux matrisinin (`inflight_flux_matrices(mode="raw")`) kolon toplamı.
   Bu filolar fırlatılmış, kesin gelecek → tam kütle.
2. **Ufuk içindeki potansiyel (opsiyonel).** `horizon` tur içinde gezegene
   *ulaşabilecek* düşman garnizonları: `eta(h→T) = dist / fleet_speed(ships_h)
   ≤ horizon` olan her düşman gezegen `h` için `ships_h` eklenir.
   `potential_weight` ile ölçeklenir (varsayılan 0 → yalnız uçan). Artırmak
   savunmayı proaktif yapar ama aşırı-rezerv riski getirir.

```
incoming(T) = Σ_{k: düşman filo → T} s_k                          (uçan, ham)
            + w · Σ_{h: düşman, eta(h→T) ≤ H} s_h                (potansiyel)
```

| Argüman | Açıklama |
|---------|----------|
| `distance_matrix, owner, ships` | Mevcut tahta durumu |
| `player` | Bizim oyuncu id'miz (düşman = diğer ≥0 sahipler) |
| `fleet_*` | Uçan filo dizileri (src, tgt, eta, ships, is_self, is_enemy, mask) |
| `horizon` | Potansiyel için ETA penceresi (tur), vars. 15 |
| `potential_weight` | Potansiyel bileşen ağırlığı (vars. 0 = yalnız uçan) |
| `board_size, max_speed` | Normalize mesafe → ham + hız tavanı |

Dönüş: `(n,)` `float32`, negatif olmayan, gezegen başına gelen düşman gemisi.

**Not:** Uçan filolar ham (zamanlamadan bağımsız) sayılır; ileride savunma
penceresinin çok ötesinde varan filolar iskonto edilebilir. Allocator
`need(T) = incoming(T) − garrison(T) − production(T)·eta` formülünde kullanır,
yani üretim + garnizon yeterse takviye gönderilmez.

---

## Sabitler (modül seviyesi)

| Sabit | Değer | Açıklama |
|-------|-------|----------|
| `DEFAULT_GAMMA` | 0.99 | Tur başı ekonomik indirim |
| `DEFAULT_NORM` | 500 | `discounted_production_potential` ölçeği |
| `MAX_ENEMY_OWNER_SLOTS` | 3 | Düşman one-hot slot sayısı |
| `SHIP_LOG_BASE` | 1000 | `log_ships` tabanı |
| `PRODUCTION_NORM` / `RADIUS_NORM` | 5 / 5 | Ham kanal ölçekleri |
| `DEFAULT_MAX_SHIP_SPEED` | 6.0 | Hız tavanı $v_{\max}$ |
| `SPEED_LOG_BASE` / `SPEED_EXPONENT` | 1000 / 1.5 | Hız formülü |
| `DEFAULT_BOARD_SIZE` | 100 | Normalize mesafe → ham birim |
| `DEFAULT_THREAT_ETA_EPS` | 1.0 | ETA tabanı (tur) |
| `DEFAULT_FLUX_MODE` | `exp_decay` | Uçan filo yakınlık ağırlığı modu |
| `DEFAULT_FLUX_TAU` | 10.0 | `exp_decay` zaman sabiti (tur) |
| `DEFAULT_FLUX_ETA_NORM` | 50.0 | `linear` mod ETA normalizasyonu (tur) |
| `DEFAULT_FLUX_SHIP_NORM` | 50.0 | Flux kanalları gemi normalizasyonu |
| `REWORKED_EDGE_FEATURE_DIM` | 4 | Edge feature kanal sayısı |
