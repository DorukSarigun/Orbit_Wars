# Orbit Wars RL Final Design

Bu dokuman, GA/scripted tabanli ajanin tavaninin dusuk kalmasi uzerine
tasarlanan yeni RL tabanli graph-policy ajaninin final mimarisini toparlar.

Yeni karar:

```text
RL policy dogrudan edge launch secmeyecek.
RL policy node-level stratejik sinyaller uretecek.
Role-aware LP / min-cost-flow allocator, gemi dagilimini cozecek.
Gercek env action yine launch listesidir.
```

Ana pipeline:

```text
Observation
  -> graph feature builder
  -> edge-aware GNN
  -> node strategic heads
  -> role-aware allocator objective
  -> LP / min-cost-flow / bounded greedy solver
  -> launch builder: compute_launch(src, tgt, ships)
  -> env action list
```

Bu tasarimda GNN "hangi gezegen ne ise yarar?" sorusuna cevap verir. Gemi
sayisi ve source-target eslemesi ise deterministic allocator tarafindan cozulur.

---

## 1. Neden Edge-Only'den Node+Allocator'a Geciyoruz?

Edge-only policy su kararlari ayni anda ogrenmek zorundaydi:

```text
hangi source?
hangi target?
hangi role?
kac gemi?
ne zaman?
```

Bu, Orbit Wars icin zor bir kredi problemi yaratiyor. Capture ve defense
genellikle tek launch ile degil, birden fazla source'un ayni hedefe dogru
koordine edilmesiyle oluyor. Edge-only policy yanlis amount secince hedef
alinsa bile overkill, hedef alinmayinca da undercommit belirsiz kalabiliyor.

Node+allocator tasariminda RL daha makro bir problem cozer:

```text
neresi savunulmali?
neresi saldiri hedefi?
neresi donor olabilir?
nerede gemi tutmak degerli?
neresi ileri us / support anchor?
neresi tahliye edilmeli?
```

Allocator ise su lokal lojistik problemi cozer:

```text
hangi source'dan hangi target'a kac gemi gitmeli?
kaynak gezegen ne kadar bosaltilabilir?
savunma acigi en ucuz nasil kapatilir?
attack threshold hangi source kombinasyonu ile tamamlanir?
support hangi cephe gezegenlerine akmali?
```

Bu ayrim, amount ratio problemine ve "gereksiz defense mi, useful support mu?"
sorusuna daha iyi cevap verir.

---

## 2. Graph Temsili

Graph tanimi:

```text
node = planet / comet
directed edge i -> j = i gezegeninden j gezegenine potansiyel rota
```

Filo nesnelerini ilk surumde ayri node yapmayacagiz. Ucan filolar node ve edge
feature olarak aggregate edilecek.

Neden:

- Filo sayisi degisken ve yuksek olabilir.
- Fleet-as-node, graph boyutunu ve credit assignment'i buyutur.
- Ilk hedef, mevcut gezegenlerden yapilabilecek stratejik dagilimi cozmek.

Ucan filolarin aggregate temsili:

```text
node-level:
  incoming_self_ships
  incoming_enemy_ships
  incoming_enemy_min_eta
  enemy_arrival_power_by_window
  friendly_arrival_power_by_window

edge-level:
  friendly_flux_i_j
  enemy_flux_i_j
  ETA-weighted in-flight mass
```

---

## 3. Node Feature Tasarimi

Node feature'lar uc gruba ayrilir: ham fiziksel bilgi, taktik durum, stratejik
potansiyel.

### 3.1 Ham Fiziksel ve Sahiplik Feature'lari

```text
owner_self
owner_enemy_0
owner_enemy_1
owner_enemy_2
owner_neutral
ships_log
production
radius
is_rotating
is_comet
comet_life
r_sun
sin_theta
cos_theta
omega
```

Notlar:

- `ships_log`, filo hizinin log yapisina daha uygun bir olcek verir.
- `production` ham haliyle verilmeli; GNN bunu threat/diffusion/centrality ile
  kendisi birlestirmeli.
- `is_comet` ve `comet_life`, comet yakalama/tahliye kararlarini ayirmak icin
  gerekli.

### 3.2 Taktik Durum Feature'lari

```text
incoming_self_ships
incoming_enemy_ships
incoming_enemy_min_eta
defense_deficit
potential_enemy_threat
potential_friendly_support
threat_support_balance
local_enemy_pressure
local_friendly_pressure
```

Mevcut kaynaklar:

- `reworked_features.potential_threat`
- `reworked_features.potential_support`
- `orbit_graph` node pressure kanallari
- `orbit_defense.compute_owned_defense`

Tasarim karari:

`potential_enemy_threat`, `potential_friendly_support` ve
`threat_support_balance` feature olarak verilebilir. Reward'da bu ucluyu ayni
anda bagimsiz hedefler gibi buyuk agirlikla saymak double count riski dogurur.

### 3.3 Stratejik Potansiyel Feature'lari

```text
discounted_production_value
closeness_centrality
friendly_compactness
friendly_diffusion_potential
enemy_diffusion_potential
diffusion_balance
attack_access_to_enemy_value
frontier_anchor_value
```

Mevcut kaynaklar:

- `reworked_features.discounted_production_potential`
- `reworked_features.closeness_centrality`
- `reworked_features.friendly_compactness`
- `orbit_production_pressure.annotate_diffusion_potential_on_graph`

Kritik karar:

Sadece `diffusion_balance` vermek yeterli degil. Bilesenler de verilmeli:

```text
friendly_diffusion
enemy_diffusion
friendly_diffusion - enemy_diffusion
```

Cunku:

- ikisi de yuksekse: contested / savas alani
- ikisi de dusukse: uzak veya onemsiz alan
- fark ayni olsa bile stratejik anlam farkli olabilir

---

## 4. Edge Feature Tasarimi

Edge feature'lar policy'nin node sinyallerini uretmesine yardim eder ve
allocator maliyetlerini belirler.

Her directed edge `i -> j` icin:

```text
reachable
eta
distance
crosses_sun
angle_sin
angle_cos
dphi_sin
dphi_cos
friendly_flux_i_j
enemy_flux_i_j
source_surplus
source_strip_risk
target_capture_cost_estimate
target_value
target_enemy_threat
target_friendly_support
delta_diffusion_if_capture_j
```

`crosses_sun` sadece "imkansiz" anlamina gelmeyebilir; `compute_launch`
hareketli hedefe baska intercept bulabilir. Bu yuzden edge feature olarak
verilir, final legalite yine geometri katmaninda kontrol edilir.

### 4.1 Delta Diffusion If Capture

`delta_diffusion_if_capture_j` su soruya cevap verir:

```text
Ben j gezegenini alirsam uretim/lojistik erisim alanim ne kadar iyilesir?
```

Temiz hesap:

```text
Phi_diff_self(s) =
  sum_t value_t * tanh(friendly_diffusion_t - enemy_diffusion_t)

diffusion_gain_after_capture =
  clip(Phi_diff_self(s with j.owner=self) - Phi_diff_self(s), -cap, cap)
```

Pahali olursa approximate hesap:

```text
local_delta(j) =
  production_j * sum_k eta_weight(j -> k) * value_k
  - enemy_owner_loss_term
```

Double-count notu:

`target_discounted_production_value` hedefin kendi uretimini sayar.
`diffusion_gain_after_capture` ise hedefin ileri us / erisim etkisini saymali.
Ayni production ham olarak iki kez eklenmemeli; diffusion gain clip edilmeli ve
kucuk katsayi ile kullanilmali.

---

## 5. GNN Encoder

Edge-aware message passing kullanilacak.

Basit form:

```text
h_i^0 = NodeMLP(node_features_i)
e_ij^0 = EdgeMLP(edge_features_ij)

for layer in layers:
    m_ij = MessageMLP([h_i, h_j, e_ij])
    a_ij = EdgeGateMLP([h_i, h_j, e_ij])
    h_j = NodeUpdate(h_j, sum_i softmax_i(a_ij) * m_ij)
    e_ij = EdgeUpdate([h_i, h_j, e_ij])

global_embedding = attention_pool({h_i})
```

Output:

```text
node_embedding[i]
edge_embedding[i,j]
global_embedding
```

Tasarim prensibi:

- Node embedding stratejik skor head'lerini besler.
- Edge embedding allocator cost/value matrislerini kosullar.
- Global embedding value head ve global aggression/budget parametrelerini besler.

---

## 6. Node Strategic Heads

Node tag'lere geri donuyoruz, ama bunlar dogrudan env action degil. Bunlar
allocator'a verilen stratejik sinyallerdir.

Hard label yerine skor/olaslik uretmek daha iyi:

```text
defense_demand[i]
support_demand[i]
attack_value[i]
expand_value[i]
donor_capacity[i]
reserve_value[i]
evacuation_urgency[i]
hold_value[i]
frontier_anchor_value[i]
```

Ownership'e gore aktif anlam:

```text
OWN planet:
  defense_demand
  support_demand
  donor_capacity
  reserve_value
  evacuation_urgency
  frontier_anchor_value

NEUTRAL planet:
  expand_value
  attack_value

ENEMY planet:
  attack_value
  pressure_value
```

Burada `tag` kelimesi karar niyetini anlatir; action birimi degildir. Gercek
launch sadece allocator flow'u sonucu olusur.

### 6.1 Neden Hard Tag Degil?

Tek bir node ayni anda birden fazla stratejik role sahip olabilir:

```text
frontline owned planet:
  hem defense_demand yuksek olabilir
  hem support anchor olabilir
  hem reserve_value yuksek olabilir

zengin enemy planet:
  hem attack_value yuksek olabilir
  hem diffusion gain yuksek olabilir
```

Bu nedenle `argmax tag` yerine continuous skorlar kullanmak daha esnek olur.
Gerekirse training icin auxiliary tag label tutulabilir, ama inference'ta
allocator continuous skorlarla calisir.

---

## 7. Role-Aware Allocator

Allocator, policy skorlarini ve edge maliyetlerini alip gemi akislarini cozer.

Karar degiskenleri:

```text
x_attack[i,j]   = owned source i -> neutral/enemy target j attack ships
x_defense[i,j]  = owned source i -> owned target j defense ships
x_support[i,j]  = owned source i -> owned target j support/consolidation ships
x_evac[i,j]     = risky owned source i -> safe owned target j evacuation ships
```

Env'e giden launch yine `i -> j, ships` olur. Role bilgisi training ledger ve
credit assignment icin saklanir.

Allocator'in deterministik olmasi su anlama gelir:

```text
node_intents z ~ policy(s)
flow_plan x = Allocator(state=s, intents=z, constraints=c)
launches   = LaunchBuilder(state=s, flow_plan=x)
```

Yani allocator sadece intent'in degil, `state + intent + constraints`
kombinasyonunun deterministik fonksiyonudur. PPO logprob'u allocator'dan degil
policy'nin sample ettigi `z` intent'lerinden gelir. Credit ise `Allocator(s,z)`
sonucunda olusan flow'larin outcome'una gore yine `z` intent'lerine geri
yazilir.

Bu ayrim onemli:

```text
policy'nin ogrendigi sey:
  bu state'te hangi node intent'leri allocator'i iyi flow'a goturur?

allocator'in yaptigi sey:
  verilen intent ve mevcut state kisitlari altinda lojistik cozum uretmek
```

Allocator'a dogrudan gradient yazmayacagiz; ancak allocator kararlari ledger'da
tutulacak ve hangi node intent'in hangi flow'a sebep oldugu izlenecek.

### 7.1 Source Budget

Her owned source icin:

```text
available[i] = current_ships[i] + safe_outbound_return_estimate

reserve_required[i] =
    base_hold[i]
  + enemy_incoming_risk[i]
  + local_enemy_pressure[i]
  + reserve_value[i] * reserve_scale
  + production[i] * production_hold_weight

send_budget[i] =
  max(0, available[i] - reserve_required[i])
```

Bu form basit tabandir. Daha dogru donor kapasitesi `current surplus` degil,
`projected safe surplus` olmalidir. Bir gezegen enemy incoming altinda olsa bile
donor olabilir; eger enemy gelmeden once production ve friendly support ile
tekrar guvenli hale gelebiliyorsa gemi verebilir.

```text
projected_garrison_at_enemy_arrival[i] =
    current_ships[i]
  - planned_outgoing_from_i
  + production[i] * enemy_eta[i]
  + friendly_support_arriving_before_enemy[i]
  - other_enemy_arrivals_before_main_threat[i]

safe_after_launch[i] =
  projected_garrison_at_enemy_arrival[i] >= enemy_power[i] + reserve_margin[i]
```

Bu kosul saglaniyorsa incoming altindaki gezegenin `donor_capacity` tamamen
kapatilmamali. Saglanmiyorsa donor budget clamp edilmeli veya source strip cost
cok yukseltilmelidir.

Kisit:

```text
sum_j (
    x_attack[i,j]
  + x_defense[i,j]
  + x_support[i,j]
  + x_evac[i,j]
) <= send_budget[i]
```

Bu, source strip riskini allocator seviyesinde bastirir. Yine de source strip
cezasi reward/credit tarafinda kalmali, cunku reserve estimate hatali olabilir.

### 7.2 Defense Need

Defense icin hedef bazli ihtiyac:

```text
projected_local_defense[j] =
    current_garrison[j]
  + production_until_enemy_arrival[j]
  + friendly_inbound_before_enemy[j]

required_defense[j] =
  max(0, enemy_arrival_power[j] - projected_local_defense[j] + margin[j])

defense_need[j] =
  required_defense[j] * defense_demand[j]
```

Kisit / objective:

```text
0 <= sum_i x_defense[i,j] <= defense_need[j] + allowed_buffer[j]
```

Eger `required_defense[j] = 0` ise defense flow'un faydasi yoktur. Bu durumda
o hedefe defense role ile gemi gondermek reward uretmez, hatta over-defense
cezasi alabilir.

### 7.3 Support Need

Support, defense'ten ayridir. Owned bir gezegene gemi gondermek su sebeple
mantikli olabilir:

```text
cephe ussu yapmak
sonraki attack wave icin staging yapmak
enemy influence altindaki frontline'i kalinlastirmak
compactness / response time iyilestirmek
```

Support ihtiyaci:

```text
support_need[j] =
    support_demand[j]
  * frontier_anchor_value[j]
  * (1 + enemy_influence[j])
  * future_attack_access[j]
```

Support defense degildir. `required_defense[j] = 0` olsa bile support flow
anlamli olabilir; fakat bunun icin `support_need[j]` pozitif olmalidir.

Over-support cezasi:

```text
over_support[j] =
  max(0, support_received[j] - support_need[j] - support_buffer[j])
```

### 7.4 Attack Need

Neutral/enemy hedefler icin:

```text
required_capture[j] =
    projected_target_garrison_at_arrival[j]
  + capture_margin[j]

attack_budget_worth[j] =
    attack_value[j]
  + expand_value[j]
  + w_diff * delta_diffusion_if_capture[j]
  + w_denial * enemy_denial_if_enemy_owned[j]
```

`projected_target_garrison_at_arrival` icin ETA semantigi kesin olmali:

```text
planned_capture_eta[j] = max effective ETA of selected attack wave to j
```

Enemy hedeflerde production capture anina kadar enemy icin calistigi icin
`min ETA` kullanmak yanlis/iyimserdir. Multi-source wave'de yavas source
eklenirse target daha uzun sure uretir ve demand artar.

Pratik allocator kurali:

```text
1. candidate attack plan uret
2. selected_eta = max(eta(src, j) for selected sources)
3. required_capture(j, selected_eta) yeniden hesapla
4. plan required_capture'i karsilamiyorsa:
   - ekstra source/ships ekle
   - veya attack planini iptal et
```

Neutral contested hedeflerde de tek "ships > neutral_ships" kuralina
guvenilmez; known enemy inbound varsa target timeline simulator ile
capture-and-hold kontrolu yapilir.

Attack flow'un amaci threshold'u tamamlamaktir:

```text
sum_i x_attack[i,j] ~= required_capture[j]
```

Undercommit ve overkill ikisi de maliyetlidir:

```text
attack_shortfall[j] = max(0, required_capture[j] - attack_received[j])
attack_overkill[j]  = max(0, attack_received[j] - required_capture[j] - margin)
```

Neutral capture icin enemy denial yoktur:

```text
enemy_denial_if_enemy_owned[j] =
  0                       if old_owner == neutral
  enemy_discounted_value  if old_owner is enemy
```

### 7.5 Evac Need

Evacuation icin:

```text
evac_need[i] =
  evacuation_urgency[i] * max(0, ships[i] - minimal_last_stand_reserve[i])
```

Source gercekten riskli degilse evacuation flow cezalandirilir:

```text
false_evac_cost =
  x_evac[i,j] * max(0, safe_source_confidence[i] - risk_threshold)
```

### 7.6 Objective

Allocator objective:

```text
maximize
    defense_coverage_value
  + support_position_value
  + capture_value
  + evacuation_value
  - travel_cost
  - source_strip_cost
  - over_defense_cost
  - over_support_cost
  - attack_shortfall_cost
  - attack_overkill_cost
  - false_evac_cost
```

Role-specific terms:

```text
defense_coverage_value =
  sum_j V_saved[j] * min(1, defense_received[j] / max(1, required_defense[j]))

support_position_value =
  sum_j support_received[j] * support_need[j] / (1 + eta_cost)

capture_value =
  sum_j capture_success_proxy[j] * attack_budget_worth[j]

evacuation_value =
  sum_i rescued_proxy[i] * evacuation_urgency[i]
```

Ilk implementasyonda tam MILP sart degil. Pratik alternatif:

```text
1. defense deficits'i once kapat
2. high-value attack thresholds icin bounded knapsack / min-cost-flow
3. support flow'u kalan budget ile doldur
4. evac urgency kritikse defense'ten once ele al
5. rounding ve max launch cap uygula
```

Ama uzun vadede role-aware min-cost-flow veya small LP daha temizdir.

---

## 8. Launch Builder

Allocator sonucu:

```text
FlowDecision:
  source_id
  target_id
  role
  ships
  allocator_value
  contributing_node_scores
```

Launch builder:

```text
for each FlowDecision:
  if ships < min_launch: skip
  action = compute_launch(source_id, target_id, ships)
  if action invalid:
    record invalid_flow_credit
    skip
  emit action
```

Direction/angle policy tarafindan secilmez. Aci, hedef hareketi, intercept,
gunes ve path collision kontrolleri deterministik `compute_launch` katmaninda
hesaplanir.

Training ve inference ayni postprocess'i kullanmali. Aksi halde train/test
mismatch olusur.

---

## 9. Reward Tasarimi

Reward sadece makro metrik iyilesmesi olmamali. Production-only deneyimlerinde
zayif sonuc alinmasinin sebebi muhtemelen production'in oyunun tamamini temsil
etmemesi:

- savunma zamanlamasini anlatmaz
- source strip riskini anlatmaz
- airborne ship kayiplarini anlatmaz
- rakibin counter-attack penceresini anlatmaz
- son turlarda production degeri hizla duser

Yeni reward yapisi:

```text
r_total =
    r_terminal
  + r_event_credit
  + r_small_potential_shaping
  - r_safety_penalty
```

### 9.1 Terminal Reward Ana Sinyal Olmali

```text
win:  +1.0
loss: -1.0
draw / tie: 0.0
turn-cap:
  z = clip((my_ships - best_other_ships) / total_ships, -1, 1)
```

Ana optimizasyon hedefi gercek oyunu kazanmak olmali. Shaping sinyalleri sadece
credit assignment'i kolaylastirir.

### 9.2 Capture Event Credit

```text
r_capture =
    + target_discounted_production_value
    + w_diff * diffusion_gain_after_capture
    + w_denial * enemy_denial_value_if_enemy_owned
    - ships_spent_cost
    - source_exposure_increase
```

Neutral capture:

```text
enemy_denial_value = 0
```

Enemy-owned capture:

```text
enemy_denial_value = target_discounted_production_value_for_enemy
```

### 9.3 Defense Event Credit

Savunma reward'i "ne kadar cok garrison korudum / yigdım" seklinde
verilmemeli. Bu, en ufak saldirida gezegene gemi yigma davranisini
odullendirir.

Dogru savunma credit'i tehdide orantili olmali:

```text
required_defense =
  enemy_arrival_power
  - projected_local_garrison_at_arrival
  - friendly_inbound_before_arrival
  + margin

effective_defense =
  min(defense_flow_arriving_before_enemy, max(0, required_defense))

over_defense =
  max(0, defense_flow_arriving_before_enemy - (required_defense + margin))
```

Gezegen threat window sonunda tutulduysa:

```text
r_defense_hold =
    V_saved * min(1, effective_defense / max(1, required_defense))
  - over_defense_cost
  - reinforcement_ship_cost
```

Gezegen dustuyse:

```text
r_defense_loss =
  - V_saved * failed_commitment_share
  - reinforcement_ship_cost
```

`preserved_garrison_value` yok. Native garrison zaten karar degildir; mutlak
garrison stokunu odullendirmek gemi yigmayi tesvik eder.

### 9.4 Support Event Credit

Support credit, defense credit'ten ayrilmali.

Support basarisi su sonuclara baglanabilir:

```text
frontier anchor daha sonra successful attack source oldu
support alan gezegen enemy pressure altinda daha iyi response verdi
future attack access / diffusion position iyilesti
support flow sonraki N turn icinde useful outgoing launch'a donustu
```

Ama support reward'i dikkatli tutulmali. Aksi halde ajan savunma ihtiyaci
olmayan gezegenlere "ileride lazim olur" diye gemi yigmaya baslar.

Yarisma V1 icin support credit:

```text
r_support =
    small_position_gain
  + future_use_credit_if_reused
  - over_support_cost
  - travel_cost
```

Support, defense yerine gecmemeli. Eger hedefte `required_defense = 0` ve
`support_need = 0` ise owned hedefe giden flow useless reinforcement sayilir.

### 9.5 Evac Event Credit

```text
r_evac =
    + rescued_ships_value
    + safe_destination_support_gain
    - abandoned_production_value
    - false_evac_cost
```

Evacuation credit sadece kaynak gercekten tehlikedeyse verilmeli. Aksi halde
ajan gereksiz gemi gezdirmeyi ogrenebilir.

### 9.6 Bad Launch / Safety Penalty

```text
r_bad =
    - ships_lost_to_sun_or_collision
    - failed_capture_overcommit
    - source_planet_lost_after_strip
    - invalid_compute_launch_flow
```

Bu penalty'ler allocator flow'una ve onu tetikleyen node skorlarina geri
yazilmali.

---

## 10. Gereksiz Defense vs Useful Support

Bu tasarimin kritik problemi:

```text
Bir owned gezegene gemi gitti.
Bu defense miydi, support muydu?
Gercekten gerekli miydi?
```

Cozum: role'u allocator icinde ayri tutmak.

```text
x_defense[i,j]  -> sadece required_defense acigini kapatir
x_support[i,j]  -> sadece support_need / frontier utility icin kullanilir
```

Eger:

```text
required_defense[j] = 0
support_need[j] = 0
```

ise `i -> j` owned flow'u gereksizdir:

```text
useless_owned_flow_penalty =
  x_defense[i,j] + x_support[i,j]
```

Eger:

```text
required_defense[j] = 0
support_need[j] > 0
```

ise flow defense degil support olarak kredilenir. Bu durumda credit, gezegenin
sonradan gercekten staging/support amaciyla kullanilip kullanilmamasina gore
verilir.

Eger:

```text
required_defense[j] > 0
```

ise once defense acigi kapatilir. Fazlasi support olarak sayilabilir, ama sadece
`support_need[j]` pozitifse. Aksi halde over-defense/over-support cezasina
gider.

Bu ayrim eski edge-only modelde belirsizdi. Node+allocator tasariminda role
decision env'e gitmese bile ledger'da korunur.

---

## 11. Credit Assignment

Yeni policy'nin karar birimleri:

```text
turn-level value
node strategic scores / sampled node intents
allocator flow decisions
multi-turn campaign outcomes
```

Allocator deterministic ise credit dogrudan LP'ye degil, LP'yi kosullayan node
policy kararlarina yazilir.

Ana prensip:

```text
reward outcome'a gore hesaplanir,
credit outcome'u tetikleyen node intents ve kullanilan flow'lara dagitilir.
```

### 11.1 Attribution Function ve Credit Conservation

Credit assignment sozlu kalmamali; her event icin once kanal bazli reward
hesaplanir, sonra bu kanal reward'u ilgili contributor'lara paylastirilir.

Genel form:

```text
Event e:
  R_e = terminal/event score degil, tek event'in lokal reward'u

Channel decomposition:
  R_e =
      R_core
    + R_hold
    + R_enable
    - R_cost
    - R_waste
    - R_source_strip
```

Her kanal icin conservation kuralini tercih ediyoruz:

```text
sum_i assigned_credit[channel, i] = R_channel
```

Bu sayede ayni event reward'u birden fazla head'e tam degerle yazilip sismez.
Value/global reward icin event bir kez sayilir:

```text
R_value += R_e
```

Policy head'leri icin ise kanal paylari kullanilir:

```text
attack_value[target]       <- R_core attack/capture payi
defense_demand[target]     <- R_hold / defense payi
support_demand[target]     <- capped R_enable payi
donor_capacity[source]     <- effective source contribution - source penalties
reserve_value[source]      <- source_strip / successful hold trade-off
evacuation_urgency[source] <- rescue + capped downstream donor payi
```

#### Capture Attribution

Stable capture icin:

```text
R_capture_event =
    R_capture_core
  + R_post_capture_hold
  + R_support_enable
  - R_ship_cost
  - R_true_overkill
  - R_source_strip
  - R_enemy_gift
```

Capture core sadece capture'a efektif katkida bulunan arrival'lara dagitilir:

```text
eligible_capture_fleets =
  fleets arriving before/at first self capture

share_f =
  effective_capture_ships_f / sum effective_capture_ships

credit_capture_core[f] =
  share_f * R_capture_core
```

Post-capture hold ayri kanaldir:

```text
eligible_hold_fleets =
  capture surplus + later arrivals that absorb known enemy inbound before H

credit_hold[f] =
  hold_effective_ships_f / sum hold_effective_ships * R_post_capture_hold
```

Eger bir support flow capture'i mumkun kildiysa full capture reward almaz:

```text
R_support_enable =
  support_enable_weight * min(downstream_success_credit, support_cap)
```

Bu kanal da kendi icinde support contributor'larina paylastirilir.

#### Defense Attribution

Defense icin her threat wave ayri sub-window olarak kredilenir:

```text
R_defense_wave =
    R_shortfall_closed
  + R_hold_if_planet_survived
  - R_over_after_threat_window
  - R_ship_cost
```

Paylastirma:

```text
share_f =
  min(ships_f, remaining_shortfall_when_f_arrives)
  / sum effective_shortfall_closing_ships
```

Ilk wave'de fazla gorunen gemi sonraki known wave'de gerekiyorsa over-defense
sayilmaz; over-defense sadece son known threat window sonrasinda faydasiz kalan
fazladir.

#### Source/Donor Attribution

Source credit ayri hesaplanir:

```text
donor_credit[source] =
    sum effective_flow_credit_from_source
  - source_strip_cost[source]
  - wasted_flow_cost[source]
```

Bir source incoming altindayken attack'a donor olduysa ve planned support ile
sonra tutulduysa donor credit pozitif olabilir. Aynisi support gelmedigi icin
source dustuyse source_strip penalty alir.

#### Conservation ve Auxiliary Loss Ayrimi

Ana event credit conservation'a uyar. Ancak asagidaki sinyaller auxiliary
olabilir ve event reward toplamindan ayri tutulabilir:

```text
semantic role mismatch penalty/logging
entropy/sparsity regularization
BC imitation loss
invalid action diagnostics
```

Bu auxiliary sinyaller value reward'u sismez; policy stabilitesi ve debug icin
kullanilir.

### 11.2 Node Intent Credit

Her node icin policy snapshot tutulur:

```text
NodeDecision:
  record_idx
  player
  turn
  node_id
  defense_demand
  support_demand
  attack_value
  expand_value
  donor_capacity
  reserve_value
  evacuation_urgency
  sampled_intents
  logprobs
```

Allocator bir node skorunu kullanirsa o skor outcome credit alir.

Ornek:

```text
target j defense_demand yuksek
allocator x_defense[:,j] > 0
gezegen threat window sonunda tutuldu
=> j.defense_demand pozitif credit alir
=> donor source'lar donor_capacity / reserve trade-off credit alir
```

Yanlis defense:

```text
j.defense_demand yuksek
required_defense[j] = 0
support_need[j] = 0
allocator yine x_defense[:,j] gonderdi
=> j.defense_demand negatif credit alir
=> source donor_capacity fazla agresifse negatif credit alir
```

### 11.3 Donor Credit

Donor credit ayri tutulmali, cunku savunma/saldiri hedefi dogru olsa bile
yanlis source secimi oyunu kaybettirebilir.

Donor source icin:

```text
positive:
  gonderdigi gemi capture/defense/evac sonucunda effective kullanildi

negative:
  source reserve ihlal edildi
  source launch sonrasi belirli pencere icinde dustu
  gonderilen gemi overkill / over-defense oldu
```

Donor credit:

```text
donor_credit[i] =
    effective_flow_value_from_i
  - source_strip_cost[i]
  - wasted_flow_cost[i]
```

### 11.4 Flow Credit

Allocator flow'u ledger'da saklanir:

```text
FlowDecision:
  record_idx
  player
  turn
  src_id
  tgt_id
  role
  ships
  expected_value
  source_node_decision_id
  target_node_decision_id
```

Flow credit, event sonucuna baglanir:

- attack flow capture'a katkida bulundu mu?
- defense flow tehditten once varip acigi kapatti mi?
- support flow sonraki N turn icinde useful outgoing launch'a donustu mu?
- evac flow gemileri source dusmeden once kurtardi mi?
- source bu flow yuzunden savunmasiz kalip dustu mu?

Flow credit daha sonra source ve target node kararlarina paylastirilir.

### 11.5 Actual Role Reclassification

Initial role/tag sadece policy intent'idir. Gercek credit, filonun timeline'da
hangi event'e efektif katkida bulunduguna gore yazilmalidir.

Ornek:

```text
t0: T hedefe attack wave A gonderildi
t1: T hedefe attack wave B gonderildi
t5: A geldi ve T capture oldu
t7: enemy inbound T'ye geldi
t8: B geldi
```

B policy tarafinda attack intent'iyle gonderilmis olabilir; fakat capture
zaten t5'te olduysa B capture credit almamalidir. Timeline sonucuna gore:

```text
fleet arrives before/at capture:
  eligible for attack capture credit

fleet arrives after self already owns target:
  reclassify as post-capture defense/support

fleet arrives after enemy recaptured target:
  maybe new recapture attack campaign

fleet arrives but no useful event occurs:
  ship cost / delay / waste
```

Bu nedenle ledger `planned_role` ve `actual_role` ayrimini tutmali:

```text
planned_role = policy/allocator intent
actual_role  = timeline event attribution
```

Credit `actual_role` uzerinden yazilir. Planned role ise semantik hata analizi
icin kullanilir. Aksi halde agent attack tag'iyle defense yapmayi ogrenebilir
ve rol semantigi bozulur.

Burada iki sinyal ayrilmali:

```text
causal credit:
  bu karar outcome'a faydali katkida bulundu mu?

semantic credit:
  planned_role ile actual_role uyumlu mu?
```

Ornek:

```text
planned_role = attack
actual_role = post_capture_defense
fleet target'i tutmaya yardim etti
```

Bu durumda:

```text
causal credit:
  sinirli pozitif alabilir, cunku filo faydali oldu

attack semantic credit:
  capture'a katkisi yoksa full attack credit almaz

post_capture_hold credit:
  actual contribution'a gore defense/support kanalina yazilir

semantic mismatch:
  logging veya kucuk auxiliary penalty olabilir
```

Bu ayrim PPO varyansini azaltir. Aksi halde attack intent'leri sik sik defense'e
reclassify olursa, planned-intent logprob'u ile actual-role reward arasindaki
korelasyon zayiflar ve gradient gurultulu olur.

### 11.6 Allocator Kullanmazsa Ne Olur?

Policy bir node'a yuksek skor verdi ama allocator kullanmadiysa iki durum var:

```text
1. Budget / constraints yuzunden kullanilmadi.
2. Skor pratikte dusuk degerliydi, objective'te kaybetti.
```

Yarisma V1'de kullanilmayan skor icin buyuk negatif credit vermeyelim. Bu PPO'yu
gurultulu yapar. Kullanilmayan intent icin:

```text
small entropy / sparsity regularization
BC auxiliary loss
value learning uzerinden dolayli sinyal
```

Buyuk credit sadece gercek flow urettiyse ve outcome biliniyorsa verilmeli.

### 11.7 PPO Logprob Meselesi

Allocator deterministic oldugu icin PPO'nun logprob'u allocator cozumunden
degil, policy'nin stochastic node intent kararlarindan gelmeli.

Pratik Yarisma V1:

```text
policy samples:
  active_defense_targets
  active_attack_targets
  active_support_targets
  active_evac_sources
  active_donors

allocator solves:
  flow among sampled/active nodes
```

PPO:

```text
L_policy =
  PPO(logprob_node_intents, node_intent_advantages)
```

Value head:

```text
R_value =
  terminal
  + sum event credits
  + small shaping
```

Bu sekilde RL action uzayi node seviyesinde kalir, allocator ise deterministic
lojistik cozumleyici olarak davranir.

---

## 12. Episode Ledger

Training sirasinda her episode icin lightweight ledger tutulmali.

```text
TurnSnapshot:
  turn
  owner[planet_id]
  ships[planet_id]
  production[planet_id]
  graph_metrics[planet_id]
  in_flight_fleets

NodeDecision:
  record_idx
  player
  turn
  node_id
  node_scores
  sampled_intents
  logprobs

FlowDecision:
  record_idx
  player
  turn
  src_id
  tgt_id
  planned_role
  actual_role optional
  ships
  source_node_decision_id
  target_node_decision_id
  campaign_id

LaunchRecord:
  fleet_id
  campaign_id
  src_id
  tgt_id
  planned_role
  actual_role
  ships
  launch_turn
  resolve_turn
  outcome
```

Ledger'in amaci post-episode offline attribution yapmaktir. Online step reward
yerine episode sonunda gercek outcome'lar bilindiginde credit karar birimlerine
yazilir.

---

## 13. Campaign Credit

Campaign, allocator flow'undan kurulur.

### 13.1 Attack Campaign

```text
AttackCampaign(target_id):
  contributors = all x_attack[i,target] flow decisions
```

Success:

```text
target owner -> self
```

Reward:

```text
R_attack =
    success * (
        V_target
      + w_diff * diffusion_gain
      + w_denial * enemy_denial_if_enemy_owned
    )
  - ship_cost
  - delay_cost
  - true_overkill_cost
  - source_loss_cost
  - fail_or_lost_race_penalty
```

Capture credit sadece capture'a gercek katkida bulunan launchlara dagitilir.
Capture'dan sonra gelen fleet'ler attack credit almamali; owned hedefe geliyorsa
support/defense olarak yeniden siniflanabilir veya void olabilir.

Overkill her zaman kotu degildir. Capture sonrasi kalan fazla gemi:

```text
capture surplus
  -> post-capture hold buffer
  -> bilinen enemy inbound'a karsi savunma
  -> sonraki attack icin staging
  -> gercekten gereksiz overkill
```

Bu nedenle sadece `true_overkill` cezalandirilir:

```text
true_overkill =
    surplus_after_capture
  - required_hold_buffer_until_H
  - planned_or_realized_reuse_within_N_turns
```

Eger fazla gemi gezegeni tutmayi sagladiysa veya kisa sure icinde yeni faydali
attack/defense/support aksiyonuna kaynak olduysa attack overkill penalty
almamali; post-capture hold veya staging credit alabilir.

### 13.2 Defense Campaign

```text
DefenseCampaign(target_id):
  contributors = all x_defense[i,target] flow decisions
```

Defense credit'in amaci:

```text
tehdit acigini gerekli oranda kapatmak
```

Degil:

```text
gezegendeki gemi stokunu maksimize etmek
```

Credit, `required_defense` acigini zamaninda kapatan flow'lara paylastirilir.
Native garrison ve per-turn production credit almaz; onlar karar degildir.

Defense campaign tek arrival ile otomatik bitmemeli. Dushman dalgalar halinde
saldiriyorsa campaign rolling threat window olarak tutulur.

Baslangic:

```text
enemy inbound within H
veya biz target'a defense/support flow gonderdik
```

Devam:

```text
enemy arrivals arasi gap <= quiet_gap
veya hala known enemy inbound var
veya target critical frontier ve hold window bitmedi
```

Bitis:

```text
planet lost
veya no enemy inbound within H
ve last combat uzerinden quiet_gap gecti
veya episode end
```

Campaign icinde her enemy wave icin ayri sub-window hesaplanir:

```text
before wave k:
  projected_garrison
  incoming_enemy_k
  required_defense_k

after wave k:
  held mi?
  hangi reinforcement shortfall kapatti?
```

Bu ayrim onemli: ilk wave'den sonra fazla gorunen gemi, ikinci wave icin
gerekliyse `over_defense` sayilmaz.

```text
over_defense =
    excess_after_last_known_threat_window
  - useful_support_or_staging_value
```

### 13.3 Support Campaign

```text
SupportCampaign(target_id):
  contributors = all x_support[i,target] flow decisions
```

Support credit icin Yarisma V1'de dar kural:

```text
support flow, sonraki N turn icinde useful outgoing attack/defense/evac flow'a
kaynak olduysa credit alir.
```

Bu kural support'un "ileride lazim olur" bahanesiyle sinirsiz gemi yigmasini
engeller. Support position shaping dusuk agirlikli kalmali.

Support credit icin counterfactual dikkatli kurulmalidir. Temel test:

```text
with_support outcome
without_support outcome
```

Yarisma V1 approximation:

```text
support_effective =
  support arrived before critical event
  and without_support metric fails
  and with_support metric succeeds
```

Bu counterfactual full-game counterfactual degildir. V1'de target/source-local
timeline simulator ile hesaplanir:

```text
with_support:
  ledger'daki gercek/planned support event'leri dahil

without_support:
  ayni timeline, ilgili support event'leri cikarilmis
```

Rakibin future policy'si yeniden optimize edilmez; known inbound ve planned
flow'lar sabit kabul edilir. Bu biased bir approximation'dir. Bu yuzden support
credit:

```text
small
capped
yalnizca pivotal oldugunda aktif
```

olmali. Counterfactual bias'i azaltmak icin support'un downstream reward'un
tamamini almasina izin verilmez.

Ornek effective support turleri:

```text
Defense enabling:
  without support: gezegen dusuyor
  with support: gezegen tutuluyor

Donor enabling:
  without support: source attack sonrasi enemy arrival'da dusuyor
  with support: source hem attack yapiyor hem tutuluyor

Attack staging:
  support alan gezegen sonraki N turn icinde successful attack source oluyor
```

Support'a downstream event'in tamamini vermek double count yaratir. Credit
capped olmalidir:

```text
support_credit =
  support_enable_weight * min(downstream_success_credit, support_cap)
```

Support full attack/defense reward'u calmaz; sadece o basariyi mumkun kildigi
oranda sinirli enabling credit alir.

### 13.4 Evac Campaign

```text
EvacCampaign(source_id):
  contributors = all x_evac[source,target] flow decisions
```

Credit sadece source gercekten riskliyse verilmeli:

```text
source lost within evac_window
owned comet expires/leaves board
source becomes unsavable by defense estimate
```

False evac kucuk ama net maliyet alir.

Evac sadece "gemi kurtardim" degildir; destination da onemlidir.

```text
rescue_credit:
  source gercekten dusecek/comet gidecekse kurtarilan gemi

destination_credit:
  kurtarilan gemiler guvenli ve faydali destination'a gittiyse
```

Evac edilen gemiler destination'a ulasinca normal donor pool'a girer. Eger daha
sonra attack/defense/support icin kullanilirsa evacuation decision kucuk,
capped downstream credit alabilir:

```text
evac -> destination -> later useful donor
```

Evac destination kotu secildiyse:

```text
safe olmayan hedef
hemen dusen hedef
sonraki faydali flow'a donusmeyen gereksiz tasima
```

evac credit dusurulur veya false evac/opportunity cost yazilir.

---

## 14. Double Count ve Metric Inflation Kurallari

Reward tasariminda en buyuk risk ayni stratejik olguyu birden fazla metrikle
tekrar saymak. Bu, ajanin gercek oyun yerine metrik sisirmeyi ogrenmesine yol
acar.

Kesin kurallar:

```text
1. Capture V_target veriliyorsa ayni ownership flip icin buyuk production delta verme.
2. Defense hold credit veriliyorsa preserved_garrison reward verme.
3. Diffusion gain capture credit'e ekleniyorsa clip + kucuk katsayi kullan.
4. Potential -> realized donusumunu tek basina reward etme.
5. No-op/pass generic reward verme.
6. Source loss cezasini hem source_loss hem exposed_delta ile buyuk sayma.
7. Neutral capture icin enemy denial verme.
8. Support credit'i defense credit'le ayni event'ten iki kez verme.
9. Capture surplus useful hold/staging olduysa overkill penalty verme.
10. Support/evac downstream credit'i full downstream reward degil capped enabling credit olsun.
11. Planned role yerine actual timeline contribution uzerinden credit yaz.
```

### 14.1 Production Double Count

Su metrikler production bilgisini farkli bicimlerde tasir:

```text
raw production
discounted_production_value
diffusion_potential
production_margin_delta
capture target value
enemy denial value
diffusion gain
```

Feature olarak hepsi verilebilir. Reward'da production icin tek ana kanal
secilmeli.

### 14.2 Threat / Support Double Count

Su metrikler ayni askeri baskiyi farkli bicimde temsil eder:

```text
incoming_enemy_ships
potential_enemy_threat
enemy_flux
defense_deficit
exposed_owned_value
```

Reward'da savunma icin event credit ana kanal olmali. Threat/support delta
sadece dusuk agirlikli shaping olmali.

### 14.3 Realized ve Potential Ayni Anda Sayilmasin

Potential:

```text
gezegendeki garnizon firlatilabilir
```

Realized:

```text
filo zaten yolda
```

Potential -> realized donusumu tek basina reward edilmemeli. Sonuc reward
edilmeli:

```text
hedef tutuldu mu?
capture oldu mu?
gemi kurtuldu mu?
support flow sonraki faydali hamleye donustu mu?
```

---

## 15. Training Plani

### 15.1 BC Warm Start

Teacher ajanlar:

```text
GA+MCTS
timed_mcts
scripted orbit
farkli aggression/reserve configleri
```

Teacher launch'lari node-level label'a cevrilir:

```text
source launched useful ships
  -> donor_capacity label

owned target received needed reinforcement
  -> defense_demand label

owned target received staging ships later reused
  -> support_demand label

neutral/enemy target attacked
  -> attack_value / expand_value label

source evacuated before loss/comet expiry
  -> evacuation_urgency label
```

BC loss:

```text
L_BC =
    BCE/CE for sampled node intents
  + regression/ranking loss for node strategic scores
  + optional allocator imitation loss on flow matrix
```

BC'nin amaci teacher'i final olarak kopyalamak degil, random policy'nin ilk
asamada tamamen anlamsiz node intents uretmesini engellemektir.

### 15.2 Real-Env PPO

Ana RL asamasi gercek `orbit_wars` env uzerinde olmali. Soyut model yardimci
olabilir ama final policy su fiziklerle egitilmeli:

- rotating planets
- comet spawn/expiration
- sun collision
- moving planet sweep collision
- simultaneous combat
- 500 turn economy

PPO action log-prob'u sampled node intents uzerinden hesaplanir. Allocator
deterministic oldugu icin logprob'a dahil edilmez; ancak allocator'in urettigi
flow outcome credit'i node intents'e geri yazilir.

Credit/shaping sinyalleri bilincli bir bias-variance trade-off'tur:

```text
terminal win/loss:
  unbiased ana hedef, fakat 500 turn horizon nedeniyle sparse ve yuksek varyansli

event/campaign credit:
  biased yardimci sinyal, fakat hangi karar faydaliydi sorusuna daha dusuk
  varyansli cevap verir

potential shaping:
  mumkunse Phi(s_next) - Phi(s) formunda tutulur
  degilse dusuk agirlikli ve clipped olur
```

Bu nedenle event credit value reward'u tamamen degistiren ana objective degil,
terminal hedefe ulasmayi kolaylastiran attribution sinyalidir. Event credit
agirliklari buyudukce ajan planner metriklerini oynamayi ogrenebilir; bu risk
paired evaluation ve ablation ile kontrol edilmelidir.

### 15.3 Opponent Pool

Sadece self-play collapse riski tasir. Havuz:

```text
GA+MCTS
timed_mcts
scripted orbit
onceki checkpointler
farkli scripted config varyantlari
```

Training sirasinda opponent sampling:

```text
50% latest/self
25% frozen checkpoint pool
25% scripted/GA baselines
```

---

## 16. Yarisma V1 Final Hatti

Yarismanin bitmesine az zaman kaldigi icin full RL/GNN egitimini final riski
yapmiyoruz. Node+allocator+ledger tasarimi uzun vadeli ana yon olarak kalir;
yarismaya yetisecek uygulanabilir V1 ise mevcut GA/scripted hattin uzerine
attack micro-planner eklemektir.

Final V1:

```text
mevcut GA/scripted/MCTS tabani
  + contested target timeline simulator
  + target-local micro MCTS veya beam planner
  + GA ile planner agirlik tuning
```

Bu karar pragmatik:

- Kucuk model egitmek 5 gun kala fazla riskli.
- Full RL egitimi icin yeterli deneme/ablation suresi yok.
- Mevcut en iyi ajan zaten GA ayarli scripted/MCTS hattidir.
- En buyuk zayifliklardan biri attack feasibility ve contested target
  degerlendirmesidir; bunu micro-planner ile hedefli sekilde iyilestirebiliriz.

### 16.1 Attack Timeline Simulator

Capture karari tek formulle verilmemeli. Ozellikle neutral hedeflerde iki taraf
ayni gezegene filo gonderiyorsa ilk capture yeterli degildir; capture sonrasi
elde tutma da hesaplanmali.

Target-local simulator input:

```text
target T
current owner / ships / production
known own inbound fleets to T
known enemy inbound fleets to T
planned own attack flows
optional estimated enemy counter flows
hold horizon H
```

Timeline event'leri:

```text
ArrivalEvent:
  eta
  owner
  ships
  source_id optional
  known_or_planned
```

Simulator:

```text
events = sorted(arrivals_to_T, key=eta)
owner = current_owner
garrison = current_ships
last_eta = 0

for event in events:
  if owner != neutral:
    garrison += production * (event.eta - last_eta)

  if event.owner == owner:
    garrison += event.ships
  else:
    if event.ships > garrison:
      owner = event.owner
      garrison = event.ships - garrison
    else:
      garrison -= event.ships

  last_eta = event.eta
```

Output:

```text
first_capture_by_self
self_capture_turn
self_surplus_after_capture
self_holds_until_H
temporary_capture_then_lost
soften_then_enemy_capture
enemy_recapture_margin
stable_hold_turns
```

Bu simulator iki amacla kullanilir:

```text
Yarisma V1:
  attack micro-planner icin capture-and-hold feasibility oracle

Uzun vadeli RL:
  attack/defense/support/evac credit assignment icin local counterfactual oracle
```

Credit oracle olarak kullanildiginda simulator full-game arama yapmaz. Sadece
ilgili target/source timeline'ini, known inbound fleet'leri ve ledger'daki
planned flow'lari simule eder. Rakibin yeni counter-response uretmesi modelin
disinda kalir; bu nedenle simulator credit'i biased ama dusuk varyansli yardimci
sinyaldir. Terminal win/loss ana hedef olarak kalir.

### 16.2 Contested Neutral Hesabi

Neutral hedef icin basit capture ihtiyaci:

```text
ships_sent > neutral_ships
```

Ama contested neutral icin bu yetersizdir. Ornek:

```text
neutral ships = N
self arrival = A ships at t_self
enemy arrival = E ships at t_enemy
t_self < t_enemy
```

Biz capture ettikten sonra kalan:

```text
surplus_after_capture = A - N
```

Dushman gelene kadar uretim:

```text
garrison_at_enemy_arrival =
  surplus_after_capture
  + production * (t_enemy - t_self)
  + friendly_reinforcement_before_enemy
```

Enemy recapture:

```text
enemy_recaptures = E > garrison_at_enemy_arrival
```

Bu nedenle neutral icin attack demand:

```text
capture_and_hold_need =
    neutral_ships
  + max(0, enemy_pressure_before_H - production_until_pressure - friendly_support)
  + margin
```

Bu formulle approximate edilebilir, ama final karar timeline simulator ile
dogrulanmali.

Reward/score ayrimi:

```text
stable_capture:
  full capture value

temporary_capture_then_lost:
  reduced value veya negatif, hold time'a bagli

soften_then_enemy_capture:
  negatif enemy gift penalty
```

Neutral target'ta enemy denial yoktur. Fakat neutral'i zayiflatip rakibin
almasini kolaylastirmak ayri bir kotu event'tir:

```text
enemy_gift_penalty =
  enemy captures target shortly after our failed/temporary attack
```

### 16.3 Enemy Target Capture Hesabi

Enemy hedefte production capture anina kadar enemy icin calisir. Bu yuzden
capture cost ETA'ya baglidir.

Dogru demand:

```text
required_capture =
    target_current_ships
  + target_production * planned_capture_eta
  + enemy_inbound_before_capture_eta
  - friendly_attack_inbound_before_capture_eta
  + capture_margin
```

Kritik karar:

```text
planned_capture_eta = selected wave'in max effective ETA'si
```

`min ETA` kullanmak enemy production'i eksik sayar. Multi-source wave'de yavas
source eklenirse hedef daha uzun sure uretir; demand yeniden hesaplanmalidir.

LP/beam tarafinda pratik cozum:

```text
1. candidate plan uret
2. actual_eta = max(selected launch ETA)
3. required_capture(actual_eta) yeniden hesapla
4. plan yetmiyorsa:
   - ekstra source ekle
   - amount artir
   - veya attack'i iptal et
```

Enemy target value:

```text
enemy_capture_value =
    own_future_production_gain
  + enemy_future_production_denial * denial_weight
  + small_diffusion_gain
  - ship_cost
  - source_strip_cost
```

Neutral target value:

```text
neutral_capture_value =
    own_future_production_gain
  + small_diffusion_gain
  - ship_cost
  - enemy_gift_risk
```

### 16.4 Micro MCTS / Beam Planner

Micro planner global oyun MCTS'i degildir. Sadece tek hedef icin su soruyu
cozer:

```text
Bu hedefe hangi wave ile girersem capture-and-hold olur?
```

Input:

```text
target T
candidate donor sources S_1..S_k
known inbound own/enemy fleets
source budgets
ETA(source, T)
source strip risk
target value
hold horizon H
```

Action uzayi:

```text
add_launch(source=i, ships=k)
stop
```

Amount adaylari continuous olmamali:

```text
min_send
needed_to_capture
needed_to_capture_and_hold
source_budget * 0.25
source_budget * 0.5
source_budget
```

Her partial/final plan timeline simulator'dan gecirilir.

Plan score:

```text
score =
    stable_capture_value
  + enemy_denial_if_enemy_owned
  + small_diffusion_gain
  - ships_sent_cost
  - source_strip_cost
  - delay_cost
  - overkill_cost
  - enemy_gift_risk
```

5 gun kala ilk implementasyon icin beam search, MCTS'ten daha kontrollu ve
debug edilebilir:

```text
for target in top_K_attack_targets:
  sources = top_M reachable donors by eta + strip_cost
  beam = [empty_plan]

  for depth in 1..max_launches_per_target:
    expand each plan with source+amount candidates
    simulate target timeline
    score partial/final plan
    keep top_B plans

  choose best plan with:
    self captures
    self holds until H
    score > 0
```

MCTS varyanti:

```text
state:
  target timeline
  remaining source budgets
  planned arrivals

action:
  add source+amount launch
  stop

rollout/eval:
  deterministic timeline simulation + score formula
```

MCTS'in avantaji daha fazla kombinasyon arayabilmesidir. Beam'in avantaji daha
deterministik ve debug edilebilir olmasidir. Final icin once beam, zaman kalirsa
MCTS denenebilir.

### 16.5 Defense ile Siralama

Attack planner, savunma butcesini tamamen yok etmemeli. Yarisma V1 icin en
guvenli sira:

```text
1. urgent savable defense budget'ini ayir
2. kalan donor budget ile attack micro-planner calistir
3. attack sonrasi source strip kontrolunu tekrar yap
4. support/consolidation en sona birak
```

Joint defense+attack optimization daha dogru olabilir, ama daha karmasiktir.
5 gun kala once bu sirali cozum uygulanmali.

Kural:

```text
attack plan, source gezegeni kritik savunma deficit'ine dusuruyorsa iptal edilir
veya source strip cost ile score negatif olur.
```

Fakat "incoming altinda" olmak tek basina source'u kilitlememeli. Eger source
enemy filosu gelmeden once baska bir gezegenden support alabiliyorsa, attack
planner bu source'u donor olarak kullanabilir:

```text
enemy_eta(source) > support_eta_to_source
ve
projected_garrison_at_enemy_arrival_after_attack >= enemy_power + margin
```

Ornek iyi plan:

```text
P1 -> enemy/neutral target attack
P2 -> P1 support, enemy P1'e ulasmadan once variyor
P1 enemy arrival aninda hala tutuluyor
```

Bu durumda P1 hem `defense_demand` hem `donor_capacity` alabilir. Bu iki head
hard mask ile birbirini dislamamali; allocator projected timeline ile karar
vermelidir.

### 16.6 GA ile Tuning

GA yeni bir policy ogrenmek icin degil, micro planner ve allocator agirliklarini
tune etmek icin kullanilmali.

Tunable parametreler:

```text
capture_value_weight
enemy_denial_weight
diffusion_gain_weight
hold_horizon
source_strip_weight
overkill_weight
delay_weight
enemy_gift_penalty
defense_priority_weight
support_priority_weight
risk_margin
capture_margin
max_attack_targets
top_M_sources
beam_width
max_launches_per_target
```

GA objective:

```text
paired seed winrate
+ final ship margin small weight
- p99 latency penalty
- illegal/drop penalty
- source-lost-after-launch penalty
- enemy-gifted-neutral penalty
```

Degerlendirme mutlaka paired seed + seat swap ile yapilmali. Aksi halde seed
sansina gore agirliklar overfit olabilir.

---

## 17. RL/Node+Allocator Scope

Ilk RL surumu dar tutulmali.

### Aktif Node Heads

```text
defense_demand
support_demand
attack_value / expand_value
donor_capacity
reserve_value
evacuation_urgency
```

### Aktif Allocator Role'leri

```text
attack
defense
support
evac
```

### Baslangicta Kapali veya Dusuk Agirlikli

```text
full MILP with integer exactness
fleet-as-node graph
multi-turn synchronized attack planner
generic no-op reserve credit
generic preserved garrison reward
large diffusion shaping
unbounded support/consolidation reward
```

### Basari Kriteri

```text
GA+MCTS'e karsi paired seed winrate artmali
timed_mcts'e karsi geriye dusmemeli
p99 act latency < 750ms olmali
illegal/drop rate dusuk olmali
source-lost-after-launch metriği kontrol altinda olmali
over-defense / over-support metriği kontrol altinda olmali
```

---

## 18. Yarisma V1 Implementasyon Sirasi

Bu sira full RL egitiminden once, yarismaya yetisecek final ajan icindir.

1. Target timeline simulator:
   - target owner/garrison production ilerletme
   - own/enemy inbound event siralama
   - temporary capture / stable hold / enemy gift ciktilari

2. Capture-and-hold evaluator:
   - neutral contested target hesabı
   - enemy target `production * max selected ETA` hesabı
   - post-capture hold horizon
   - enemy recapture margin

3. Beam micro-planner:
   - top-K attack target secimi
   - top-M donor source secimi
   - incoming altinda ama support ile guvenli kalabilen donor source'lari acik tut
   - source+amount candidate expansion
   - timeline simulator ile plan score
   - max launch cap ve source budget kontrolu

4. Mevcut scripted/GA attack phase entegrasyonu:
   - eski `allocate_greedy_captures` yerine veya oncesinde micro-planner
   - fallback olarak eski allocator
   - invalid launch / no positive plan durumunda pass

5. Defense-first budget entegrasyonu:
   - urgent savable defense budget'i ayir
   - attack planner'a kalan donor budget ver
   - planned support sonrasi projected safe surplus'u tekrar hesapla
   - attack sonrasi source strip check

6. Logging:
   - simulated stable capture vs actual capture
   - enemy-gifted-neutral count
   - temporary capture loss count
   - source-lost-after-launch
   - planner latency

7. GA tuning:
   - micro planner weights
   - beam width / top-M / hold horizon
   - paired seed + seat swap benchmark

8. Regression tests:
   - neutral contested timeline
   - enemy production until max ETA
   - enemy inbound recapture after neutral capture
   - incoming altindaki source'un timely support ile donor olabilmesi
   - source strip cancellation
   - fallback behavior

---

## 19. Uzun Vadeli RL Implementasyon Sirasi

1. `GraphPolicyObservation` tasarimi:
   - node features
   - edge features
   - masks
   - id/slot mapping

2. Metric builder:
   - threat/support
   - diffusion
   - discounted production
   - source strip risk
   - delta diffusion approximations

3. GNN model:
   - node encoder
   - edge encoder
   - message passing
   - node strategic heads
   - global value head

4. Allocator:
   - source budgets
   - reserve constraints
   - defense need
   - support need
   - attack threshold
   - evac need
   - flow rounding

5. Launch builder:
   - `compute_launch`
   - invalid launch logging
   - max launches cap

6. Ledger:
   - node decisions
   - flow decisions
   - launch records
   - campaign outcomes

7. Credit attribution module:
   - event/channel decomposition
   - credit conservation checks
   - planned_role -> actual_role reclassification
   - causal credit vs semantic mismatch ayrimi
   - support/evac local counterfactual oracle
   - source strip attribution

8. Teacher rollout recorder:
   - GA+MCTS/scripted actions -> node intent labels
   - optional flow labels

9. BC training.

10. Real-env PPO training.

11. Evaluation harness:
   - paired seeds
   - seat swap
   - winrate
   - p99 latency
   - illegal action/drop metrics
   - over-defense / over-support metrics
   - attribution conservation diagnostics
   - planned_role vs actual_role mismatch rate

---

## 20. Ana Tasarim Karari

Makro metrikleri reward'in kendisi yapmiyoruz.

Dogru ayrim:

```text
Network input:
  metrikler zengin ve ayrintili verilir.

Node policy:
  metriklerden stratejik niyet ve oncelik uretir.

Allocator:
  bu niyetleri kullanarak gemi dagilimini cozer.

Reward:
  terminal outcome ana sinyal.
  event credit ikinci sinyal.
  macro metric delta dusuk agirlikli shaping.
```

Bu ayrim, eski production-only deneyiminin problemini azaltir. Production,
diffusion, threat ve support metrikleri policy'ye guclu bir koordinat sistemi
verir; ama ajan sadece bu metrikleri sisirerek degil, gercek oyunu kazanarak
optimize edilir.

Final prensip:

```text
RL stratejiyi secer.
Allocator lojistigi cozer.
Ledger sonucu stratejiye geri yazar.
Reward metrik sisirmeyi degil, oyun kazanmayi destekler.
```

---

## 21. Mevcut Final Implementasyon Haritasi

Bu bolum, repodaki `final_*` dosyalarinin hangi parcayi gercekledigi ve hangi
komutla calistirilacagini tarif eder.

### 21.1 Deterministic Final Micro Agent

Bu hat torch/checkpoint gerektirmez ve Kaggle submission icin paketlenebilir.

Dosyalar:

- `final_timeline.py`
  - target-local timeline simulator
  - neutral/enemy/self arrival resolution
  - enemy production until arrival
  - contested neutral capture -> enemy recapture durumlarini simule eder

- `final_micro_planner.py`
  - beam search ile target-local capture-and-hold plan uretir
  - source budget, ETA, ship cost, source strip, true overkill ve enemy gift
    cezalarini hesaba katar
  - stable capture yapan planlari sadece `score > 0` olmadigi icin elemez;
    skor negatif olsa bile plan capture-and-hold acisindan gecerliyse allocator
    tarafina aday olabilir
  - enemy hedeflerde amount adaylari sadece anlik garrison'a degil, ETA boyunca
    enemy production'a gore de uretilir

- `final_agent.py`
  - defense-first deterministic agent
  - kritik ve strained savunma icin all-or-nothing reinforcement
  - saldiri/expansion icin micro planner
  - defense/attack launch sonrasinda source strip risk snapshot'ini kalan
    budget'e gore yaklasik gunceller

- `final_train.py`
  - neural RL degil
  - deterministic micro planner agirliklarini GA ile tune eder

- `final_eval.py`
  - GA genome ile deterministic final agent'i paired seed / seat swap degerlendirir

- `final_submission_agent.py`
  - Kaggle-style `agent(obs, config=None)` entrypoint

- `scripts/build_final_submission.py`
  - final deterministic micro agent bundle olusturur
  - `networkx` importlarini `minigraph` shim ile patch eder
  - smoke-test yapar

Calistirma:

```bash
../orbit_wars_env/bin/python final_train.py \
  --generations 8 --population 12 --games-per-eval 2 \
  --max-steps 300 --opponent orbit \
  --out ga_runs/final_micro/best_genome.json
```

```bash
../orbit_wars_env/bin/python final_eval.py \
  --genome ga_runs/final_micro/best_genome.json \
  --opponent orbit --games 20 --max-steps 300
```

```bash
../orbit_wars_env/bin/python scripts/build_final_submission.py \
  --games 1 --max-steps 80
```

Output:

```text
dist/submission_final_micro.tar.gz
```

### 21.2 Neural Final GNN PPO Agent

Bu hat yeni RL/GNN deneyleri icindir. Policy artik node-only degil,
edge-aware dense message passing GNN'dir.

Dosyalar:

- `final_obs.py`
  - padded planet table uretir
  - node features: ownership, ships, production, geometry, orbit, incoming fleet mass
  - edge features: reachable, ETA, distance, direction, source/target ownership,
    source ships, target ships, target production
  - masks: valid node, valid directed edge, source, target
  - global production feature'lari node doldurulmadan once hesaplanir; boylece
    slot sirasi kaynakli prefix-sum bug'i yoktur
  - incoming fleet feature'lari `predict_all_fleet_targets` ile tahmini hedefe
    yazilir

- `final_model.py`
  - `FinalNodePolicy`
  - node encoder + edge encoder + message passing layers
  - strategic node heads:
    - defense_demand
    - support_demand
    - attack_value
    - expand_value
    - donor_capacity
    - reserve_value
    - evacuation_urgency
    - frontier_anchor_value
  - sampled action:
    - target slot
    - source slot
    - amount fraction
  - value head

Not: mevcut calisan PPO implementation'i tam node-tag + role-aware allocator
mimarisi degildir. Model strategic head'ler uretse de training action'i su anda:

```text
target_idx
source_idx
amount_fraction
```

Bu nedenle "yanlis tag cezasi" su an dogrudan uygulanmaz. Calisan sistemde
cezalandirilan sey tag mismatch degil, secilen source/target/amount niyetinin
validation sonucudur.

- `final_rollout.py`
  - self-play rollout
  - production-share delta shaping
  - terminal reward
  - PPO clipped update
  - policy action -> launch conversion
  - non-owned targets icin micro planner gate
  - non-owned target icin policy'nin sectigi source tek donor olarak zorlanmaz;
    secili source oncelikli anchor kabul edilir, planner top-M owned donor ile
    multi-source capture-and-hold cozer
  - amount fraction donor budget scale olarak kullanilir
  - micro planner non-owned hedefte stable capture bulamazsa weak raw launch
    gondermez; no-op olarak loglar
  - planner horizon'i training truncate `max_steps` yerine env episode horizon'i
    ile uyumlu tutulur
  - oyun `max_steps` yuzunden truncate olursa terminal fallback olarak
    production-share margin reward'a eklenir

- `final_train_rl.py`
  - full neural PPO training script
  - checkpoint save/resume

- `final_rl_controller.py`
  - trained checkpoint'i match/eval controller olarak kullanir

- `final_eval_rl.py`
  - neural checkpoint'i `orbit_match` uzerinden degerlendirir

Full training:

```bash
../orbit_wars_env/bin/python final_train_rl.py \
  --iterations 100 --episodes-per-iter 2 --max-steps 150 \
  --num-agents 2 --d-model 128 --n-heads 4 --n-layers 2 \
  --lr 3e-4 --ppo-epochs 2 --minibatch-size 128 \
  --entropy-coef 0.02 --shaping-weight 0.05 --terminal-weight 1.0 \
  --invalid-action-penalty 0.02 \
  --reject-action-penalty 0.005 \
  --capture-action-bonus 0.002 \
  --planner-top-m-sources 5 \
  --planner-max-launches-per-target 3 \
  --planner-budget-floor 0.35 \
  --truncate-terminal-weight 1.0 \
  --checkpoint-every 1 --checkpoint-dir checkpoints_final_rl
```

Resume:

```bash
../orbit_wars_env/bin/python final_train_rl.py \
  --resume-from checkpoints_final_rl/latest.pt \
  --iterations 400 --checkpoint-dir checkpoints_final_rl
```

Eval:

```bash
../orbit_wars_env/bin/python final_eval_rl.py \
  --ckpt checkpoints_final_rl/latest.pt \
  --opponent orbit --games 20 --max-steps 300
```

Smoke test:

```bash
../orbit_wars_env/bin/python final_train_rl.py \
  --iterations 1 --episodes-per-iter 1 --max-steps 8 \
  --checkpoint-dir /tmp/final_rl_gnn_smoke \
  --minibatch-size 16 --ppo-epochs 1 --seed 6
```

### 21.3 Testler

```bash
../orbit_wars_env/bin/python -m pytest \
  tests/test_final_timeline.py \
  tests/test_final_obs.py \
  tests/test_final_micro_planner.py \
  tests/test_final_model.py \
  tests/test_final_rollout.py -q
```

Kapsam:

- contested neutral timeline
- enemy target production until arrival
- enemy inbound recapture
- node global production feature'larinin prefix-sum olmamasi
- predicted incoming fleet mass feature'i
- micro planner stable capture
- negatif skorlu ama stable capture yapan planin elenmemesi
- GNN policy action/evaluate finite checks
- action decoder'in selected target icin multi-source capture uretebilmesi
- empty mask fallback
- PPO single-item advantage normalization

### 21.4 Full Game Logging ve Replay

Tum oyunu bastan sona debug etmek icin:

```bash
../orbit_wars_env/bin/python scripts/final_game_recorder.py record \
  --agent final_rl --ckpt checkpoints_final_rl/latest.pt \
  --opponent orbit --games 2 --max-steps 300 \
  --out-dir game_logs --prefix final_rl_vs_orbit --html
```

Bu komut her oyun icin JSON log yazar:

```text
game_logs/final_rl_vs_orbit_000_p0_seed1000.json
game_logs/final_rl_vs_orbit_000_p1_seed1000.json
...
```

`--html` verilirse ayni oyunun HTML replay'i de yazilir.

JSON log icerigi:

- initial state
- her turn icin state_before
- player action listesi
- policy/controller debug reason
- state_after
- reward/status
- production/ship summary
- final rewards
- reason_counts
- Kaggle `steps` replay payload

Terminalde oynatmak:

```bash
../orbit_wars_env/bin/python scripts/final_game_recorder.py replay \
  --log game_logs/final_rl_vs_orbit_000_p0_seed1000.json \
  --sleep 0.05 --debug
```

Kayitli JSON'dan HTML tekrar render etmek:

```bash
../orbit_wars_env/bin/python scripts/final_game_recorder.py render \
  --log game_logs/final_rl_vs_orbit_000_p0_seed1000.json
```

Deterministic micro-agent icin:

```bash
../orbit_wars_env/bin/python scripts/final_game_recorder.py record \
  --agent final_micro --opponent orbit --games 2 \
  --max-steps 300 --out-dir game_logs --prefix final_micro_vs_orbit --html
```

Bu arac training loop'un parcasi degil; eval/debug maclarini kaydeder. Training
sirasinda her episode'u kaydetmek istenirse ayni format `collect_episode`
icinden opsiyonel hook olarak baglanabilir, ama bu ciddi disk ve runtime maliyeti
yaratir.

### 21.5 Son Duzeltme Kararlari

Bu turda, egitim kosmadan once reward/action noise uretecek net hatalar
kapatildi.

1. Micro-planner stable plan filtresi:

Eski davranis:

```text
valid plan = score > 0 and stable_hold
```

Bu yanlisti, cunku score agirliklari GA/RL tarafindan degisebilir. Pahali ama
stratejik olarak gerekli bir capture negatif skor alabilir. Capture-and-hold
gecerliligi ile skor esigi ayni sey degildir.

Yeni davranis:

```text
valid plan = score >= min_plan_score and stable_hold
```

`min_plan_score` default olarak cok dusuktur. Bu sayede stable plan elenmez;
istenirse GA veya elle config ile esik sonradan daraltilabilir.

2. Enemy denial double-count riski:

Enemy gezegeni ele gecirmek zaten bize future production value verir. Ayni
production akisini tam agirlikla `enemy_denial` olarak ikinci kez eklemek enemy
hedefleri gereksiz sisirir.

Yeni default:

```text
enemy_denial_weight = 0.35
```

Bu denial'i tamamen silmez, ama capture value ile ayni buyuklukte ikinci sinyal
olmasini engeller. Bu agirlik GA ile tune edilebilir.

3. Observation prefix bug:

Node feature 15-17 artik loop icinde artan prefix toplam degil. Tum production
toplamlari once hesaplanir, sonra her node'a ayni global context yazilir.

Bu onemli, cunku prefix bug policy'ye slot sirasi uzerinden yapay ve board
permutasyonuna hassas bir sinyal veriyordu.

4. Incoming fleet feature:

Fleet observation dogrudan target id vermedigi icin incoming feature'lar daha
once bos kaliyordu. Artik `predict_all_fleet_targets` ile trajectory uzerinden
tahmini target bulunur ve `incoming_self` / `incoming_enemy` node feature'lari
doldurulur.

Bu tam timeline oracle degildir; ama policy'nin ucan filo baskisini tamamen
gormemesinden daha iyidir. Kesin capture/hold hesabi yine `final_timeline.py`
ve micro-planner tarafinda yapilir.

5. Weak enemy raw launch fallback:

Neural rollout'ta policy enemy hedef secip micro-planner stable capture
bulamazsa artik raw sampled launch gonderilmez. Bu undercommit attack'lar
genellikle dusmana gemi hediye eder ve PPO reward'unu gurultulu hale getirir.

Yeni davranis:

```text
planner success -> launch
planner reject  -> no-op + reason log
```

Bu daha konservatif ama egitim icin daha temizdir. Ajan once stable capture
uretebilen target/source niyetlerini ogrenmelidir.

5.1 Action-validity penalty:

Hard mask sadece acikca imkansiz slotlari policy logits'inden siler. Ancak
mask sonrasi secilen `(source, target, amount)` yine de deterministic validation
tarafindan reddedilebilir:

- invalid slot / owner / amount / geometry
- micro planner stable capture bulamadi
- trajectory gecersiz

Bu durumlar eski PPO reward'unda zayif gorunuyordu. Artik rollout'a ayri dense
validity reward eklendi:

```text
invalid_*           -> -invalid_action_penalty
micro_reject/error  -> -reject_action_penalty
micro_capture       -> +capture_action_bonus
raw_launch          -> +raw_launch_bonus
```

Default:

```text
invalid_action_penalty = 0.02
reject_action_penalty  = 0.005
capture_action_bonus   = 0.002
raw_launch_bonus       = 0.0
```

Ceza sadece state'te kaynak ve hedef secme imkani varsa uygulanir. Yani oyuncunun
hic source'u veya hic target'i yoksa fallback/no-op cezalandirilmaz.

Training log artik bu kanali da gosterir:

```text
valid=...
cap=...
raw=...
rej=...
inv=...
```

Bu ayrim onemli:

```text
masking = impossible action'i bastirir
validation = secilen action'in fizik/timeline kontrolu
validity reward = validation'dan donen hatayi policy'ye ceza olarak yazar
```

5.2 Multi-source planner decoder:

Onceki rollout decoder micro-planner'a sadece policy'nin sectigi tek source'u
veriyordu. Bu, capture'in coklu donor koordinasyonu gerektirdigi ana tasarim
gerekcesini iptal ediyordu ve `noop/reject` oranini yukseltiyordu.

Yeni davranis:

```text
policy target_idx secer
policy source_idx secer = preferred donor / anchor
policy amount secer = donor budget scale
decoder selected source'u ilk siraya koyar
decoder top-M owned donor'u planner'a verir
micro planner multi-source capture-and-hold cozer
```

Bu tag/allocator sisteminin tam hali degildir; ama edge-style action ile
role-aware allocator arasinda daha dogru bir ara katmandir. Source action'i
tamamen ignore edilmez, fakat tek-source capture zorlamasi da kalkar.

5.3 Turn-cap terminal fallback:

Training `max_steps` env'in `episodeSteps` degerinden kisa olabilir. Oyun
bitmeden truncate edilirse env terminal reward'u gelmez. Bu durumda son
transition terminal kabul edilir ve production-share margin reward'a eklenir:

```text
truncate_terminal_reward = truncate_terminal_weight * production_share_margin
```

Bu, tasarimdaki terminal ana sinyal prensibini kisa training rollout'larinda
yaklasik olarak korur.

6. Defense ve strip risk:

Deterministic final agent artik sadece `CRITICAL` degil, `STRAINED` savunma
state'lerini de ele alir. Ayrica defense veya attack launch sonrasinda source
budget azaldikca strip-risk view yaklasik guncellenir.

Bu tam yeniden defense hesaplama degildir; ama ayni turn icinde ayni source'un
arka arkaya fazla soyulmasini azaltir.

### 21.6 Bilerek Sinirli Birakilan Kisimlar

Su anki neural training calisir, fakat dokumandaki tam credit/ledger tasariminin
hepsi bitmis degil.

Implemented:

- edge-aware GNN observation + model
- target/source/amount stochastic policy
- PPO self-play
- multi-source micro planner decoder/gate
- action-validity penalty
- truncate terminal fallback
- deterministic final micro submission builder

Henuz tam degil:

- full role-aware LP/min-cost-flow allocator
- attack/defense/support/evac flow matrix ledger
- post-episode campaign credit attribution
- planned_role -> actual_role reclassification
- credit conservation diagnostics
- support/evac counterfactual credit oracle

Bu ayrim onemli: mevcut neural script egitim kosabilir; ama dokumandaki ileri
credit assignment sistemi henuz production training loop'una bagli degil.
