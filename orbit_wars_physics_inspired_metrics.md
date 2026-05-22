# Orbit Wars için Fizik-Esinli Yayılım, Akış ve Dağılım Metrikleri

Bu not, gazların ve sıvıların yayılım/akış mekanizmalarından esinlenen metrikleri Orbit Wars gibi graph tabanlı bir RL problemi için yorumlar.

Temel fikir:

- Gezegenler: graph node
- Gezegenler arası hareket: graph edge
- Edge weight: distance değil, tercihen `predicted_travel_time`
- Gezegen üzerindeki filolar: node üzerinde kütle / yoğunluk / stok
- Filo hareketleri: akış / flux
- Production: kaynak üretimi / potansiyel
- Enemy pressure: karşı kuvvet alanı
- Friendly support: destek alanı

---

## 1. Diffusion Coefficient — Yayılım Katsayısı

Fizikte difüzyon, parçacıkların yüksek yoğunluktan düşük yoğunluğa yayılmasıdır.

Temel sezgi:

```text
flux = -D * gradient
```

Burada:

```text
D = diffusion coefficient
```

`D` ne kadar yüksekse madde o kadar hızlı yayılır.

Orbit Wars karşılığı:

```text
D ≈ mobility / travel_time
```

Bir gezegen grubunda filolar hızlıca diğer gezegenlere yayılabiliyorsa, o bölgenin “diffusion coefficient”ı yüksek gibi düşünülebilir.

Örnek metrik:

```text
fleet_diffusion_i =
    sum_j owned_ships_j / (travel_time(i, j) + 1)
```

Yorum:

```text
Yüksek fleet_diffusion = bu gezegen hızlı desteklenebilir.
Düşük fleet_diffusion = izole / savunmasız gezegen.
```

---

## 2. Flux — Akı / Akış Miktarı

Fizikte flux, bir yüzeyden birim zamanda geçen madde veya enerji miktarıdır.

Orbit Wars karşılığı:

```text
fleet_flux(i -> j) =
    ships_sent(i, j) / travel_time(i, j)
```

Bu action-level bir metriktir.

Örneğin:

```text
attack_flux =
    attacking_ships / travel_time_to_target
```

Yorum:

```text
Aynı ship sayısı kısa mesafeye gidiyorsa daha yüksek flux üretir.
```

Bu metrik saldırı yoğunluğu veya reinforcement yoğunluğu olarak kullanılabilir.

---

## 3. Gradient — Yoğunluk Eğimi

Difüzyonda hareketi belirleyen temel şey yoğunluk farkıdır.

```text
gradient = density_high - density_low
```

Orbit Wars’ta density yerine farklı kavramlar koyulabilir.

### 3.1 Ship Density Gradient

```text
ship_density_i = ships_i / planet_capacity_or_radius_i
```

veya daha basit:

```text
ship_density_i = ships_i
```

Sonra:

```text
ship_gradient(i, j) =
    ship_density_i - ship_density_j
```

Yorum:

```text
Pozitifse i gezegeninde j'ye göre fazla filo var.
Negatifse j daha dolu.
```

Bu, filoları yeniden dağıtmak için kullanılabilir.

### 3.2 Production Gradient

```text
production_gradient(i, j) =
    production_j - production_i
```

Yorum:

```text
Eğer j'nin production'ı yüksek ama savunması zayıfsa, j doğal hedef olabilir.
```

---

## 4. Fick Yasası — Yayılım Yönü

Fick yasası kabaca şunu söyler:

```text
Akış, yüksek yoğunluktan düşük yoğunluğa doğrudur.
```

Orbit Wars için doğrudan karşılık:

```text
expected_flow(i -> j) =
    -D * (ship_density_j - ship_density_i) / distance(i, j)
```

Fakat oyunda her zaman yüksek ship’ten düşük ship’e akmak istemeyiz. Çünkü hedef sadece dengeleme değil, stratejik üstünlüktür.

Daha iyi uyarlama:

```text
strategic_flow(i -> j) =
    D_ij * strategic_gradient(i, j)
```

Burada:

```text
strategic_gradient(i, j) =
    target_value_j - source_value_i
```

Örneğin:

```text
target_value_j =
    production_j
    - enemy_pressure_j
    - capture_cost_j
    + frontier_value_j
```

Bu durumda fiziksel difüzyonu doğrudan değil, stratejik potansiyel farkı üzerinden kullanmış oluruz.

---

## 5. Viscosity — Viskozite / Akışkanlık Direnci

Viskozite, sıvının akmaya karşı direncidir.

Orbit Wars karşılığı:

```text
viscosity_ij ≈ travel_time(i, j)
```

Daha zengin versiyon:

```text
viscosity_ij =
    travel_time(i, j)
    + sun_risk(i, j)
    + enemy_interception_risk(i, j)
```

Yorum:

```text
Düşük viscosity = iki gezegen arasında hareket kolay.
Yüksek viscosity = lojistik zor.
```

Bu edge weight için iyi bir isimlendirme olabilir.

---

## 6. Reynolds Number — Düzenli Akış mı, Kaotik Akış mı?

Fizikte Reynolds sayısı akışın laminar mı turbulent mi olduğunu gösterir.

Klasik form:

```text
Re = density * velocity * length / viscosity
```

Orbit Wars için yorumlanmış versiyon:

```text
strategic_reynolds =
    fleet_speed * fleet_mass * frontier_width / movement_resistance
```

Daha pratik versiyon:

```text
reynolds_i =
    outgoing_ship_flow_i / local_stability_i
```

Burada:

```text
outgoing_ship_flow_i =
    sum ships_sent_from_i / recent_turn_window

local_stability_i =
    owned_support_i + defense_i - enemy_pressure_i
```

Yorum:

```text
Yüksek Reynolds = agresif, hareketli, kaotik cephe.
Düşük Reynolds = stabil, pasif, düşük akışlı bölge.
```

RL state içinde “oyun bölgesi sakin mi, dinamik mi?” bilgisini verebilir.

---

## 7. Péclet Number — Taşınım mı Baskın, Difüzyon mu?

Péclet sayısı, taşınımın difüzyona göre baskın olup olmadığını ölçer.

```text
Pe = advective_transport / diffusive_transport
```

Orbit Wars karşılığı:

```text
Pe =
    directed_attack_flow / passive_reinforcement_spread
```

Yani:

```text
peclet =
    offensive_fleet_flux / defensive_mobility
```

Yorum:

```text
Pe yüksekse: aktif saldırı / taarruz baskın.
Pe düşükse: pasif yayılım / savunma / dengeleme baskın.
```

Agent’ın oyun stilini analiz etmek için kullanılabilir.

---

## 8. Mean Free Path — Ortalama Serbest Yol

Gazlarda bir molekülün çarpışmadan önce aldığı ortalama mesafedir.

Orbit Wars karşılığı:

```text
mean_free_path =
    average distance a fleet travels before encountering meaningful enemy pressure
```

Yaklaşık:

```text
mean_free_path_i =
    min_travel_time(i, enemy_influence_above_threshold)
```

Yorum:

```text
Yüksek mean_free_path = güvenli hareket alanı geniş.
Düşük mean_free_path = düşman baskısı yakın.
```

Bu frontier ve güvenli genişleme metrikleri için iyi bir proxy olabilir.

---

## 9. Entropy — Dağılım Düzensizliği

Fizikte entropy, sistemin düzensizliği veya yayılmışlığı gibi yorumlanabilir.

Orbit Wars’ta filo dağılımı için çok kullanışlıdır.

```text
p_i = ships_i / total_ships
entropy = -sum p_i * log(p_i)
```

Yorum:

```text
Düşük entropy = filolar birkaç gezegende toplanmış.
Yüksek entropy = filolar çok yayılmış.
```

Ama yüksek entropy her zaman iyi değildir. Daha iyi bir metrik:

```text
relative_entropy =
    KL(ship_distribution || desired_distribution)
```

Mesela desired distribution production’a göre kurulabilir:

```text
desired_p_i = production_i / total_production
```

Sonra:

```text
KL divergence yüksekse:
    filo dağılımı production dağılımıyla uyumsuz.
```

---

## 10. Concentration Field — Yoğunluk Alanı

Fizikte gaz veya sıvı yayılımı bir concentration field ile modellenir.

Orbit Wars’ta haritaya sürekli bir alan gibi bakılabilir:

```text
friendly_concentration(x, y)
enemy_concentration(x, y)
production_concentration(x, y)
```

Gezegen bazında discrete versiyon:

```text
friendly_concentration_i =
    sum_j friendly_ships_j * exp(-distance(i,j) / tau)
```

Bu, influence map mantığına çok benzer.

Daha fiziksel isimlerle:

```text
friendly_ship_concentration_field
enemy_ship_concentration_field
production_potential_field
```

Bunlar RL state’e node feature olarak verilebilir.

---

## 11. Potential Field — Potansiyel Alan

Elektrostatik veya gravitasyonel alan gibi düşünülebilir. Gezegenler production kapasitesine göre “çekim” uygular.

```text
potential_i =
    sum_j production_j / (distance(i, j) + epsilon)
```

Enemy için:

```text
enemy_threat_potential_i =
    sum_j enemy_ships_j / (distance(i, j) + epsilon)
```

Friendly için:

```text
friendly_support_potential_i =
    sum_j friendly_ships_j / (distance(i, j) + epsilon)
```

Net potansiyel:

```text
net_potential_i =
    production_potential_i
    + friendly_support_potential_i
    - enemy_threat_potential_i
```

Yorum:

```text
Yüksek net potential = stratejik olarak cazip ve desteklenebilir bölge.
```

---

## 12. Graph Laplacian Smoothness — Dağılım Ne Kadar Pürüzlü?

Graph üzerinde diffusion denince en önemli matematiksel araçlardan biri graph Laplacian’dır.

Bir node feature’ı düşün:

```text
x_i = ships_i
```

Graph Laplacian smoothness:

```text
smoothness = x^T L x
```

Daha açık hali:

```text
smoothness =
    sum_edges w_ij * (ships_i - ships_j)^2
```

Yorum:

```text
Yüksekse: filo dağılımı keskin / dengesiz.
Düşükse: komşu gezegenler arasında filo dengeli.
```

Bu özellikle “reinforcement distribution düzgün mü?” sorusu için çok iyi bir metriktir.

Aynısı başka feature’lar için de kullanılabilir:

```text
ship_smoothness
production_smoothness
pressure_smoothness
vulnerability_smoothness
```

---

## 13. Conductance — Akış Geçirgenliği

Graph teoride conductance, bir kümenin dış dünyayla bağlantı gücünü ölçer. Fizikte iletkenlik veya akış geçirgenliği gibi düşünülebilir.

Kendi gezegen kümen için:

```text
conductance(owned_region) =
    cut_edges_weight / internal_edges_weight
```

Yorum:

```text
Yüksek conductance = bölgen dışa çok açık.
Düşük conductance = daha kapalı / kompakt bölge.
```

Ama oyunda yüksek conductance bazen iyidir; daha fazla expansion route sağlar. Savunma açısından ise riskli olabilir.

---

## 14. Permeability — Geçirgenlik

Bir bölgeden filo geçişi ne kadar kolay?

```text
permeability_region =
    number_of_low_travel_time_edges_crossing_region_boundary
```

veya:

```text
permeability =
    sum_boundary_edges 1 / travel_time
```

Yorum:

```text
Yüksek permeability = cephe geçirgen, saldırı ve savunma hareketli.
Düşük permeability = cephe dar, chokepoint gibi.
```

---

## 15. Mixing Time — Karışma Süresi

Markov chain veya diffusion process’te mixing time, sistemin dengeli dağılıma ulaşması için gereken süredir.

Orbit Wars karşılığı:

```text
mixing_time =
    filoların kendi gezegenlerin arasında dengeli şekilde yeniden dağıtılması için gereken yaklaşık süre
```

Pratik yaklaşım:

```text
mixing_time_owned =
    average_shortest_path_length(G_owned, weight="travel_time")
```

veya:

```text
mixing_time_owned =
    graph_diameter(G_owned, weight="travel_time")
```

Yorum:

```text
Düşük mixing time = imparatorluk içinde hızlı yeniden konumlanma.
Yüksek mixing time = savunma koordinasyonu zor.
```

---

## 16. Diffusion Distance — Yayılım Mesafesi

Bir node’dan başlayan kaynak belli süre sonra graph üzerinde nasıl yayılır?

Orbit Wars karşılığı:

```text
Bir gezegenden çıkan reinforcement gücü t turn içinde nerelere ulaşabilir?
```

Basit metrik:

```text
reachable_production_i(t) =
    sum_{j reachable within t} production_j
```

veya:

```text
reachable_friendly_support_i(t) =
    sum_{j owned and travel_time(j,i) <= t} ships_j
```

Feature örnekleri:

```text
support_radius_5
support_radius_10
support_radius_20
```

---

## Orbit Wars için En Yararlı Fizik-Esinli Metrikler

| Fizik Kavramı | Orbit Wars Metriği | Kullanım |
|---|---|---|
| Diffusion coefficient | Mobility | Filo yayılım hızı |
| Flux | Fleet flux | Saldırı / savunma akışı |
| Gradient | Strategic gradient | Hedef cazibesi |
| Viscosity | Movement resistance | Edge maliyeti |
| Entropy | Fleet distribution entropy | Filo dağılımı |
| Potential field | Production / threat / support potential | Influence map |
| Mean free path | Safe expansion radius | Güvenli hareket alanı |
| Laplacian smoothness | Distribution imbalance | Filo dengesizliği |
| Conductance | Region openness | Cephe açıklığı |
| Mixing time | Internal redistribution time | Savunma esnekliği |

---

## Önerilen Feature İsimleri

```text
ship_concentration
enemy_concentration
production_potential
friendly_support_potential
enemy_threat_potential
net_potential
fleet_flux
attack_flux
defense_flux
strategic_gradient
movement_viscosity
owned_region_conductance
owned_region_mixing_time
ship_distribution_entropy
ship_laplacian_smoothness
support_radius_5
support_radius_10
frontier_permeability
safe_mean_free_path
```

---

## Güçlü Bir Fiziksel Feature Seti

RL state için temiz fiziksel yapı aşağıdaki gibi kurulabilir.

### 1. Friendly Concentration

```text
C_friendly(i) =
    sum_j friendly_ships_j * exp(-travel_time(i,j) / tau)
```

### 2. Enemy Concentration

```text
C_enemy(i) =
    sum_j enemy_ships_j * exp(-travel_time(i,j) / tau)
```

### 3. Net Pressure

```text
P_net(i) =
    C_friendly(i) - C_enemy(i)
```

### 4. Production Potential

```text
V_prod(i) =
    sum_j production_j / (travel_time(i,j) + 1)
```

### 5. Strategic Potential

```text
V_strategy(i) =
    alpha * V_prod(i)
    + beta * C_friendly(i)
    - gamma * C_enemy(i)
```

### 6. Flow Opportunity

```text
Flow(i -> j) =
    available_ships_i
    * max(0, V_strategy(j) - V_strategy(i))
    / (travel_time(i,j) + 1)
```

Bu, fiziksel akış mantığını oyuna uyarlayan güçlü bir action scoring fonksiyonu olarak kullanılabilir.

---

## Kısa Özet

Orbit Wars graph-RL tasarımında fiziksel yayılım ve akış kavramları özellikle şu alanlarda işe yarar:

```text
1. Potential field:
   Hedef ve tehdit alanı çıkarımı.

2. Flux:
   Filo akış gücü / saldırı yoğunluğu / reinforcement yoğunluğu.

3. Laplacian smoothness ve entropy:
   Filo dağılım kalitesi.

4. Conductance ve permeability:
   Cephe açıklığı ve bölge geçirgenliği.

5. Mixing time:
   İç lojistik ve savunma esnekliği.
```

En pratik başlangıç üçlüsü:

```text
1. Potential field
2. Fleet flux
3. Laplacian smoothness / entropy
```
