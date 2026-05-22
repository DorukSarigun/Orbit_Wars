# Orbit Wars için Graph Tabanlı RL Metrikleri

Bu doküman, Kaggle Orbit Wars oyununu Reinforcement Learning ile çözmeye çalışırken gezegenleri bir `NetworkX` graph yapısı gibi modellemek için kullanılabilecek metrikleri özetler.

Temel fikir:

- Gezegenler: node
- Gezegenlerin production kapasitesi: node weight / node feature
- Gezegenler arası mesafe veya tahmini ulaşım süresi: edge weight
- Gezegen üzerindeki filo sayısı: node feature
- Uçuş halindeki filolar: dynamic edge/node-context feature
- Ele geçirilen gezegenler: oyuncuya ait induced subgraph

Orbit Wars için graph yaklaşımı mantıklı, fakat graph statik değildir. Gezegenler hareket ettiği için edge distance / travel time zamanla değişebilir. Bu nedenle edge weight olarak yalnızca Euclidean distance değil, mümkünse **predicted travel time** veya **intercept travel time** kullanılmalıdır.

---

## 1. Node-level metrikler

Node-level metrikler, tek bir gezegenin ekonomik, stratejik ve taktik değerini ölçmek için kullanılır.

### 1.1 Temel node feature'ları

Her gezegen için ilk aşamada şu feature'lar tutulabilir:

```text
owner
ships_on_planet
production
radius
x, y
distance_to_sun
orbital/static flag
incoming_friendly_ships
incoming_enemy_ships
turns_until_first_incoming_fleet
```

Bunlar klasik graph metriği değildir, ancak graph tabanlı feature'ların temel girdileridir.

---

### 1.2 Production-weighted value

Bir gezegenin çıplak ekonomik değeri:

```text
planet_value_i = production_i * remaining_turns
```

Daha gerçekçi değer:

```text
net_value_i =
    production_i * (remaining_turns - travel_time_to_i)
    - required_ships_to_capture_i
    - opportunity_cost
```

Yorum:

- Production yüksekse gezegen uzun vadede değerlidir.
- Travel time yüksekse gezegeni ele geçirmenin geri dönüşü azalır.
- Capture cost yüksekse hedef pahalıdır.

---

### 1.3 Safety / vulnerability score

Bir gezegenin düşme riskini ölçer.

```text
vulnerability_i =
    enemy_reachable_power_i / max(1, friendly_defense_i + incoming_reinforcement_i)
```

Burada:

```text
enemy_reachable_power_i =
    sum(enemy_ships_j / travel_time(j, i) for enemy planet j)
```

Daha smooth versiyon:

```text
enemy_pressure_i =
    sum(enemy_ships_j * exp(-lambda * travel_time(j, i)))
```

Yorum:

- Yakın düşman gezegenleri daha fazla baskı üretir.
- Uzak düşman gezegenleri daha düşük ağırlık alır.

---

### 1.4 Reinforcement accessibility / supportability

Kendi gezegenlerimizden belirli bir gezegeni ne kadar kolay destekleyebileceğimizi ölçer.

```text
friendly_support_i =
    sum(friendly_available_ships_j * exp(-lambda * travel_time(j, i)))
```

Yorum:

- Yüksekse gezegen savunulabilir.
- Düşükse gezegen izole olabilir.
- Özellikle ileri karakol gibi duran gezegenlerde önemlidir.

---

## 2. Edge-level metrikler

Edge-level metrikler, iki gezegen arasındaki bağlantının kalitesini ve stratejik anlamını ölçer.

### 2.1 Edge weight

İlk baseline için:

```text
edge_weight(i, j) = euclidean_distance(i, j)
```

Daha doğru versiyon:

```text
edge_weight(i, j, t) = estimated_travel_time(i, j, t)
```

Çünkü gezegenler hareket ettiği için şu anki mesafe, gerçek ulaşım maliyetini tam temsil etmeyebilir.

---

### 2.2 Edge feature'ları

Her edge için şu feature'lar tutulabilir:

```text
distance_now(i, j)
predicted_arrival_time(i, j)
sun_intersection_risk(i, j)
fleet_speed_given_ships
enemy_interception_risk
```

---

### 2.3 Strategic edge value

Bir hedefe gitmenin stratejik değerini ölçer.

```text
edge_value(i, j) =
    production_j / (travel_time(i, j) + capture_cost_j)
```

Yorum:

- Production yüksek hedefler caziptir.
- Travel time ve capture cost arttıkça hedefin cazibesi azalır.

---

## 3. Centrality metrikleri

Centrality metrikleri, haritadaki stratejik gezegenleri bulmak için kullanılabilir.

---

### 3.1 Closeness centrality

Bir gezegenin diğer gezegenlere ortalama olarak ne kadar yakın olduğunu ölçer.

```text
closeness_i = 1 / average_shortest_path_distance(i, all_nodes)
```

Orbit Wars yorumu:

```text
high closeness = iyi lojistik merkez
```

Kendi gezegenlerin içinde closeness değeri yüksek olanları reinforcement hub gibi düşünebilirsin.

---

### 3.2 Betweenness centrality

Bir gezegenin shortest path'lerin üzerinde ne kadar sık yer aldığını ölçer.

```text
high betweenness = geçiş noktası / chokepoint
```

Fakat Orbit Wars'ta filolar doğrudan hedefe gidebildiği için klasik betweenness her zaman anlamlı olmayabilir.

Daha anlamlı kullanım için graph şöyle kurulabilir:

```text
edge exists if travel_time(i, j) < threshold
edge weight = travel_time(i, j)
```

Bu durumda betweenness, yakınlık ağı üzerindeki lojistik ara merkezleri gösterebilir.

---

### 3.3 Current-flow betweenness

Klasik betweenness yalnızca shortest path'lere bakar. Current-flow betweenness ise alternatif makul yolları da hesaba katar.

Yorum:

```text
high current-flow betweenness = çok sayıda makul rota üzerinde etkili gezegen
```

Orbit Wars gibi sürekli uzayda hareket edilen oyunlarda, klasik betweenness'ten daha anlamlı olabilir.

---

### 3.4 Eigenvector / PageRank tarzı skor

Bir gezegen sadece çok bağlantılı olduğu için değil, değerli gezegenlere yakın olduğu için de stratejik olabilir.

```text
planet_importance_i =
    alpha * production_i
    + beta * sum(importance_j / travel_time(i, j))
```

Yorum:

- Production cluster'ları bulmak için kullanılabilir.
- Stratejik olarak yoğun bölgeleri tespit eder.

---

## 4. Subgraph metrikleri

Kendi ele geçirdiğimiz gezegenleri ayrı bir induced subgraph olarak düşünebiliriz:

```python
G_own = G.subgraph(owned_planets)
```

Bu subgraph üzerinden mobility, compactness, fragmentation gibi metrikler hesaplanabilir.

---

### 4.1 Mobility score

Kendi gezegenlerin arasında gemi transferinin ne kadar kolay olduğunu ölçer.

```text
mobility =
    average over owned i,j of 1 / travel_time(i, j)
```

Daha pratik versiyon:

```text
mobility =
    sum_{i,j in owned} friendly_ships_i / (travel_time(i, j) + 1)
```

Yorum:

- Yüksek mobility, savunma ve saldırı için esnek pozisyon anlamına gelir.
- Düşük mobility, gezegenlerin birbirini destekleyemediğini gösterebilir.

---

### 4.2 Production-weighted mobility

Her gezegen eşit değildir. Production yüksek gezegenlerden hareket kabiliyeti daha değerlidir.

```text
prod_mobility =
    sum_{i,j in owned} production_i * production_j / (travel_time(i, j) + 1)
```

Yorum:

- Kendi ekonominin birbirini destekleyip destekleyemediğini ölçer.
- Yüksek production'lı gezegenlerin yakın olması güçlü bir lojistik yapı oluşturur.

---

### 4.3 Compactness / spread

Kendi gezegenlerinin ne kadar dağınık olduğunu ölçer.

```text
spread =
    average_pairwise_distance(owned_planets)
```

Daha stratejik versiyon:

```text
efficient_spread =
    owned_production / average_pairwise_travel_time
```

Yorum:

- Düşük spread: kompakt imparatorluk.
- Yüksek spread: geniş ama potansiyel olarak savunması zor imparatorluk.
- Tek başına düşük spread iyi değildir; çok kompakt kalmak expansion'ı zayıflatabilir.

---

### 4.4 Fragmentation

Kendi gezegenlerinin kaç parçaya ayrıldığını ölçer.

```text
num_connected_components(G_own)
```

Ek metrik:

```text
largest_component_ratio =
    ships_in_largest_component / total_owned_ships
```

Yorum:

- Component sayısı yüksekse imparatorluk lojistik olarak parçalıdır.
- Largest component ratio düşükse gücün çok dağılmış olabilir.

---

### 4.5 Boundary / frontier score

Düşmana veya neutrallara yakın kendi gezegenlerini belirler.

```text
frontier_i =
    min_travel_time(i, enemy_or_neutral_planets)
```

Kendi bölgenin savunma riskini ölçmek için:

```text
frontier_exposure =
    sum_{i in owned} production_i / min_enemy_travel_time(i)
```

Yorum:

- Değerli gezegenlerin düşmana yakın olması risklidir.
- Frontier exposure yüksekse cephe hattı tehlikelidir.

---

## 5. Enemy pressure ve influence map

RL state için en güçlü feature gruplarından biri influence map'tir.

Her gezegen için friendly influence ve enemy influence hesaplanabilir:

```text
friendly_influence_i =
    sum_{j in friendly planets} available_ships_j * exp(-lambda * travel_time(j, i))

enemy_influence_i =
    sum_{j in enemy planets} enemy_ships_j * exp(-lambda * travel_time(j, i))
```

Sonra:

```text
influence_balance_i =
    friendly_influence_i - enemy_influence_i
```

Yorum:

```text
positive = bizim kontrol alanımız
negative = düşman kontrol alanı
near zero = contested zone
```

Kullanım:

- Node feature olarak verilebilir.
- Hedef seçimi için kullanılabilir.
- Savunma / saldırı kararlarında kullanılabilir.

---

## 6. Capture opportunity metrikleri

Neutral veya enemy planet hedefleri için kullanılır.

---

### 6.1 Capture margin

```text
capture_margin_i =
    available_attack_ships_to_i - required_ships_to_capture_i
```

Yorum:

- Pozitifse hedef alınabilir.
- Negatifse mevcut güçle hedef almak riskli veya imkansızdır.

---

### 6.2 Future defense

Target planet travel time boyunca production üretmeye devam ediyorsa savunma tahmini yapılmalıdır.

```text
future_defense_i =
    current_ships_i + production_i * travel_time
```

Bu, özellikle enemy planet hedeflerinde önemlidir.

---

### 6.3 ROI score

```text
roi_i =
    production_i * (remaining_turns - travel_time_i)
    / (ships_required_i + travel_time_i + 1)
```

Yorum:

- Hedefin ekonomik geri dönüşünü ölçer.
- Erken oyunda production ağırlığı yüksek hedefler avantajlı olabilir.
- Geç oyunda travel time daha kritik hale gelir.

---

### 6.4 Denial value

Enemy planet almak iki yönlü kazanç sağlar:

1. Sen production kazanırsın.
2. Rakip production kaybeder.

Bu nedenle enemy planet değeri yaklaşık şöyle düşünülebilir:

```text
enemy_target_value_i =
    2 * production_i * remaining_turns
```

Yorum:

- Enemy planet bazen neutral planet'ten daha değerlidir.
- Özellikle yüksek production'lı düşman gezegenleri stratejik hedef olabilir.

---

## 7. Fleet-flow / distribution metrikleri

Gezegen üzerindeki filo sayısını modele vermek için sadece `ships_on_planet` yeterli olmayabilir. Önemli olan gemilerin doğru yerde olup olmadığıdır.

---

### 7.1 Ship-production mismatch

Production yüksek ama ship sayısı düşük olan gezegenler savunmasız olabilir.

```text
defense_per_production_i =
    ships_i / max(1, production_i)
```

Yorum:

- Düşük değer: değerli ama zayıf savunulan gezegen.
- Yüksek değer: iyi savunulan veya aşırı savunulan gezegen.

---

### 7.2 Idle fleet ratio / idle score

Çok ship var ama üretime, düşmana veya hedeflere uzaksa verimsiz olabilir.

```text
idle_score_i =
    ships_i * average_travel_time_to_useful_targets_i
```

Yorum:

- Yüksek idle score, gemilerin yanlış yerde biriktiğini gösterebilir.

---

### 7.3 Overstacking

Bir gezegende gereğinden fazla ship tutmayı ölçer.

```text
overstack_i =
    max(0, ships_i - needed_defense_i)
```

Reward shaping için:

```text
penalty = - sum(overstack_i)
```

Dikkat:

- Bunu dikkatli kullanmak gerekir.
- Bazen büyük filo biriktirmek doğru strateji olabilir.

---

### 7.4 Distribution entropy

Gemilerin ne kadar dağılmış olduğunu ölçer.

```text
p_i = ships_i / total_ships
entropy = - sum p_i log(p_i)
```

Yorum:

- Düşük entropy: gemiler birkaç gezegende toplanmış.
- Yüksek entropy: gemiler yayılmış.
- Tek başına iyi/kötü değildir.

Daha anlamlı kullanım:

```text
production_aligned_distribution =
    similarity(ship_distribution, production_distribution)
```

Yani gemi dağılımı production değerleriyle uyumlu mu?

---

## 8. Community / cluster metrikleri

Haritadaki gezegenleri travel-time graph üzerinden community'lere bölebilirsin.

Orbit Wars yorumu:

```text
cluster = lojistik olarak birbirine yakın gezegen grubu
```

Her cluster için şu metrikler hesaplanabilir:

```text
cluster_production
cluster_ships
cluster_owner_majority
cluster_contestedness
cluster_enemy_pressure
cluster_mobility
```

---

### 8.1 Cluster control score

```text
cluster_control_c =
    friendly_ships_c + friendly_production_c * horizon
    - enemy_ships_c - enemy_production_c * horizon
```

Yorum:

- Pozitifse cluster içinde üstünlük bizde olabilir.
- Negatifse düşman üstünlüğü vardır.

---

### 8.2 Cluster expansion score

```text
cluster_expansion_score =
    neutral_production_nearby / average_capture_cost
```

Yorum:

- Expansion için cazip bölgeleri bulur.
- Neutral gezegenlerin üretim kapasitesi ile capture cost arasındaki ilişkiyi ölçer.

---

## 9. Group centrality

Tek tek node centrality yerine, sahip olduğumuz gezegenler kümesini ölçmek daha doğru olabilir.

Örnek metrikler:

```text
owned_group_closeness
owned_group_betweenness
owned_group_degree
```

Yorum:

```text
owned_group_closeness high = haritanın geneline erişimimiz iyi
owned_group_betweenness high = harita akışını kontrol ediyoruz
owned_group_degree high = çok sayıda hedefe erişimimiz var
```

---

## 10. RL için pratik feature set önerisi

Başta çok fazla metrik koymak yerine, yüksek sinyal taşıyan feature'larla başlamak daha mantıklı olabilir.

---

### 10.1 Node features

Her planet için:

```text
owner_one_hot
ships
production
remaining_turns
x, y
is_owned
is_enemy
is_neutral
incoming_friendly_ships
incoming_enemy_ships
friendly_influence
enemy_influence
influence_balance
vulnerability
supportability
closeness
production_weighted_closeness
frontier_score
```

---

### 10.2 Edge features

```text
current_distance
predicted_travel_time
sun_risk
same_owner
source_available_ships
target_capture_cost
```

---

### 10.3 Global features

```text
owned_planet_count
enemy_planet_count
owned_total_ships
enemy_total_ships
owned_total_production
enemy_total_production
production_diff
ship_diff
mobility_diff
frontier_exposure_diff
largest_component_ratio
average_owned_spread
turn / 500
```

---

## 11. En değerli 12 metrik

Başlangıç için en yüksek sinyal taşıyacak metrikler:

| Metrik | Nerede kullanılır? | Anlamı |
|---|---:|---|
| `production` | node | Ekonomik değer |
| `travel_time` | edge | Gerçek mesafe / ulaşım süresi |
| `required_ships_to_capture` | node/action | Capture maliyeti |
| `ROI` | action scoring | Hedef seçimi |
| `friendly_influence` | node | Destek gücü |
| `enemy_influence` | node | Tehdit gücü |
| `influence_balance` | node | Kontrol alanı |
| `vulnerability` | node | Savunma riski |
| `supportability` | node | Reinforce kolaylığı |
| `mobility` | owned subgraph | İç lojistik |
| `frontier_exposure` | owned subgraph | Cephe riski |
| `production_weighted_closeness` | node | Stratejik üretim merkezi |

---

## 12. Basit NetworkX taslağı

Aşağıdaki kod ilk baseline için kullanılabilir. İlk versiyonda travel time olarak Euclidean distance kullanılır. Daha sonra bu fonksiyon intercept solver ile değiştirilebilir.

```python
import networkx as nx
import math


def travel_time(p1, p2):
    """
    İlk baseline: Euclidean distance.
    Daha sonra bunu gerçek intercept / predicted arrival time hesabı ile değiştirmek gerekir.
    """
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return math.sqrt(dx * dx + dy * dy)


def build_planet_graph(planets, max_edge_time=None):
    G = nx.Graph()

    for p in planets:
        G.add_node(
            p["id"],
            owner=p["owner"],
            ships=p["ships"],
            production=p["production"],
            x=p["x"],
            y=p["y"],
            radius=p["radius"],
        )

    for i, p1 in enumerate(planets):
        for p2 in planets[i + 1:]:
            tt = travel_time(p1, p2)

            if max_edge_time is None or tt <= max_edge_time:
                G.add_edge(
                    p1["id"],
                    p2["id"],
                    weight=tt,
                    inv_weight=1.0 / (tt + 1e-6),
                )

    return G


def add_graph_metrics(G):
    closeness = nx.closeness_centrality(G, distance="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")

    for node in G.nodes:
        G.nodes[node]["closeness"] = closeness[node]
        G.nodes[node]["betweenness"] = betweenness[node]

        prod = G.nodes[node]["production"]
        G.nodes[node]["prod_closeness"] = prod * closeness[node]

    return G


def owned_mobility(G, player_id):
    owned = [n for n, d in G.nodes(data=True) if d["owner"] == player_id]
    if len(owned) <= 1:
        return 0.0

    total = 0.0
    count = 0

    for i, a in enumerate(owned):
        for b in owned[i + 1:]:
            try:
                dist = nx.shortest_path_length(G, a, b, weight="weight")
                total += 1.0 / (dist + 1e-6)
                count += 1
            except nx.NetworkXNoPath:
                pass

    return total / max(1, count)
```

---

## 13. Kritik öneri

Bu oyunda graph representation güzel, fakat edge weight'i sadece distance olmamalıdır.

Daha doğru yaklaşım:

```text
edge_weight = predicted_arrival_time
```

Çünkü:

- Gezegenler hareket eder.
- Fleet speed değişken olabilir.
- Hedef gezegen arrival zamanına kadar production üretebilir.
- Rota / güneş / geometri etkileri olabilir.

Bu nedenle RL modeline iki katmanlı feature vermek mantıklı olur:

```text
1. Geometrik graph:
   distance, angle, orbital position

2. Stratejik graph:
   travel_time, capture_cost, influence, production ROI
```

---

## 14. Önerilen RL mimarisi

Başlangıç için şu mimari düşünülebilir:

```text
GNN encoder over planet graph
+
global features
+
action mask / candidate action pruning
+
policy head: source planet, target planet, ship fraction
```

Action space büyük olduğu için full action space yerine candidate action üretmek daha mantıklı olabilir.

Örnek action formatı:

```text
action = (source_planet_id, target_planet_id, ship_fraction)
```

Candidate pruning örnekleri:

```text
source candidates:
    owned planets with available_ships > threshold

target candidates:
    high ROI neutral planets
    vulnerable enemy planets
    threatened friendly planets needing reinforcement

ship_fraction candidates:
    0.25, 0.5, 0.75, 1.0
```

---

## 15. Başlangıç için önerilen minimum feature grubu

İlk RL denemesinde şu feature'larla başlamak yeterli olabilir:

### Node

```text
owner_one_hot
ships
production
x, y
friendly_influence
enemy_influence
influence_balance
vulnerability
closeness
prod_closeness
```

### Edge

```text
travel_time
same_owner
```

### Global

```text
turn_ratio
production_diff
ship_diff
owned_planet_count
enemy_planet_count
mobility_diff
```

### Action scoring yardımcıları

```text
ROI
capture_margin
future_defense
frontier_exposure
```

---

## 16. Genel sonuç

Orbit Wars için graph tabanlı RL yaklaşımında en kritik fikirler şunlardır:

1. Gezegenleri node, ulaşım maliyetlerini edge olarak modelle.
2. Edge weight olarak mümkünse Euclidean distance yerine predicted travel time kullan.
3. Production, ship count, owner ve incoming fleet bilgilerini node feature yap.
4. Influence map ile haritanın kontrol alanlarını çıkar.
5. Kendi gezegenlerin üzerinde mobility, spread, fragmentation ve frontier exposure ölç.
6. Hedef seçimi için ROI, capture margin ve denial value hesapla.
7. Action space'i doğrudan tüm kombinasyonlar yerine candidate pruning ile küçült.
8. İlk baseline için sade feature set kur, sonra metrikleri ablation ile artır.
