# Orbit Wars Evaluation Notes

## Strategy Agent + Scripted Logistics

Son tartisma icin ana ayrim:

```text
Agent = strateji karari
Scripted optimizer = lojistik uygulama
```

Agent'in her tur dogrudan her edge icin `source -> target` gemi miktari secmesi action space'i buyutuyor ve credit assignment'i dagitiyor. Daha temiz yaklasim, agent'in campaign niyetini secmesi ve gemi dagitimini deterministik bir lojistik solver'a birakmasi.

## Agent Her Tur Neyi Secmeli?

Agent'in karar birimi target/campaign seviyesinde olmali:

```text
target_gate[t]      # bu gezegen icin campaign ac / acma
target_mode[t]      # attack / defend / ignore
target_priority[t]  # secilen hedefler arasinda oncelik
target_margin[t]    # minimum capture mi, guclu margin mi?
source_reserve[h]   # kendi gezegenlerinde ne kadar reserve birak?
global_aggression   # genel risk/agresiflik ayari
```

Minimal ilk versiyon:

```text
target_gate[t]
target_mode[t]
priority_order
margin_level[t]     # minimum / normal / guclu
global_reserve
```

Bu tasarim mevcut target-level reward ile iyi eslesir:

```text
agent target t'yi secti
lojistik solver target t icin launch urettti
campaign sonucu credit_target[t]'ye yazildi
```

Boylece policy karar birimi ile reward karar birimi ayni semantik seviyeye gelir.

## Agent Neyi Secmemeli?

Agent'in asagidaki mikro lojistik detaylari secmesi gerekmiyor:

```text
source h'den target t'ye tam kac ship?
her edge icin ayri fraction?
hangi source hangi hedefe gitsin?
```

Bunlar scripted logistics katmani tarafindan daha stabil cozulur. Agent bu detaylari secmeye basladiginda action space buyur, spray davranisi kolaylasir ve reward ayni target sonucu icin cok fazla edge kararina dagilir.

## Scripted Logistics: LP / Min-Cost Flow

Lojistik problemi linear programming / transportation problem olarak yazilabilir:

```text
x[h,t] = source h'den target t'ye gonderilecek ship sayisi

constraints:
  sum_t x[h,t] <= surplus[h]       # source kapasitesi
  sum_h x[h,t] >= demand[t]        # hedef ihtiyaci
  x[h,t] >= 0

objective:
  minimize sum_h,t x[h,t] * cost[h,t]
```

Cost fonksiyonu:

```text
cost[h,t] =
  eta[h,t]
  + source_strip_risk[h]
  + path_risk[h,t]
  + opportunity_cost[h]
  - priority_value[t]
```

Mevcut `reworked_allocator.py` buna yakin bir cozum iceriyor. `mcf` solver, transportation LP'yi `scipy.optimize.linprog(method="highs")` ile cozuyor ve hedef onceligini lexicographic buyuk agirliklarla koruyor.

## En Hizli Capture Problemi

"Bir gezegeni en hizli nasil alirim?" tam haliyle lineer degil, cunku:

```text
fleet_speed = f(ship_count)
eta = distance / fleet_speed(ship_count)
demand = target_ships + target_production * eta + margin
```

Gonderilen ship sayisi ETA'yi, ETA da gereken ship sayisini etkiliyor.

Pratik cozumler:

1. Approximate LP:

```text
eta[h,t] = distance / speed(surplus[h])
demand[t] = target_ships + target_prod * min_eta[t] + margin
```

Sonra transportation LP cozulur. Basit ve hizli.

2. Fastest feasible binary search:

```text
Verilen T icin:
  kullanilabilir source'lar = eta[h,t] <= T
  demand = target_ships + target_prod * T + margin
  sum_h x[h,t] >= demand mumkun mu?

T uzerinde binary search ile minimum feasible T bulunur.
```

Bu, "en hizli capture" hedefi icin daha dogru ve hala yonetilebilir bir yaklasimdir.

3. MILP / piecewise bundle:

```text
10 ship -> eta 20
20 ship -> eta 14
40 ship -> eta 9
```

Her source-target icin aday bundle'lar secilir. Daha dogru ama daha karmasik ve pahali.

## Onerilen Sonraki Mimari

En mantikli sonraki policy/logistics ayrimi:

```text
Target-first strategy head
  -> target_gate
  -> target_mode
  -> priority
  -> margin / reserve / aggression

Scripted logistics solver
  -> LP / min-cost-flow ile source-target ship allocation
  -> concrete launch list
```

Bu mimari edge-head'e gore daha stratejik ve daha sample-efficient olabilir. Agent "hangi gezegen onemli?" sorusunu ogrenir; "hangi kaynaktan 7 ship, hangisinden 13 ship?" gibi operasyonel detaylari solver cozer.

## Reward Ile Iliski

Bu ayrim v13 remastered reward ile uyumludur:

```text
campaign reward -> target slot
terminal reward -> target returns'e yayilir
logistics deterministic -> credit agent'in target/campaign kararina gider
```

Bu nedenle hedeflenen uzun vadeli tasarim:

```text
strategy-level policy action == reward credit unit == target campaign
```

Bu eslesme, hem credit assignment'i sade hale getirir hem de spray davranisini azaltir.
