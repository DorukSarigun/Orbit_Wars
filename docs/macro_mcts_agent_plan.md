# Macro MCTS Agent Plan

Bu dokuman, mevcut `TimedCaptureMCTS` fikrini daha genel bir karar agacina
tasimak icin uygulanabilir bir plan tarif eder. Amac sadece "hangi gezegeni ele
gecirelim?" sorusunu degil, ayni agac icinde "nereyi savunalim, nereyi
destekleyelim, nereden tahliye edelim?" sorularini da degerlendiren yeni bir
agent tasarlamaktir.

Mevcut agentlarda launch lojistigi, rota hesabi ve gemi dagitimi zaten karmasik
ve pahali. Bu nedenle yeni tasarimda MCTS'in isi dogrudan tek tek launch
uretmek olmayacak. MCTS daha yuksek seviyeli kararlar verecek; LP/allocator ise
bu kararlarin hangi kaynaklardan, kac gemiyle finanse edilecegini cozecek.

## Neden Yeni Bir Macro MCTS?

Su anki `TimedCaptureMCTS` sadece capture hedeflerine bakar. Bu iyi bir ilk
adimdi, cunku production potential ve arrival time gibi etkileri search icine
tasidi. Fakat oyunun tamaminda sadece saldiri karari yok:

- Kritik gezegenler dusmeden once desteklenmeli.
- Dusmesi kesin gezegenlerden gemiler kurtarilmali.
- Backline surplus, dogru frontline noktasina tasinmali.
- Bazen saldiri yerine beklemek veya savunma yapmak daha iyi olabilir.

Bu kararlar scripted fazlarda ayri ayri verildiginde birbirleriyle rekabet
etmiyorlar. Macro MCTS'in asil degeri burada: ayni turn icinde saldiri,
savunma, destek ve tahliye kararlarini ayni uzun vadeli value fonksiyonu ile
karsilastirabilir.

## Temel Ilke

Yeni agent iki seviyeli calismali:

1. **Macro MCTS** stratejik karari secer.
   - `Capture(target)`
   - `Reinforce(target)`
   - `Evacuate(source)`
   - `Consolidate(target)`
   - `Pass`

2. **LP / allocator** secilen macro karari uygular.
   - Hangi kaynak gezegenlerden gemi cikacak?
   - Kac gemi gidecek?
   - ETA kisiti saglaniyor mu?
   - Kaynak gezegen fazla zayiflatiliyor mu?

Bu ayrim onemli. RL ya da MCTS'in dogrudan exact launch uretmesi sample
inefficient olur ve path/collision detaylari yuzunden agaci cok buyutur.

## Yeni Dosya Yapisi

Bu is mevcut `orbit_timed_mcts.py` icine sikistirilmeyecek. Ayrica yazilmali:

```text
orbit_macro_mcts_agent.py
tests/test_orbit_macro_mcts_agent.py
```

Ilk dosyada beklenen temel siniflar:

```python
MacroActionType
MacroAction
MacroTimedState
MacroMCTSConfig
MacroTimedMCTS
OrbitMacroMCTSAgent
```

Ileride RL ile branch prior ogrenmek icin mevcut
`orbit_learned_mcts_agent.py` icindeki prior hook tasarimi da tekrar
kullanilabilir.

## Macro Action Tipleri

### 1. Capture

Amac neutral veya enemy gezegeni ele gecirmek.

Mevcut timed MCTS'teki capture logic buraya tasinabilir:

- kaynak gezegenler bulunur
- ETA ve source reserve hesaba katilir
- target'in arrival anindaki savunmasi tahmin edilir
- yeterli gemi varsa future arrival event olusturulur

Capture value:

```text
+ target production hold value
+ enemy production denial
- ships spent
- late arrival penalty
- source strip penalty
```

### 2. Reinforce

Amac kendi gezegenimizi dusmeden once desteklemek.

Candidate target'lar:

- `DefenseStatus.CRITICAL`
- positive `support_need`
- yuksek enemy influence altindaki owned planet

Reinforce gecerlilik kosullari:

- hedef kendi gezegenimiz olmali
- yardim enemy threat ETA'sindan once varabilmeli
- kaynak gezegen kendisi kritik olmamali
- kaynakta yeterli surplus bulunmali

Reinforce value:

```text
+ saved production hold value
+ prevented planet loss
+ preserved garrison / ships
- ships moved from source
- late or unnecessary reinforcement penalty
```

Ilk versiyonda reinforce action'i all-or-nothing olmali. Yani kritik bir
gezegeni kurtarmaya yetmiyorsa MCTS o branch'i guclu gormemeli.

### 3. Evacuate

Amac dusmesi neredeyse kesin olan gezegenden gemileri kurtarmak.

Candidate source'lar:

- kritik ve unsavable gezegenler
- comet expiration riski olan owned planet'ler
- enemy arrival sonrasi kaybedilecek gezegenler

Evacuation value:

```text
+ rescued ships
+ safe destination production/defense contribution
- abandoned production
- wrong evacuation penalty
```

Bu action ilk surume girmek zorunda degil. Capture + Reinforce stabil olduktan
sonra eklenmesi daha guvenli.

### 4. Consolidate

Amac idle backline surplus'u daha yararli frontier noktasina tasimak.

Candidate target'lar:

- enemy influence yuksek ama kritik olmayan owned planet
- gelecekte saldiri icin gateway olabilecek owned planet
- merkezi / route avantaji olan friendly planet

Consolidate value:

```text
+ future supportability
+ shorter future capture routes
+ frontline safety
- source strip risk
- unnecessary movement
```

Bu da ikinci asama action'i olmali.

### 5. Pass

Pass action gereklidir. Bazi durumlarda gemi biriktirmek veya production
beklemek daha dogrudur. Pass olmadan MCTS her zaman bir sey yapmaya zorlanir ve
gereksiz launch uretebilir.

## State Tasarimi

Mevcut timed state su alanlari tutuyor:

```python
turn
player_to_move
owners
ships
pending arrivals
reward
```

Macro MCTS icin bu iskelet korunabilir. Eklenmesi gereken sey, state uzerinden
yaklasik savunma degerlerini hesaplayan helper'lardir:

```python
enemy_inbound_by_target
friendly_inbound_by_target
defense_surplus_estimate
support_need_estimate
doomed_estimate
```

Bunlari her node'da tam `build_orbit_graph + compute_owned_defense` ile
hesaplamak cok pahali olur. Ilk surumde pending event'lerden ve mevcut ship /
production bilgisinden yaklasik hesap yapmak daha dogru.

## Candidate Pruning

Butun gezegenleri ve butun action tiplerini agaca koymak pahali olur. Her node
icin candidate sayisi sinirli olmali:

```text
top_k_capture = 4
top_k_reinforce = 3
top_k_evac = 2
top_k_consolidate = 1
include_pass = true
```

Boylece branch sayisi genelde 8-12 civarinda kalir.

Candidate siralama once heuristic ile yapilir. Sonra learned prior varsa PUCT
selection icinde kullanilir.

## Score / Reward

Macro MCTS score sadece capture income'a bakmamali. Oyun sonu degeri soyle
kurulmali:

```text
score =
    production_margin
  + ship_margin
  + planet_margin
  + capture_income_to_game_end
  + defense_hold_bonus
  + rescued_ship_bonus
  - lost_owned_production_penalty
  - wasted_reinforcement_penalty
  - source_strip_penalty
```

Savunma icin en kritik terimler:

- Bir owned planet dusmeden tutulduysa: pozitif `defense_hold_bonus`
- Owned production kaybedildiyse: negatif `lost_owned_production_penalty`
- Unsavable gezegenden gemi kurtarildiysa: pozitif `rescued_ship_bonus`
- Threat ETA'dan sonra gelen support: negatif veya gecersiz

Bu value fonksiyonu elle baslar. Daha sonra RL value head bu skoru tahmin etmek
icin egitilebilir.

## RL Branch Prior ile Iliski

Yeni learned agent'ta fikir su:

```text
state + candidate action features -> prior logits
```

Macro MCTS'te bu daha degerli hale gelir, cunku model artik sadece "hangi
capture target?" degil, "capture mi reinforce mi evacuate mi?" ayrimini da
ogrenebilir.

Candidate action feature'lari:

```text
action_type
target production
target ships
target defense surplus
enemy influence
arrival ETA
ships required
source strip risk
capture hold value
defense save value
turn fraction
```

PUCT formu:

```text
score = Q(s, a) + c_puct * prior(a) * sqrt(N(s)) / (1 + N(s, a))
```

Ilk implementation model gerektirmemeli. Basit linear prior veya heuristic
prior ile baslayip sonradan Torch model takilabilir.

## Uygulama Sirasi

### Asama 1: Capture + Reinforce Macro MCTS

Ilk surum sadece sunlari icermeli:

```text
Capture
Reinforce
Pass
```

Neden evacuation ve consolidation hemen degil?

Capture + Reinforce bile action tree'yi buyutuyor. Once savunma kararinin tree
icinde dogru calistigini gormek gerekir. Evacuation ve consolidation sonraki
asamada daha guvenli eklenir.

### Asama 2: Agent entegrasyonu

Ilk entegrasyonda scripted fazlar tamamen silinmemeli. Macro MCTS su sinyalleri
uretebilir:

```text
capture_priority_bonus
reinforce_priority_bonus
```

Sonra mevcut LP allocator bu hedefleri uygular.

Bu daha guvenli bir gecis saglar. Direkt tum agent akisini MCTS'e tasirsak
debug etmek zor olur.

### Asama 3: Evacuate ekleme

Unsavable gezegenler icin `Evacuate(source)` action'i eklenir.

Bu action'in hedef secimi:

- nearest safe friendly
- high support value friendly
- non-doomed friendly

### Asama 4: Consolidate ekleme

Backline surplus'u frontline'a tasiyan action eklenir.

Bu action, support kadar acil degildir. O yuzden daha dusuk priority ve daha
sert source-strip penalty ile baslamali.

### Asama 5: Learned prior training

Replay / self-play sirasinda her MCTS root icin sunlar kaydedilir:

```text
state features
candidate macro action features
MCTS visit counts
chosen action
final reward
terminal production margin
terminal ship margin
terminal win/loss
```

Ilk egitim hedefi:

```text
policy target = MCTS visit distribution
value target = terminal reward
```

Daha sonra self-play fine-tune yapilabilir.

## Baslangic Testleri

Yeni agent icin minimum testler:

1. Capture-only ortamda eski timed MCTS ile benzer capture bonus uretir.
2. Kritik owned planet varsa `Reinforce` action candidate olarak uretilir.
3. Threat ETA'dan sonra varacak support candidate olmaz.
4. Reinforce all-or-nothing kurtaramiyorsa dusuk deger alir veya elenir.
5. Pass action her zaman legal kalir.
6. Learned prior action type siralamasini degistirebilir.
7. Agent `act()` cagrisi env action listesi dondurur.

## Riskler

### Action space patlamasi

Butun target/source kombinasyonlarini agaca koyarsak MCTS yavaslar. Bu nedenle
action'lar target-level macro olmali; kaynak secimini LP yapmali.

### Defense hesabinin pahali olmasi

Her node'da full graph-defense hesaplamak pahali olur. Node state icinde
yaklasik defense hesaplari kullanilmali.

### Reward karisikligi

Capture income cok baskin olursa agent savunmayi ihmal eder. Defense hold ve
lost production penalty acik sekilde score'a girmeli.

### LP ile MCTS mismatch

MCTS abstract olarak feasible gordugu bir reinforce/capture kararini LP gercek
allocator cozemeyebilir. Bu nedenle MCTS candidate generation LP feasibility'ye
yaklasik bakmali, final uygulama yine fallback'li olmali.

## Sonuc

En dogru tasarim, mevcut timed MCTS'i dogrudan buyutmek yerine yeni bir
`MacroTimedMCTS` yazmak. Bu MCTS action type'lari ile calisir, sadece capture
degil reinforce ve daha sonra evacuation/consolidation kararlarini da tree icine
alir. LP lojistik cozum olarak kalir. Learned prior ise bu macro action'lar
arasinda hangi dali aramanin daha degerli oldugunu ogrenebilir.

Ilk uygulanacak minimal surum:

```text
Capture + Reinforce + Pass
```

Bu stabil olduktan sonra:

```text
Evacuate -> Consolidate -> RL prior training
```

## Uygulama Durumu (AlphaGo tarzi)

Plan asagidaki dosyalarla uctan uca kuruldu (hepsi testli):

| Dosya | Rol |
|---|---|
| `orbit_macro_mcts_agent.py` | Macro MCTS motoru: `MacroActionType`, `MacroAction`, `MacroMCTSConfig`, `MacroTimedMCTS`, `MacroSearchResult`, `OrbitMacroMCTSAgent`. Capture + Reinforce + Pass (+ opsiyonel Evacuate/Consolidate). PUCT, kok Dirichlet gurultusu, hibrit yaprak-degerlendirme ve ogrenilen prior hook'lari. Torch'suz (Kaggle uyumlu). |
| `macro_alphazero_net.py` | `MacroAlphaZeroNet` (`f(s) -> (prior logits, value)`), soyut-state featurizer, prior+value saglayicilarini aramaya baglayan `attach_net`, ve `MacroAlphaZeroAgent`. |
| `macro_selfplay.py` | Soyut modelde self-play (Option A): her karar icin `(state, ziyaret-pi, oyun sonucu z)` ornekleri uretir. |
| `train_macro_alphazero.py` | AlphaZero dongusu: self-play -> egitim (policy=ziyaret dagilimi cross-entropy, value=outcome MSE) -> soyut eval-gate ile promosyon. |
| `tests/test_orbit_macro_mcts_agent.py`, `tests/test_macro_alphazero.py` | 22 test. |

### Onemli tasarim kararlari

* **Self-play soyut modelde oynanir (Option A).** MCTS soyut modelde arar; tutarlilik icin self-play de soyut modelde oynanir. Boylece policy/value hedefleri ile arama dinamigi tek temsilde kalir (root/leaf temsil boslugu yok). Gercek-env gucu, egitilen agi gercek `orbit_wars` maclarinda gate'leyerek geri kazanilir (gelecek is).
* **Kompakt ag, agir GNN encoder'i yerine.** `ReworkedPlanetGraphEncoder` 16-D gercek gozlem ozellikleri ister; soyut ic dugumlerde bunlar yok. Bu yuzden ag, soyut-state ozellikleri uzerinde calisan permutation-invariant kucuk bir net; her dugumde (kok + yaprak) sorgulanabilir.
* **Value olcegi.** Value hedefi ham `_score` olceginde tutulur (rollout ile blend tutarli olsun diye); bu yuzden `value_coef` kucuktur. Buyuk capture-income bilesenleri value MSE'sini buyutur — gelecekte value hedef normalizasyonu eklenebilir.
* **Evacuate/Consolidate** uygulandi ama varsayilan **kapali** (`enable_evacuate`, `enable_consolidate`), boylece taban davranis Capture+Reinforce+Pass kalir.

### Kullanim

```bash
# AlphaZero tarzi egitim (soyut model self-play):
python -m train_macro_alphazero --iterations 50 --games-per-iter 32 \
    --simulations 64 --horizon-turns 12 --checkpoint-dir checkpoints_macro_alphazero

# Egitilen agi gercek macta kullanan agent: macro_alphazero_net.MacroAlphaZeroAgent
```

### Sonraki adimlar

1. Value hedef normalizasyonu (ogrenmeyi hizlandirir).
2. Gercek-env eval-gate (`orbit_match`) ile soyut-gate'i tamamlama.
3. Soyut modelin gercek env ile kalibrasyonu (model dogrulugu tavanini olcmek).
4. Evacuate/Consolidate'i acip self-play'de degerlendirme.

