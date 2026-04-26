# Yapılacaklar

## Agent'a Verilecek Observation Feature'ları

`reachability_scan` çıktısını agent observation'ına eklemek. Tek bir kaynak gezegen için her hedefe `(k_wait, eta, total_time, feasible)` vektörü.

### Önerilen Format

Agent'ın gördüğü gözlem zenginleştirmesi:

```python
# Her (kendi_gezegen, hedef) çifti için:
{
    "src_id": int,
    "tgt_id": int,
    "feasible": bool,
    "k_wait": int,            # 0 = şimdi fırlat, k>0 = beklemek gerekiyor
    "eta": float,             # uçuş süresi (turn)
    "total_time": float,      # k_wait + eta
    "min_ships_to_capture": int,    # opsiyonel, üretim hesaba katılarak
    "value_score": float,     # opsiyonel, p_target * (T_end - total_time)
}
```

### Yapılacak Adımlar

- [ ] `agent_features.py` modülü: `build_reachability_features(obs, player) -> dict` veya `np.ndarray`.
- [ ] Boyut sabiti tanımla: `MAX_PLANETS = 40` (env limiti) → tensor şekli `(MAX_PLANETS, MAX_PLANETS, F)` veya bipartite `(MY_PLANETS, ALL_PLANETS, F)`.
- [ ] Padding/masking stratejisi: feasible=False olanlara `total_time=∞` yerine büyük sabit (ör 999) + maske bayrağı.
- [ ] Cache: aynı tur içinde her gezegen için bir kez hesapla.
- [ ] `min_ships_to_capture(src, tgt)` helper'ı (binary search, navigation.py'ye eklenecek).
- [ ] Performance budget: 1 sn limit içinde 40×40×30-horizon scan yapılabiliyor mu? Profiling.
- [ ] Notebook test hücresi: feature tensoru build et, shape ve örnek değerleri yazdır.

### Açık Sorular

- Agent neural net mi rule-based mi olacak? Şekil ona göre.
- Tek-tur observation mı, yoksa tarihsel pencere de eklenecek mi?
- `reachability_scan` her tur tüm `(src, tgt)` için → 40×40 = 1600 çağrı; her biri ms altı ama toplam 1-2 saniye sürebilir. Caching/precompute gerekecek.

---

## Diğer Yapılacaklar

### Navigation/Strategy
- [ ] `min_ships_to_capture(src, tgt)` — binary search, `s ≥ G + p·t*(s) + 1` örtük denklemini çözer.
- [ ] `optimal_ship_count(src, tgt)` — `dV/ds = 0` numerik çözüm (zaman vs maliyet tradeoff).
- [ ] `dijkstra_travel_times(src, planets)` — çok-hop shortest path, time-dependent.
- [ ] `tangent_around_sun(A, B)` — teorik kısa yol formülü, waypoint seçimi için.
- [ ] **Kombineli saldırı senkronizasyonu** — iki kaynaktan aynı turda varış (`launch_time_i = T_arr - eta_i`).

### Playground / Visualization
- [ ] `pg.render(animate=True)` — `pg.step()`'i otomatik tekrarlayıp matplotlib animation üretsin.
- [ ] `pg.render` üzerine `reachability_scan` overlay'i: feasible hedeflere ince yeşil çizgi, infeasible'lara kırmızı X.
- [ ] Filo trail (geçtiği yolu çiz) — debug için faydalı.

### Defense / Reactive
- [x] Düşman filolarının ETA hesaplama — `predict_fleet_target` + `incoming_threats` (navigation.py).
- [x] Render overlay: tahmini intercept noktası (kesikli + halka).
- [ ] Garrison ihtiyacı: `defenders_needed = sum(incoming_ships) - my_ships - production_until_arrival`.

### Tests
- [x] `tests/test_navigation.py` — 50 test, pytest, < 0.1s.
- [ ] Comet kinematiği için ayrı bir test grubu (`position_at` path interpolation).
- [ ] Playground smoke test: `Playground().step()` 10 tur crash'siz.
- [ ] `path_collision` rotating engelleyici zaman-bağımlı yakalama testi (pozitif + negatif case).

---

## Tamamlananlar

- [x] `feature_extraction.Planet` — observation row → rich object.
- [x] `feature_extraction.Fleet` + `parse_fleets`.
- [x] `navigation.find_intercepts` — multiple-lap intercept solver.
- [x] `navigation.crosses_sun`, `path_collision`.
- [x] `navigation.compute_launch` — top-level launch computer.
- [x] `navigation.find_intercepts_from`, `path_collision_from`, `compute_launch_at` — delayed-launch variants.
- [x] `navigation.earliest_reachable`, `reachability_scan`.
- [x] `playground.Playground` — interactive game wrapper.
- [x] `playground.render_board` — matplotlib visualization.
- [x] `docs/math.md` + 5 figür.
