# Orbit Wars — RL Eğitim Stack'i

Self-play ile PPO / DQN / Dueling DQN eğitimi, parametre bazında loglama,
ve kaggle native HTML replay export.

---

## 1. Kavramlar

```
1 epoch = N self-play oyunu + 1 update     (--games-per-epoch N)
1 oyun  = 1 episode (≤500 turn)            (--max-episode-steps 500)
1 turn  = her oyuncu 1 hamle (paralel)
```

| CLI flag | Anlamı | Default |
| --- | --- | --- |
| `--epochs` (= `--iterations`) | Toplam eğitim turu | 100 |
| `--games-per-epoch` (= `--episodes-per-iter`) | Epoch başına oyun | 4 |
| `--max-episode-steps` | Oyun başına maks turn | 500 |
| `--checkpoint-every` | Her N epoch'ta `.pt` kaydet | 10 |
| `--checkpoint-dir` | Kayıt klasörü | `checkpoints` |
| `--device` | `cpu` / `cuda` | `cpu` |

> **33s ≠ bir hamle.** Console'daki sürE bütün epoch için (örn. 5 oyun × 500
> turn = 5000 forward pass + PPO update). Hamle başına gerçek inference
> ~16ms CPU, ~3ms GPU.

---

## 2. Eğitim komutları

### PPO (önerilen başlangıç)

```bash
python -m rl.train \
    --epochs 100 --games-per-epoch 5 --max-episode-steps 500 \
    --d-model 128 --n-heads 4 --n-layers 4 \
    --batch-size 256 --ppo-epochs 4 \
    --entropy-coef 0.01 \
    --capture-value-weight 0.05 --denial-value-weight 0.05 \
    --checkpoint-dir checkpoints/ppo_run1 \
    --log-path checkpoints/ppo_run1/train_log.jsonl
```

### DQN

```bash
python -m rl.dqn_train \
    --epochs 100 --games-per-epoch 5 --max-episode-steps 500 \
    --d-model 128 --n-layers 4 \
    --replay-capacity 200000 --learn-starts 5000 \
    --batch-size 256 \
    --eps-start 1.0 --eps-end 0.05 --eps-decay-steps 100000 \
    --capture-value-weight 0.05 --denial-value-weight 0.05 \
    --checkpoint-dir checkpoints/dqn_run1
```

### Dueling DQN

```bash
python -m rl.dqn_train --dueling \
    --epochs 100 --games-per-epoch 5 \
    --replay-capacity 200000 --learn-starts 5000 \
    --eps-decay-steps 100000 \
    --checkpoint-dir checkpoints/duel_run1
```

### GPU üzerinde

Tüm yukarıdaki komutlara `--device cuda` ekle. Önce CUDA-li PyTorch:

```bash
orbit_wars_venv/bin/pip uninstall -y torch
orbit_wars_venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch
# RTX 50-series (Blackwell) için cu126 ya da nightly gerekebilir.
```

### Resume

```bash
python -m rl.train \
    --resume-from checkpoints/ppo_run1/latest.pt \
    --epochs 200 --games-per-epoch 5
```

### Hızlı sanity test (~30 sn)

```bash
python -m rl.train --epochs 5 --games-per-epoch 2 --max-episode-steps 500 \
    --d-model 64 --n-layers 2 \
    --checkpoint-dir /tmp/ppo_smoke
```

---

## 3. Reward shaping

Capture-value'lu, time-decay'li dense reward + sparse terminal:

```
reward = terminal_term
       + capture_value_weight * time_factor * d_my_production
       + denial_value_weight  * time_factor * (-d_opp_production)

time_factor = max(time_decay_floor, (max_turns - turn) / max_turns)
```

| Strateji | Komut |
| --- | --- |
| **Simetrik** (default zero-sum) | `--capture-value-weight 0.05 --denial-value-weight 0.05` |
| **Agresif** (steal/disrupt favori) | `--capture-value-weight 0.05 --denial-value-weight 0.15` |
| **Pasif** (sadece nötr büyüme) | `--capture-value-weight 0.05 --denial-value-weight 0.0` |
| **Late-game'de bile sinyal** | `--time-decay-floor 0.1` |

---

## 4. Loglama

### Console (her epoch)

```
[epoch  12/100] games=5 len=487.2 launch/p=18.4 P0=2 P1=2 D=1 |
                polL=-0.012 valL=0.34 H=4.81 KL=0.005 clip=0.08 ur=4.2e-03 (28.3s)
```

| Alan | Anlamı |
| --- | --- |
| `len` | Oyun başına ortalama turn |
| `launch/p` | Oyuncu başına lansman sayısı |
| `P0/P1/D` | 5 oyunda kim kazandı (denge sağlık göstergesi) |
| `polL`, `valL`, `H`, `KL`, `clip` | PPO/DQN training metrikleri |
| **`ur`** | **Mean update_ratio (1e-3..1e-2 ideal, lr sağlık check'i)** |

### JSONL (`train_log.jsonl`)

Her epoch için bir satır — pandas/jq ile analiz edilebilir:

```python
import pandas as pd
df = pd.read_json("checkpoints/ppo_run1/train_log.jsonl", lines=True)
df.plot(x="iter", y=["wins_p0", "wins_p1"], figsize=(12, 4))

# Per-module update ratios (sağlık takibi)
pd.json_normalize(df["update_ratios"]).plot(figsize=(12, 4), logy=True)

# Per-module gradient norms
pd.json_normalize(df["grad_norms"]).plot(figsize=(12, 4))
```

Kaydedilen alanlar:

- `iter`, `episodes`, `transitions`
- `avg_episode_length`, `avg_launches_per_player`
- `wins_p0`, `wins_p1`, `draws`
- `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`
- `learning_rate`, `epsilon` (DQN)
- **`grad_norms`** : `{module: L2_grad_norm}` (clipping öncesi)
- **`weight_norms`** : `{module: L2_weight_norm}`
- **`update_norms`** : `{module: ||Δw||}`
- **`update_ratios`** : `{module: ||Δw|| / ||w||}` ← lr sağlık check'i

`update_ratio` sağlık aralıkları:

- `< 1e-4` → öğrenmiyor (lr az)
- `1e-3 .. 1e-2` → ✓ sağlıklı
- `> 1e-1` → divergence riski (lr yüksek)

---

## 5. Checkpoint

**Otomatik isimlendirme** — sen sadece klasörü kontrol ediyorsun:

```
checkpoints/ppo_run1/
├── policy_iter00009.pt    ← her N epoch için (--checkpoint-every)
├── policy_iter00019.pt
└── latest.pt              ← her zaman en son (üzerine yazılır)
```

DQN için `qnet_iterNNNNN.pt`. `latest.pt`'i watch için kullan.

Checkpoint içerikleri (resume edilebilir):

- model state_dict
- optimizer state_dict
- iteration sayacı
- `train_cfg`, `ppo_cfg` / `dqn_cfg`, `reward_cfg`

---

## 6. Oyunu izle (3 yol)

### A. **HTML replay** (en efektif, önerilen)

Kaggle'ın native interaktif player'ı — scrub bar, play/pause, animasyonlu:

```bash
python -m rl.watch --ckpt checkpoints/ppo_run1/latest.pt \
    --html replay.html --seed 42 --max-steps 500
```

5 oyun toplu üretim:

```bash
python -m rl.watch --ckpt checkpoints/ppo_run1/latest.pt \
    --html replays/game.html --games 5 --seed 0
# -> replays/game_000.html ... game_004.html
```

Notebook inline:

```python
from rl.watch import load_any_checkpoint, record_game_html
from IPython.display import IFrame

net, _, _ = load_any_checkpoint("checkpoints/ppo_run1/latest.pt")
record_game_html(net, "game.html", max_steps=500, seed=42)
IFrame("game.html", width=900, height=650)
```

### B. **Notebook matplotlib** (snapshot tabanlı)

```python
from rl.watch import play_self_play_game
play_self_play_game(net, max_steps=500, render=True, render_every=25, seed=42)
```

Her 25 turda bir matplotlib karesi inline çizer. HTML kadar dinamik
değil ama eğitim çıktıları yanında görmek için iyi.

### C. **CLI matplotlib popup** (sadece GUI desktop)

```bash
python -m rl.watch --ckpt checkpoints/ppo_run1/latest.pt --render-every 25
```

⚠️ X11/Wayland gerekir; SSH'tan çalışmaz.

### Eğitim sırasında canlı izleme

Eğitim devam ederken paralel terminal:

```bash
# Her 30 sn'de güncel checkpoint'ten replay üret
while sleep 30; do
    python -m rl.watch --ckpt checkpoints/ppo_run1/latest.pt \
        --html replays/live.html --seed $RANDOM --max-steps 300 2>/dev/null
done
# Browser'da replays/live.html'i açık tut, F5 ile yenile.
```

---

## 7. Test

```bash
python -m pytest tests/ -q                     # tüm RL testleri (106 case)
python -m pytest tests/test_rl.py::TestMetrics -v   # sadece metrics
```

---

## 8. Notebook

Tam pipeline notebook: `orbit-wars-data/notebooks/notebooks/training_watch.ipynb`

İçinde:

1. PPO küçük eğitim
2. DQN küçük eğitim
3. Dueling DQN küçük eğitim
4. Çıktıları karşılaştır
5. Watch (matplotlib snapshot)

```bash
jupyter notebook orbit-wars-data/notebooks/notebooks/training_watch.ipynb
```

---

## 9. Pratik öneriler

- **İlk koşu**: küçük model, kısa eğitim. d=64, n_layers=2, epochs=10.
  Pipeline'ın çalıştığını gör, sonra büyüt.
- **GPU'ya geçmeden**: `update_ratio` 1e-3..1e-2'de mi? `KL` < 0.05 mi?
  Eğer evet → büyüt + GPU.
- **Win rate dengesizse** (P0=10 P1=0 gibi): self-play çökme sinyali.
  Reward shaping'i ya da learning rate'i azalt.
- **Replay büyük (~10MB)**: kaggle tüm frame'leri embed ediyor.
  Büyük taramalarda sil, sadece "interesting" oyunları sakla.
