# Orbit Wars Tutorials

Bu klasor RL notlarini okuma sirasi ve kavram haritasi olarak duzenler. Notebooklar teorik agirliklidir, ama Orbit Wars kodundaki karsiliklari da gosterir.

## Onerilen Okuma Sirasi

1. `rl_temel_kavramlar_teori.ipynb`
   - On-policy/off-policy, bootstrap, model-free/model-based ve value/policy/actor-critic ayrimlarini kurar.
2. `dqn_teori.ipynb`
   - Q-learning, TD target, TD error, replay buffer ve target network temelini kurar.
3. `duelling_dqn_teori.ipynb`
   - DQN mimarisindeki value/advantage ayrimini anlatir.
4. `policy_gradient_teori.ipynb`
   - REINFORCE, baseline, actor-critic, GAE ve PPO icin genel cerceve verir.
5. `reinforce_baseline_teori.ipynb`
   - Policy gradient'in en temel algoritmasini ve baseline'in neden bias eklemedigini detaylandirir.
6. `actor_critic_teori.ipynb`
   - Actor, critic, TD residual, advantage ve value loss iliskisini netlestirir.
7. `a2c_a3c_teori.ipynb`
   - Senkron/asenkron actor-critic rollout mantigini anlatir.
8. `ppo_trpo_teori.ipynb`
   - PPO clipping ve TRPO trust region fikrini ayrintilandirir.

## Kisa Kavram Haritasi

| Kavram | En ilgili notebook | Kisa anlam |
| --- | --- | --- |
| TD target | `dqn_teori.ipynb`, `actor_critic_teori.ipynb` | Mevcut tahminin yaklasmasi istenen bootstrap hedefi |
| TD error / residual | `dqn_teori.ipynb`, `actor_critic_teori.ipynb` | Hedef ile mevcut tahmin arasindaki fark |
| On-policy | `rl_temel_kavramlar_teori.ipynb`, `policy_gradient_teori.ipynb` | Veri toplayan politika ile ogrenilen politikanin ayni veya cok yakin olmasi |
| Off-policy | `rl_temel_kavramlar_teori.ipynb`, `dqn_teori.ipynb` | Eski/farkli behavior policy verisinden ogrenebilme |
| Bootstrap | `rl_temel_kavramlar_teori.ipynb`, `dqn_teori.ipynb` | Hedefte sonraki durum tahminini kullanma |
| Replay buffer | `dqn_teori.ipynb` | Off-policy gecisleri karisik minibatch olarak tekrar kullanma |
| Target network | `dqn_teori.ipynb` | TD hedefini stabil tutan yavas guncellenen ag |
| Advantage | `policy_gradient_teori.ipynb`, `actor_critic_teori.ipynb` | Aksiyonun ayni durumdaki beklenene gore farki |
| Baseline | `reinforce_baseline_teori.ipynb` | Gradyan varyansini azaltan, duruma bagli referans |
| GAE | `policy_gradient_teori.ipynb`, `actor_critic_teori.ipynb` | Cok adimli TD residual karisimiyle advantage tahmini |
| PPO clipping | `ppo_trpo_teori.ipynb` | Politika update'ini eski politikadan fazla uzaklastirmama |

## Orbit Wars Kod Eslesmesi

| Kod | Ilgili teori |
| --- | --- |
| `model_q.py` | DQN / Dueling DQN Q agi, branch aksiyon basliklari |
| `rl/dqn.py` | TD target, Double DQN hedefi, Huber/MSE loss |
| `rl/dqn_train.py` | Replay buffer, epsilon-greedy self-play, target sync |
| `model.py` | Actor-Critic policy/value agi |
| `rl/ppo.py` | PPO clipped policy loss, value loss, entropy, KL metrikleri |
| `rl/rollout.py` | Rollout, return ve advantage hazirligi |
