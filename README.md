# 🏎️ Carro Autônomo — Q-Learning Tabular

Implementação de um agente de **aprendizado por reforço** que aprende a pilotar um carrinho em pistas 2D usando **Q-Learning tabular**. O agente não vê o mapa — ele navega apenas com 5 sensores LIDAR e a própria velocidade, sem conhecer posição ou orientação absoluta.

---

## Como funciona

O carrinho percorre pistas representadas em grid. A cada passo ele recebe uma observação de 6 valores:

```
[distância_frente, dist_+30°, dist_-30°, dist_+60°, dist_-60°, velocidade]
```

E escolhe uma de 5 ações: **nada · acelerar · frear · virar esquerda · virar direita**.

O agente aprende por tentativa e erro: recebe recompensa positiva por avançar na pista, −100 por colidir e +500 por cruzar a linha de chegada. O Q-Learning tabular com discretização K=5 permite representar esse espaço em ~15.625 estados possíveis.

---

## Estrutura do repositório

```
rf-carro-autonomo/
├── solucao.py              ← implementação do Q-Learning (ponto de entrada)
├── src/
│   ├── env.py              ← ambiente do carrinho (física + LIDAR + recompensas)
│   ├── track.py            ← parser de pistas em emoji
│   └── visualize.py        ← animação no terminal
├── pistas/
│   ├── pista_01.txt … pista_16.txt   ← pistas de treino
│   └── pista_17.txt, pista_18.txt    ← pistas de holdout (avaliação)
├── treinamento/
│   └── qlearning.pkl       ← modelo treinado (gerado ao rodar solucao.py)
├── docs/
│   └── relatorio.md        ← relatório completo do projeto
└── enunciado/              ← documentação de apoio (LIDAR, Q-Learning, etc.)
```

---

## Requisitos

- Python 3.10+
- numpy

```bash
pip install numpy
```

Nenhuma biblioteca de RL externa (gymnasium, stable-baselines, torch) é utilizada — o Q-Learning é implementado do zero.

---

## Como rodar

### 1. Treinar o agente

Treina em round-robin nas 16 pistas de treino (30.000 episódios por pista = 480.000 no total) e salva o modelo em `treinamento/qlearning.pkl`:

```bash
python solucao.py
```

Se o arquivo `.pkl` já existir, o treino é pulado e o modelo é carregado diretamente para avaliação. Para forçar re-treino:

```bash
python solucao.py --recarregar
```

Para continuar um treino interrompido (adiciona mais episódios ao modelo existente):

```bash
python solucao.py --continuar
python solucao.py --continuar --episodios-por-pista 10000  # sessão mais curta
```

### 2. Ver o agente em ação (animação no terminal)

```bash
PYTHONPATH=src python src/visualize.py pistas/pista_01.txt
```

O script carrega automaticamente `treinamento/qlearning.pkl` e anima o agente pilotando a pista escolhida. Funciona em qualquer terminal moderno (macOS, Linux, Windows Terminal).

```bash
# Qualquer pista funciona:
PYTHONPATH=src python src/visualize.py pistas/pista_17.txt
```

### 3. Avaliar numa pista específica

```bash
python solucao.py --avaliar pistas/pista_17.txt
```

Roda 10 episódios com política gulosa (ε = 0) e imprime as métricas: passos até a chegada, velocidade média, recompensa total e taxa de sucesso.

### 4. Opções da linha de comando

| Flag | Padrão | Descrição |
|---|---|---|
| `--episodios-por-pista` | 30000 | Episódios de treino por pista |
| `--max-passos` | 500 | Limite de passos por episódio |
| `--K` | 5 | Baldes da discretização (K^6 estados possíveis) |
| `--recarregar` | — | Força re-treino do zero |
| `--continuar` | — | Retoma o treino a partir do pkl existente |
| `--avaliar` | — | Avalia o modelo numa pista específica (pula treino) |
| `--n-avaliacao` | 10 | Episódios de avaliação gulosa |

---

## Saída esperada

Ao rodar `python solucao.py`, o terminal exibe progresso em tempo real:

```
  ep   1000/480000 [ 0.2%]  ε=0.982  reward=  -87.3  sucesso=  0.0%  estados=   312  ...
  ep  50000/480000 [10.4%]  ε=0.631  reward=   43.1  sucesso= 12.4%  estados=  1847  ...
  ep 480000/480000 [100%]   ε=0.050  reward=  318.7  sucesso= 71.2%  estados=  5923  ...
```

Ao final, gera `q_learning_pista_17.txt` e `q_learning_pista_18.txt` com as métricas de holdout.

---

## Documentação adicional

- [`docs/relatorio.md`](docs/relatorio.md) — justificativa dos hiperparâmetros, análise de generalização e resultados detalhados.
- [`enunciado/qlearning.md`](enunciado/qlearning.md) — matemática do Q-Learning (equação de Bellman, atualização TD, convergência).
- [`enunciado/discretizacao.md`](enunciado/discretizacao.md) — como o vetor de floats vira chave da tabela Q.
- [`enunciado/anexo_a_lidar.md`](enunciado/anexo_a_lidar.md) — o que são sensores LIDAR e como são simulados.
- [`enunciado/anexo_c_velocidade.md`](enunciado/anexo_c_velocidade.md) — por que velocidade é o componente mais difícil de aprender.
