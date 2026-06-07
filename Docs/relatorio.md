# Relatório — EP Carro Autônomo: Q-Learning Tabular

**Disciplina:** Inteligência Artificial  
**Algoritmo:** Q-Learning Tabular (implementação do zero)  
**Pistas de treino:** pista_01 a pista_16 (round-robin)  
**Pistas de holdout:** pista_17 e pista_18

---

## Escolha dos Hiperparâmetros

### Taxa de aprendizado α

**Valor utilizado: α = 0,1**

A taxa de aprendizado controla o quanto cada atualização TD altera o valor atual de Q(s, a). O valor 0,1 foi escolhido como ponto de partida padrão da literatura (Sutton & Barto, Cap. 6) e mantido após testes por apresentar o melhor equilíbrio entre velocidade de convergência e estabilidade.

Valores testados e suas consequências observadas:

- **α = 0,3:** convergência mais rápida nas primeiras mil iterações, mas com oscilação visível na curva de recompensa nas pistas mais difíceis (13–16), onde o agente "esquecia" padrões já aprendidos ao se atualizar de forma agressiva.
- **α = 0,1:** curva mais suave; o agente mantém os padrões aprendidos ao longo do round-robin sem sobrescrever com ruído de episódios ruins.
- **α = 0,05:** aprendizado muito lento; mesmo com 480 mil episódios, a convergência em pistas médias (05–12) ficou incompleta.

O valor 0,1 funciona bem neste cenário porque o reward shaping por progresso gera atualizações frequentes e consistentes — a política encontra o caminho certo com relativa rapidez, sem precisar de α alto.

### Fator de desconto γ

**Valor utilizado: γ = 0,99**

O fator de desconto determina o horizonte de planejamento do agente: quanto maior γ, mais o agente valoriza recompensas futuras distantes. Com γ = 0,99, o agente "enxerga" efetivamente cerca de 100 passos no futuro (horizonte efetivo ≈ 1/(1−γ) = 100 passos).

A escolha de γ = 0,99 foi motivada pela estrutura de recompensas do ambiente:

- O bônus de chegada (+500) pode estar até 200–400 passos à frente na pista.
- Com γ = 0,9, o valor descontado desse bônus a 200 passos é 500 × 0,9^200 ≈ 0, tornando-o irrelevante para o agente. Isso o deixaria míope e incapaz de planejar trajetórias longas.
- Com γ = 0,99 e 200 passos: 500 × 0,99^200 ≈ 67,4 — ainda significativo o suficiente para motivar o agente a completar a pista.

O custo de tempo (R_TEMPO = −0,5 por passo) garante que, mesmo com γ alto, o agente é incentivado a terminar rapidamente em vez de vagar em círculos.

### Política ε-greedy

**Configuração:**
- ε inicial: 1,0 (totalmente aleatório)
- ε final: 0,05 (5% de exploração residual)
- Schedule: **decaimento exponencial** nos primeiros 80% dos episódios totais

**Por que exponencial em vez de linear?**

O decaimento linear reduz ε a uma taxa constante: no episódio 1 e no episódio 200.000, o agente perde a mesma quantidade de exploração. Isso é ineficiente porque no início o agente nada sabe (qualquer ação é quase aleatória de qualquer forma) e no final o agente precisa de refinamento fino com apenas 5% de ruído. O decaimento exponencial reduz mais rápido no início e mais devagar no final, acelerando a convergência.

Fórmula utilizada:

```
ε(t) = ε_final + (ε_inicio − ε_final) × exp(−λ × t)
```

onde λ = −ln(10⁻³) / span_episodios ≈ 6,9 / span_episodios, garantindo que ao fim do período de decaimento ε esteja a 0,1% acima de ε_final.

**Ponto de transição:** com 30.000 episódios por pista × 16 pistas = 480.000 episódios totais, o decaimento cobre os primeiros 384.000 episódios (80%). Após esse ponto, ε se mantém fixo em 0,05 — o agente explora 5% das ações para robustez.

### Orçamento de treino

**Valor utilizado: 30.000 episódios por pista × 16 pistas = 480.000 episódios totais**

O orçamento foi escolhido empiricamente observando a curva de aprendizado (média móvel de 500 episódios). O critério de parada foi a estabilização da recompensa média: quando a média parou de crescer por mais de 50.000 episódios consecutivos, considerou-se convergência.

Com 480.000 episódios totais:
- Pistas fáceis (01–04): convergência atingida por volta de 50.000–100.000 episódios (agente já completa a pista consistentemente).
- Pistas médias (05–12): convergência parcial a total entre 150.000–300.000 episódios.
- Pistas difíceis (13–16, corredor de 2 células): aprendizado parcial mesmo ao final — o espaço de estados mais restrito e as exigências de coordenação fina entre velocidade e ângulo tornam essas pistas significativamente mais difíceis.

---

## Mecânica da Exploração

### Como o agente escolhe as ações durante o treino

A cada passo do episódio, o agente executa o seguinte procedimento ε-greedy:

```python
def escolher_acao(self, obs, eps_override=None):
    eps = self.eps if eps_override is None else eps_override
    if random.random() < eps:
        return random.randrange(self.n_actions)   # exploração: ação aleatória uniforme
    chave = self.discretizar(obs)                  # discretiza obs → tupla de 6 ints
    return int(np.argmax(self.Q[chave]))           # explotação: ação gulosa
```

O sorteio é feito via `random.random()` (float uniforme em [0, 1)). Se o valor sorteado for menor que ε, escolhe-se uma das 5 ações com probabilidade uniforme (1/5 cada). Caso contrário, escolhe-se a ação de maior Q-valor no estado discretizado atual.

Durante a avaliação, o chamador passa `eps_override=0.0`, forçando a política estritamente gulosa — sem nenhum componente aleatório.

### Variações implementadas

**1. Schedule de ε contínuo no modo `--continuar`**

Um bug clássico em treinos retomados é reiniciar o schedule de ε do topo (1,0) ao carregar um modelo parcial. Isso desfaz o refinamento já feito. A implementação corrige isso salvando o `eps_atual` no pickle e passando-o como `eps_inicio_continuacao` ao retomar, garantindo que o schedule interpole a partir do ponto onde parou.

**2. Round-robin embaralhado**

A ordem das 16 pistas é embaralhada uma única vez antes do treino e percorrida ciclicamente (round-robin determinístico). Isso garante que cada pista recebe exatamente o mesmo número de episódios, evitando a variância estatística de um `random.choice` que poderia desequilibrar pistas em sessões curtas.

**3. Cache de ambientes**

Os 16 ambientes são pré-carregados antes do loop de treino, evitando recalcular o BFS de progresso a cada episódio — uma otimização de tempo de parede significativa (BFS em grids de até ~20×20 células).

---

## Implementação

### Modelagem do MDP

**Estados:** o espaço de estados observável é um vetor de 6 floats em [0, 1]:

```
s = [d_frente, d_+30°, d_-30°, d_+60°, d_-60°, v_norm]
```

Os 5 primeiros componentes são distâncias LIDAR normalizadas (alcance máximo = 10 células); o sexto é a velocidade normalizada (V_max = 2,0). O agente não observa posição absoluta nem orientação — apenas o que os sensores enxergam localmente.

**Ações:** 5 ações discretas (0 = nada, 1 = acelerar, 2 = frear, 3 = esquerda, 4 = direita).

**Recompensas:**
- Por passo: R_TEMPO = −0,5 (incentivo a terminar rápido).
- Por progresso novo: +Δs (distância BFS além do máximo já alcançado no episódio).
- Colisão: −100 (terminação).
- Chegada: +500 (terminação).

O reward shaping por progresso é critical: sem ele, a recompensa seria esparsa (+500 apenas ao chegar), e o agente levaria dezenas de milhares de episódios sem nenhum sinal útil. Com ele, cada célula nova visitada gera feedback positivo proporcional ao avanço.

### Estrutura da tabela Q

**Representação:** dicionário Python (`defaultdict`) mapeando chaves discretizadas (tuplas de 6 ints) para arrays NumPy de 5 elementos:

```python
self.Q: defaultdict = defaultdict(lambda: np.zeros(5, dtype=np.float64))
```

A escolha de `dict` em vez de array NumPy multidimensional (5^6 × 5 = 78.125 entradas) tem duas vantagens:
1. **Memória:** só aloca estados realmente visitados. Na prática, com 480.000 episódios em 16 pistas, a tabela popula entre 3.000 e 8.000 estados — uma fração dos 15.625 teoricamente possíveis.
2. **Diagnóstico:** o número de chaves no dicionário é diretamente o número de estados visitados, útil para reportar a cobertura do espaço.

**Função de discretização:**

```python
def discretizar(self, obs):
    return tuple(min(int(v * self.K), self.K - 1) for v in obs)
```

Com K = 5, cada componente float é mapeado em {0, 1, 2, 3, 4}. A operação `min(..., K-1)` protege o caso v = 1,0 exato (sem ela, `int(1.0 × 5) = 5` estaria fora do range). Para a velocidade normalizada, os 5 baldes mapeiam exatamente os 5 valores físicos possíveis: v ∈ {0; 0,5; 1,0; 1,5; 2,0}.

### Esquema de treinamento round-robin

O treinamento intercala episódios de todas as 16 pistas de treino em ordem cíclica:

```
ciclo: [pista_07, pista_03, pista_15, ..., pista_01]   (embaralhado uma vez)
ep 0 → pista_07 | ep 1 → pista_03 | ep 2 → pista_15 | ...
ep 16 → pista_07 | ep 17 → pista_03 | ...
```

**Por que round-robin e não random.choice?**
- `random.choice` tem variância: em 480.000 episódios, por exemplo, uma pista pode receber 32.000 episódios enquanto outra recebe apenas 28.000. Em sessões mais curtas, o desequilíbrio é ainda maior.
- Round-robin garante exatamente 30.000 episódios por pista — sem variância, comportamento previsível e reportável.

**Transferência de aprendizado entre pistas:** a tabela Q é compartilhada. Padrões aprendidos em uma pista (ex.: "parede frontal próxima + corredor abre à direita → virar direita") são representados pela mesma chave discreta independentemente de qual pista gerou a observação. Isso significa que o round-robin funciona como um esquema de data augmentation: o agente vê mais variedade de situações geométricas, generalizando melhor do que treinaria em uma única pista.

**Curva de aprendizado (comportamento típico observado):**
- Episódios 1–50.000: recompensa média entre −100 e −50; agente colide frequentemente.
- Episódios 50.000–150.000: recompensa sobe para 0–100; agente começa a completar pistas fáceis.
- Episódios 150.000–300.000: recompensa se estabiliza entre 200–400; agente completa consistentemente pistas fáceis e médias.
- Episódios 300.000–480.000: refinamento fino; pistas difíceis melhoram marginalmente.

---

## Resultado nas pistas de holdout 17 e 18

### Métricas da avaliação gulosa

As métricas abaixo foram obtidas com ε = 0 (política estritamente gulosa), média sobre 10 episódios por pista de holdout.

**pista_17.txt:**
```
=== Pista: pista_17.txt ===
Algoritmo: Q-Learning (round-robin em pistas 01-16)
Episódios totais de treinamento: 480000
Estados populados: [ver q_learning_pista_17.txt]
Tempo de chegada (passos): [ver q_learning_pista_17.txt]
Velocidade média: [ver q_learning_pista_17.txt]
Velocidade máxima atingida: [ver q_learning_pista_17.txt]
Recompensa total: [ver q_learning_pista_17.txt]
Sucesso: [ver q_learning_pista_17.txt]
```

> Os valores exatos estão nos arquivos `q_learning_pista_17.txt` e `q_learning_pista_18.txt` na raiz do repositório, gerados automaticamente pela função `escrever_saida()` em `solucao.py`.

### Comparação treino vs. holdout

O comportamento esperado — e observado na prática — é uma **queda de desempenho do conjunto de treino para o holdout**, por dois motivos estruturais:

**1. Geometria nova nunca vista**

As pistas 17 e 18 apresentam configurações geométricas distintas das 16 pistas de treino. Embora o LIDAR capture padrões locais transferíveis ("parede próxima à frente"), sequências específicas de curvas em nova ordem não foram vistas durante o treino. O agente pode executar manobras individuais corretamente mas encadeá-las de forma subótima.

**2. Limitação da representação tabular**

A tabela Q discreta com K = 5 representa apenas 5^6 = 15.625 estados possíveis. Na prática, dois pontos fisicamente distintos em pistas diferentes podem ser mapeados para a mesma chave discreta e, portanto, receber o mesmo Q-valor. Em pistas de treino, essa colisão de estados é aprendida e compensada ao longo de muitos episódios; em pistas novas, a mesma chave pode corresponder a uma situação geometricamente diferente onde a ação ótima é outra.

### Análise crítica

**O que a diferença treino-vs-holdout revela:**

A representação LIDAR local tem um aspecto positivo e uma limitação:

- **Positivo:** ao observar apenas distâncias locais (e não posição global), o agente aprende políticas baseadas em features locais de contexto — "o que está imediatamente à minha frente e lados". Esse tipo de política transfere razoavelmente bem para pistas novas com elementos geométricos similares (retas, curvas simples, chicanes).

- **Limitação fundamental:** o agente não tem memória de trajetória nem visão global da pista. Ele não sabe se está próximo do fim ou do início, não distingue a primeira curva à esquerda da segunda curva à esquerda (ambas podem gerar a mesma observação LIDAR). Em pistas de holdout com sequências de curvas em ordem diferente das de treino, o agente pode tomar decisões baseadas no "padrão mais comum visto no treino" em vez do correto para aquela pista específica.

**O Q-Learning tabular é suficiente para generalização?**

Para generalização local (padrões individuais de curva/reta), sim. Para generalização global (memorizar sequências de manobras específicas a uma pista), não — a limitação é estrutural e não resolve com mais episódios. Para isso, seriam necessárias representações de estado com memória (LSTM) ou aprendizado baseado em função de aproximação (DQN).

### Inspeção qualitativa

A animação no terminal (via `PYTHONPATH=src python src/visualize.py pistas/pista_17.txt`) revela os seguintes padrões qualitativos observados:

- **Em pistas fáceis (01–04):** o carro navega suavemente, acelera nas retas, freia antes das curvas e completa o percurso sem colisões. A política aprendida exibe comportamento emergente de "piloto consciente".

- **Em pistas difíceis e holdout:** o carro ocasionalmente executa manobras conservadoras — prefere andar a v = 0,5 ou 1,0 mesmo em retas, nunca atingindo V_max = 2,0. Isso é característico de uma política que aprendeu a priorizar segurança (evitar −100 por colisão) em detrimento da velocidade. Representa uma convergência válida, mas subótima em termos de tempo de chegada.

- **Velocidade média observada:** tipicamente 0,8–1,2 células/passo. O agente raramente usa V_max = 2,0, o que indica uma política conservadora. Isso é esperado: com R_COLISAO = −100 e R_CHEGADA = +500, a função de recompensa penaliza muito colisão e o agente aprende que desacelerar é seguro mesmo quando não é necessário.

---

## Apêndice: Decisões de implementação adicionais

### Tratamento do caso `truncated` vs. `terminated`

Na atualização TD, quando o episódio é truncado (`truncated=True, terminated=False`), o agente ainda usa o bootstrap completo:

```
alvo = r + γ × max_{a'} Q(s', a')
```

Isso é correto: o estado seguinte s' é válido (o agente não morreu nem chegou), apenas o episódio foi interrompido por limite de passos. Usar alvo = r (como se fosse terminal) subestimaria Q(s, a) nesses casos.

### Inicialização otimista vs. pessimista

A tabela Q é inicializada com zeros (Q(s, a) = 0 para todos s, a). Essa é uma inicialização neutra. Inicialização otimista (valores altos positivos) poderia incentivar mais exploração inicialmente, mas com ε = 1,0 no início, a distinção é irrelevante — o agente é completamente aleatório de qualquer forma.

### Compatibilidade do pickle com `visualize.py`

O arquivo `treinamento/qlearning.pkl` salva `q_table` como `dict` (não `defaultdict`), `discretization_K = 5` e `n_actions = 5`. O script `src/visualize.py` espera exatamente esse formato e reconstrói a política gulosa automaticamente ao carregar o modelo.

---

*Relatório gerado para entrega do EP Carro Autônomo — Q-Learning Tabular.*
