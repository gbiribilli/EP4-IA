"""
Solução do EP do carrinho — Q-Learning Tabular.

Implementa:
- AgenteQLearning (tabular, discretização K=5, ε-greedy com decaimento linear)
- treinar_round_robin: treinamento em round-robin nas pistas 01-16
- avaliar: avaliação gulosa (ε=0) em qualquer pista
- treinar_ou_carregar: salva/carrega modelo via pickle
- escrever_saida: gera arquivos de saída no formato do README §4.3
- main(): orquestra treinamento + avaliação holdout

Uso:
    python solucao.py                           # treina do zero (ou carrega pkl existente) + avalia holdout
    python solucao.py --recarregar              # força re-treino do zero
    python solucao.py --continuar               # continua treinamento a partir do pkl existente
    python solucao.py --continuar --episodios-por-pista 10000  # +10k eps/pista e reavalia
    python solucao.py --avaliar pistas/X.txt   # avalia modelo salvo numa pista específica

Termos canônicos de RL (step, reset, obs, action, reward, terminated,
truncated, info) mantidos em inglês conforme Sutton & Barto / Gymnasium.
"""

from __future__ import annotations

import sys
import math
import random
import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402

# from visualize import renderizar_episodio  # descomente para animar no terminal

# === Configuração global ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

PISTAS_TREINO  = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]  # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)] # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Agente Q-Learning tabular com discretização binning uniforme.

    Estado: vetor de 6 floats em [0,1] → tupla de 6 ints em {0,...,K-1}.
    Tabela Q: dict {chave: np.ndarray(n_actions)} — só aloca estados visitados.
    Política: ε-greedy; durante avaliação, chame escolher_acao com eps_override=0.

    Hiperparâmetros padrão:
        K=5      → K^6 = 15.625 estados; casa com os 5 níveis de velocidade.
        alpha=0.1 → taxa de aprendizado moderada, estável.
        gamma=0.99 → valoriza recompensas futuras (meta +500 a distância).
        eps_inicial=1.0 → começa totalmente exploratório.
        eps_final=0.05  → termina quase guloso (5% de exploração residual).
    """

    def __init__(
        self,
        obs_dim: int = 6,
        n_actions: int = 5,
        K: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_inicial: float = 1.0,
        eps_final: float = 0.05,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_final = eps_final

        # Tabela Q: chave discreta → vetor de Q-valores por ação.
        # defaultdict evita verificação manual de "estado nunca visto".
        self.Q: dict = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

    # ------------------------------------------------------------------
    # Discretização
    # ------------------------------------------------------------------

    def discretizar(self, obs: np.ndarray) -> tuple:
        """
        Converte vetor de floats em [0,1] para tupla de ints em {0,...,K-1}.

        Fórmula: balde = min(int(v * K), K-1)
        O min(..., K-1) protege v=1.0 exato (sem ele geraria índice K).

        Exemplo com K=5: obs=[0.35, 1.0, 0.30, 0.41, 0.18, 0.50]
                          → chave=(1, 4, 1, 2, 0, 2)
        """
        return tuple(min(int(v * self.K), self.K - 1) for v in obs)

    # ------------------------------------------------------------------
    # Política ε-greedy
    # ------------------------------------------------------------------

    def escolher_acao(self, obs: np.ndarray, eps_override: Optional[float] = None) -> int:
        """
        Seleciona ação pela política ε-greedy.

        Args:
            obs: vetor de observação (6 floats em [0,1]).
            eps_override: se fornecido, usa esse ε em vez de self.eps
                          (útil para avaliação com eps=0).

        Returns:
            Ação escolhida (int em {0,...,4}).
        """
        eps = self.eps if eps_override is None else eps_override
        if random.random() < eps:
            return random.randrange(self.n_actions)
        chave = self.discretizar(obs)
        return int(np.argmax(self.Q[chave]))

    # ------------------------------------------------------------------
    # Atualização TD (regra do Q-Learning)
    # ------------------------------------------------------------------

    def atualizar(
        self,
        obs: np.ndarray,
        a: int,
        r: float,
        obs_prox: np.ndarray,
        terminou: bool,
    ) -> None:
        """
        Aplica a regra de update do Q-Learning:
            Q(s,a) ← Q(s,a) + α [alvo − Q(s,a)]

        Onde:
            alvo = r                          (se terminou)
            alvo = r + γ max_{a'} Q(s', a')   (caso contrário)

        Nota: é off-policy porque o alvo usa max sobre Q(s'), não a
        ação que o agente vai de fato tomar em s'.
        """
        s  = self.discretizar(obs)
        sp = self.discretizar(obs_prox)

        if terminou:
            alvo = r
        else:
            alvo = r + self.gamma * np.max(self.Q[sp])

        self.Q[s][a] += self.alpha * (alvo - self.Q[s][a])

    # ------------------------------------------------------------------
    # Serialização / reconstituição a partir do pickle
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serializa apenas o essencial: Q-table convertida para dict normal."""
        return {
            "Q": dict(self.Q),   # defaultdict → dict para pickle limpo
            "K": self.K,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

    @classmethod
    def from_modelo(cls, modelo: dict) -> "AgenteQLearning":
        """
        Reconstrói o agente a partir do dicionário carregado do pickle.
        Preserva o eps atual caso estejamos continuando um treinamento.
        """
        agente = cls(
            n_actions=modelo.get("q_table_n_actions", 5),
            K=modelo.get("discretization_K", 5),
            alpha=modelo.get("config", {}).get("alpha", 0.1),
            gamma=modelo.get("config", {}).get("gamma", 0.99),
            eps_final=modelo.get("config", {}).get("eps_final", 0.05),
        )
        q_raw = modelo["q_table"]
        # Suporta tanto dict quanto np.ndarray multidimensional
        if isinstance(q_raw, dict):
            for k, v in q_raw.items():
                agente.Q[k] = np.array(v, dtype=np.float64)
        else:
            # np.ndarray multidimensional: converte para dict
            for idx in np.ndindex(*q_raw.shape[:-1]):
                agente.Q[idx] = q_raw[idx].copy()

        # Restaura o eps onde o treinamento anterior parou.
        # Se for carregar para avaliação, o chamador seta eps_override=0.
        agente.eps = modelo.get("eps_atual", 0.0)
        return agente

    @property
    def n_estados_populados(self) -> int:
        """Número de estados (chaves) que já foram visitados ao menos uma vez."""
        return len(self.Q)


# ============================================================================
# SCHEDULE DE EPSILON
# ============================================================================

def schedule_epsilon(
    ep_global: int,
    eps_inicial: float,
    eps_final: float,
    eps_decai_em: int,
    eps_global_inicio: float = 1.0,
    ep_global_offset: int = 0,
) -> float:
    """
    Decaimento EXPONENCIAL de eps_global_inicio a eps_final ao longo de
    eps_decai_em episódios globais. Após eps_decai_em, mantém eps_final fixo.

    Por que exponencial em vez de linear?
    - Linear reduz epsilon a ritmo constante: no episódio 1 e no 200.000 o
      agente perde a mesma quantidade de exploração, mesmo já sabendo muito mais.
    - Exponencial reduz mais rápido no início (quando o agente nada sabe e
      qualquer ação é quase aleatória de qualquer jeito) e mais devagar no fim
      (quando o agente já converge e precisa de refinamento fino).
    - Na prática isso acelera convergência e melhora a qualidade final da
      política, especialmente em espaços de estados maiores (K alto).

    Fórmula: eps(t) = eps_final + (eps_inicio - eps_final) * exp(-lambda * t)
    onde t = episódios decorridos desde ep_global_offset e
    lambda é calculado para que eps(eps_decai_em) chegue perto de eps_final.

    Args:
        ep_global: episódio atual no contexto global acumulado.
        eps_inicial: mantido por compatibilidade de assinatura (não usado diretamente).
        eps_final: valor mínimo de epsilon após o decaimento.
        eps_decai_em: episódio global em que epsilon atinge eps_final.
        eps_global_inicio: valor de epsilon no ep_global_offset. Para treino do
                           zero é 1.0; para --continuar é o eps_atual do pickle.
        ep_global_offset: episódio global em que esta sessão começa.
    """
    if ep_global >= eps_decai_em:
        return eps_final

    span_ep = eps_decai_em - ep_global_offset
    if span_ep <= 0:
        return eps_final

    t = ep_global - ep_global_offset  # passos decorridos desde o início desta sessão

    # lambda tal que exp(-lambda * span_ep) = residual_alvo ~= 0
    # residual de 1e-3 significa que em eps_decai_em a curva chegou a 0.1%
    # acima de eps_final — próximo o suficiente sem divisão por zero
    residual_alvo = 1e-3
    lam = -math.log(residual_alvo) / span_ep  # ~= 6.9 / span_ep

    frac = math.exp(-lam * t)
    return eps_final + (eps_global_inicio - eps_final) * frac


# ============================================================================
# LOOP DE TREINAMENTO (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(
    pistas_treino: list[str],
    agente: AgenteQLearning,
    n_episodios_novos: int,
    max_passos: int,
    ep_offset: int,
    eps_decai_em: int,
    eps_inicio_continuacao: Optional[float] = None,
    verbose: bool = True,
) -> tuple[list, list, dict]:
    """
    Treinamento em round-robin determinístico: a lista de pistas é embaralhada
    uma vez e percorrida ciclicamente. Cada pista recebe exatamente o mesmo
    número de episódios — sem a variância do random.choice, que em sessões
    curtas pode desequilibrar pistas. A tabela Q é compartilhada — o agente
    aprende padrões locais (via LIDAR) que transferem entre pistas.

    Evita catastrophic forgetting: ao intercalar episódios de pistas diferentes,
    o agente não "esquece" o que aprendeu em uma pista ao treinar outra.

    Args:
        pistas_treino: lista de caminhos para as pistas de treino.
        agente: instância de AgenteQLearning (modificada in-place).
        n_episodios_novos: quantos episódios rodar nesta sessão.
        max_passos: limite de passos por episódio.
        ep_offset: número de episódios já treinados antes desta sessão.
                   Usado para posicionar o ε corretamente no schedule global.
        eps_decai_em: episódio global em que ε atinge eps_final.
        eps_inicio_continuacao: ε no momento em que esta sessão começa.
                                None = treino do zero (usa 1.0). Ao usar
                                --continuar, passe o eps_atual salvo no pickle
                                para evitar que o schedule reinicie do topo.
        verbose: se True, imprime progresso.

    Returns:
        historico_recompensas: lista de rewards desta sessão.
        historico_sucessos: lista de booleans desta sessão.
        rewards_por_pista: dict pista → lista de rewards desta sessão.

    CORREÇÃO: parâmetro eps_inicio_continuacao adicionado para corrigir o bug
    onde --continuar fazia o schedule_epsilon subir o ε de volta a 1.0.
    """
    historico_recompensas = []
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}

    # Se não for continuação, o ε começa em 1.0
    eps_global_inicio = eps_inicio_continuacao if eps_inicio_continuacao is not None else 1.0

    # Cache de ambientes — evita recalcular o BFS a cada episódio
    # seed=SEED por consistência — o rng do ambiente não é usado em reset() nem
    # em step(), então não afeta aleatoriedade. A posição inicial é sempre fixa
    # (celula_largada + 0.5, theta=0, v=0) conforme implementação do env.py.
    print("Pré-carregando ambientes...")
    envs = {p: AmbienteCarro(p, max_steps=max_passos, seed=SEED) for p in pistas_treino}
    print(
        f"Ambientes prontos. Iniciando {n_episodios_novos} episódios "
        f"(offset global: {ep_offset})..."
    )

    t_inicio = time.time()
    t_ultimo_print = t_inicio - 1.0  # força impressão imediata no 1º episódio

    # Round-robin determinístico: embaralha a lista uma vez e cicla por ela.
    # Garante que cada pista recebe exatamente o mesmo número de episódios —
    # sem a variância estatística do random.choice, que pode desequilibrar
    # pistas em sessões curtas.
    pistas_ciclicas = pistas_treino[:]
    random.shuffle(pistas_ciclicas)
    n_pistas = len(pistas_ciclicas)

    for ep_local in range(n_episodios_novos):
        ep_global = ep_offset + ep_local

        # Atualiza epsilon usando o número de episódio GLOBAL — assim o
        # decaimento é contínuo mesmo ao continuar um treinamento anterior.
        # CORREÇÃO: passa ep_offset e eps_global_inicio para que o schedule
        # interpole a partir do ponto atual, não do início absoluto.
        agente.eps = schedule_epsilon(
            ep_global,
            eps_inicial=1.0,
            eps_final=agente.eps_final,
            eps_decai_em=eps_decai_em,
            eps_global_inicio=eps_global_inicio,
            ep_global_offset=ep_offset,
        )

        # Round-robin determinístico: percorre as pistas na ordem embaralhada,
        # reiniciando o ciclo ao esgotar todas. Assim cada pista recebe
        # exatamente floor(n_episodios_novos / n_pistas) episódios, com as
        # primeiras (n_episodios_novos % n_pistas) recebendo um episódio a mais.
        pista = pistas_ciclicas[ep_local % n_pistas]
        env = envs[pista]

        # Loop do episódio
        obs = env.reset()
        reward_total = 0.0
        sucesso = False

        done = False
        while not done:
            action = agente.escolher_acao(obs)
            obs_prox, reward, term, trunc, info = env.step(action)
            agente.atualizar(obs, action, reward, obs_prox, term)
            obs = obs_prox
            reward_total += reward
            done = term or trunc
            if done:
                sucesso = info.get("chegada", False)

        historico_recompensas.append(reward_total)
        historico_sucessos.append(sucesso)
        rewards_por_pista[pista].append(reward_total)

        # Atualiza status em tempo real: reescreve a mesma linha no terminal
        # a cada segundo usando \r (carriage return), sem rolar o histórico.
        if verbose:
            agora = time.time()
            if agora - t_ultimo_print >= 1.0:
                t_ultimo_print = agora
                elapsed = agora - t_inicio
                progresso = (ep_local + 1) / n_episodios_novos
                eps_restante = elapsed / progresso - elapsed if progresso > 0 else 0.0
                ultimos = historico_recompensas[-500:]
                media = np.mean(ultimos)
                taxa_suc = np.mean(historico_sucessos[-500:]) * 100
                estados = agente.n_estados_populados

                # Formata tempo restante em hh:mm:ss
                def fmt_tempo(s: float) -> str:
                    s = int(s)
                    h, m = divmod(s, 3600)
                    m, s = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                linha = (
                    f"\r  ep {ep_global+1:>7}/{ep_offset+n_episodios_novos:<7} "
                    f"[{progresso*100:>5.1f}%]  "
                    f"ε={agente.eps:.3f}  "
                    f"reward={media:>8.1f}  "
                    f"sucesso={taxa_suc:>5.1f}%  "
                    f"estados={estados:>6}  "
                    f"decorrido={fmt_tempo(elapsed)}  "
                    f"restante≈{fmt_tempo(eps_restante)}"
                )
                # Padding para apagar resíduos de linhas anteriores mais longas
                print(linha.ljust(120), end="", flush=True)

    elapsed_total = time.time() - t_inicio
    print()  # encerra a linha de status em tempo real
    print(f"Sessão concluída em {elapsed_total:.1f}s")
    print(f"Estados populados na Q-table: {agente.n_estados_populados}")
    taxa_final = np.mean(historico_sucessos[-min(5_000, len(historico_sucessos)):]) * 100
    print(f"Taxa de sucesso (últimos {min(5_000, len(historico_sucessos))} eps): {taxa_final:.1f}%")

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(
    env: AmbienteCarro,
    agente: AgenteQLearning,
    n_episodios: int = 10,
) -> dict:
    """
    Avalia o agente com política gulosa (ε=0) por n_episodios episódios.

    CORREÇÃO: versão original retornava apenas o melhor episódio bem-sucedido,
    distorcendo as métricas do arquivo de saída. Agora retorna as médias reais
    sobre todos os episódios, com o melhor resultado como referência secundária.

    Returns:
        dict com métricas: médias de n_passos, recompensa_total, velocidade;
        taxa_sucesso; e campos _melhor_* com o episódio de melhor desempenho.
    """
    resultados = []
    for _ in range(n_episodios):
        obs = env.reset()
        reward_total = 0.0
        velocidades = []
        n_passos = 0
        sucesso = False

        done = False
        while not done:
            action = agente.escolher_acao(obs, eps_override=0.0)  # gulosa
            obs, r, term, trunc, info = env.step(action)
            reward_total += r
            v_norm = float(obs[5])  # último componente é v_norm
            velocidades.append(v_norm * 2.0)  # desnormaliza (V_MAX=2.0)
            n_passos += 1
            done = term or trunc
            if done:
                sucesso = info.get("chegada", False)

        resultados.append({
            "n_passos": n_passos,
            "recompensa_total": reward_total,
            "sucesso": sucesso,
            "velocidade_media": float(np.mean(velocidades)) if velocidades else 0.0,
            "velocidade_max": float(np.max(velocidades)) if velocidades else 0.0,
        })

    taxa_sucesso = sum(r["sucesso"] for r in resultados) / len(resultados)

    # Métricas médias — representam o desempenho real do agente
    n_passos_medio = float(np.mean([r["n_passos"] for r in resultados]))
    reward_medio   = float(np.mean([r["recompensa_total"] for r in resultados]))
    vel_media      = float(np.mean([r["velocidade_media"] for r in resultados]))
    vel_max        = float(np.max([r["velocidade_max"] for r in resultados]))

    # Melhor episódio bem-sucedido (ou o de maior reward se nenhum chegou)
    sucessos = [r for r in resultados if r["sucesso"]]
    melhor = max(sucessos, key=lambda r: r["recompensa_total"]) if sucessos else \
             max(resultados, key=lambda r: r["recompensa_total"])

    return {
        # Médias reais (usadas no arquivo de saída)
        "n_passos": round(n_passos_medio),
        "recompensa_total": reward_medio,
        "sucesso": taxa_sucesso >= 0.5,   # considera sucesso se >= 50% chegaram
        "velocidade_media": vel_media,
        "velocidade_max": vel_max,
        "taxa_sucesso": taxa_sucesso,
        "n_episodios_avaliados": n_episodios,
        # Referência do melhor episódio individual
        "_melhor_n_passos": melhor["n_passos"],
        "_melhor_recompensa": melhor["recompensa_total"],
        "_melhor_sucesso": melhor["sucesso"],
    }


# ============================================================================
# SALVAR MODELO (ver enunciado/anexo_b_pickle.md)
# ============================================================================

def salvar_modelo(arquivo: Path, modelo: dict) -> None:
    """Serializa o modelo para disco via pickle."""
    with open(arquivo, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo salvo em {arquivo}")


def carregar_modelo(arquivo: Path) -> dict:
    """Desserializa o modelo do disco via pickle."""
    print(f"Carregando modelo de {arquivo} ...")
    with open(arquivo, "rb") as f:
        return pickle.load(f)


def montar_modelo(
    agente: AgenteQLearning,
    n_total_acumulado: int,
    rewards_acumulados: list,
    sucessos_acumulados: list,
    rewards_por_pista_acumulados: dict,
    config: dict,
) -> dict:
    """
    Monta o dicionário padrão para salvar em pickle.
    Inclui todos os metadados exigidos pelo README §2.7.
    """
    return {
        "q_table": dict(agente.Q),
        "q_table_n_actions": agente.n_actions,
        "discretization_K": agente.K,
        "n_episodes_trained": n_total_acumulado,
        "rewards_history": rewards_acumulados,
        "sucessos_history": sucessos_acumulados,       # CORREÇÃO: salva histórico real
        "rewards_por_pista": rewards_por_pista_acumulados,
        "config": config,
        "seed": SEED,
        "tracks_used": PISTAS_TREINO,        # NÃO inclui 17 e 18 (holdout)
        "n_estados_populados": agente.n_estados_populados,
        # Guarda o ε atual para que --continuar retome o schedule de onde parou
        "eps_atual": agente.eps,
    }


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA (formato README §4.3)
# ============================================================================

def escrever_saida(
    caminho: str,
    pista: str,
    resultado: dict,
    n_episodios_treinados: int,
    n_estados_populados: int,
) -> None:
    """
    Escreve arquivo de resultado no formato esperado pelo README §4.3.

    Exemplo de saída:
        === Pista: pista_17.txt ===
        Algoritmo: Q-Learning (round-robin em pistas 01-16)
        Episódios totais de treinamento: 480000
        Estados populados: 3412
        Tempo de chegada (passos): 87
        Velocidade média: 1.23
        Velocidade máxima atingida: 2.00
        Recompensa total: 483.52
        Sucesso: SIM
    """
    nome_pista = Path(pista).name
    sucesso_str = "SIM" if resultado["sucesso"] else "NAO"

    linhas = [
        f"=== Pista: {nome_pista} ===",
        f"Algoritmo: Q-Learning (round-robin em pistas 01-16)",
        f"Episódios totais de treinamento: {n_episodios_treinados}",
        f"Estados populados: {n_estados_populados}",
        f"Tempo de chegada (passos): {resultado['n_passos']}",
        f"Velocidade média: {resultado['velocidade_media']:.2f}",
        f"Velocidade máxima atingida: {resultado['velocidade_max']:.2f}",
        f"Recompensa total: {resultado['recompensa_total']:.2f}",
        f"Sucesso: {sucesso_str}",
        f"Taxa de sucesso ({resultado['n_episodios_avaliados']} episódios): "
        f"{resultado['taxa_sucesso']*100:.1f}%",
    ]
    conteudo = "\n".join(linhas) + "\n"

    Path(caminho).write_text(conteudo, encoding="utf-8")
    print(f"  → {caminho}")
    print(conteudo)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EP Carro Autônomo — Q-Learning Tabular",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python solucao.py                              # treina do zero (480k eps) + avalia holdout
  python solucao.py --recarregar                 # força re-treino mesmo com pkl existente
  python solucao.py --continuar                  # +30k eps/pista sobre o pkl existente
  python solucao.py --continuar --episodios-por-pista 10000  # sessão curta de continuação
  python solucao.py --avaliar pistas/pista_01.txt            # só avalia numa pista
        """,
    )
    parser.add_argument(
        "--episodios-por-pista", type=int, default=30_000,
        help="Episódios de treino por pista (default: 30000; total=480k para 16 pistas)",
    )
    parser.add_argument("--max-passos", type=int, default=500,
                        help="Limite de passos por episódio (default: 500)")
    parser.add_argument(
        "--K", type=int, default=5,
        help="Baldes da discretização (default: 5; ver README §3.2)",
    )
    parser.add_argument(
        "--recarregar", action="store_true",
        help="Força re-treino do zero, ignorando qualquer pkl existente",
    )
    parser.add_argument(
        "--continuar", action="store_true",
        help=(
            "Carrega pkl existente e continua treinando por --episodios-por-pista "
            "episódios adicionais por pista. O schedule de ε continua de onde parou."
        ),
    )
    parser.add_argument(
        "--avaliar", type=str, default=None,
        help="Apenas avalia o modelo salvo na pista especificada (pula treino)",
    )
    parser.add_argument(
        "--n-avaliacao", type=int, default=10,
        help="Episódios de avaliação gulosa por pista (default: 10)",
    )
    args = parser.parse_args()

    arquivo_pkl = DIR_TREINAMENTO / "qlearning.pkl"

    # ── Validação de flags mutuamente exclusivas ────────────────────────
    if args.recarregar and args.continuar:
        print("ERRO: --recarregar e --continuar são mutuamente exclusivos.")
        sys.exit(1)
    if args.avaliar and (args.recarregar or args.continuar):
        print("ERRO: --avaliar não pode ser combinado com --recarregar ou --continuar.")
        sys.exit(1)

    # ── Modo somente avaliação ──────────────────────────────────────────
    if args.avaliar is not None:
        if not arquivo_pkl.exists():
            print(f"ERRO: {arquivo_pkl} não encontrado. Rode sem --avaliar primeiro.")
            sys.exit(1)
        modelo = carregar_modelo(arquivo_pkl)
        agente_avaliacao = AgenteQLearning.from_modelo(modelo)
        agente_avaliacao.eps = 0.0  # garante modo guloso
        print(
            f"\nAgente carregado: K={agente_avaliacao.K}, "
            f"estados populados={agente_avaliacao.n_estados_populados}, "
            f"eps_treino={modelo.get('eps_atual', '?'):.3f}"
        )
        pistas_avaliar = [args.avaliar]

    # ── Modo treinamento (do zero, continuação, ou carga) ───────────────
    else:
        n_episodios_novos = args.episodios_por_pista * len(PISTAS_TREINO)

        if args.continuar:
            # ── --continuar: carrega estado anterior e retoma ──────────
            if not arquivo_pkl.exists():
                print(
                    f"AVISO: {arquivo_pkl} não encontrado. "
                    "Iniciando treinamento do zero (--continuar ignorado)."
                )
                modelo_anterior = None
            else:
                modelo_anterior = carregar_modelo(arquivo_pkl)

        elif args.recarregar:
            # ── --recarregar: ignora pkl existente ────────────────────
            print("Flag --recarregar: treinamento do zero.")
            modelo_anterior = None

        else:
            # ── Modo padrão: carrega se existir, treina se não existir ─
            if arquivo_pkl.exists():
                print(
                    f"\nPickle encontrado em {arquivo_pkl}.\n"
                    "Use --recarregar para treinar do zero ou "
                    "--continuar para adicionar mais episódios.\n"
                    "Carregando modelo para avaliação...\n"
                )
                modelo = carregar_modelo(arquivo_pkl)
                agente_avaliacao = AgenteQLearning.from_modelo(modelo)
                agente_avaliacao.eps = 0.0
                print(
                    f"Agente carregado: K={agente_avaliacao.K}, "
                    f"estados populados={agente_avaliacao.n_estados_populados}"
                )
                pistas_avaliar = PISTAS_HOLDOUT
                # Pula direto para a avaliação
                _avaliar_e_escrever(pistas_avaliar, agente_avaliacao, modelo, args)
                print("\nPronto.")
                return
            else:
                modelo_anterior = None

        # ── Reconstrói agente (continuação ou do zero) ─────────────────
        if modelo_anterior is not None:
            # Continuação: usa os hiperparâmetros salvos, restaura Q-table
            agente = AgenteQLearning.from_modelo(modelo_anterior)
            ep_offset = modelo_anterior["n_episodes_trained"]
            rewards_acumulados = list(modelo_anterior.get("rewards_history", []))
            sucessos_acumulados = list(modelo_anterior.get("sucessos_history", [False] * len(rewards_acumulados)))
            rewards_por_pista_acumulados = dict(modelo_anterior.get("rewards_por_pista", {}))
            # Garante que todas as pistas de treino estão no dict acumulado
            for p in PISTAS_TREINO:
                if p not in rewards_por_pista_acumulados:
                    rewards_por_pista_acumulados[p] = []

            config = modelo_anterior.get("config", {})
            # Respeita o --K da linha de comando apenas se diferente do salvo
            if args.K != agente.K:
                print(
                    f"AVISO: K da linha de comando ({args.K}) difere do modelo salvo "
                    f"({agente.K}). Usando K={agente.K} para manter compatibilidade."
                )
            eps_decai_em = config.get("eps_decai_em", int(0.8 * (ep_offset + n_episodios_novos)))
            # Captura o ε atual para que treinar_round_robin não reinicie do topo
            eps_inicio_continuacao = agente.eps
            print(
                f"\nContinuando treinamento a partir do episódio {ep_offset}.\n"
                f"ε atual={agente.eps:.4f}  |  estados populados={agente.n_estados_populados}\n"
                f"Novos episódios nesta sessão: {n_episodios_novos}"
            )
        else:
            # Do zero: cria agente fresco
            agente = AgenteQLearning(
                obs_dim=6,
                n_actions=5,
                K=args.K,
                alpha=0.1,
                gamma=0.99,
                eps_inicial=1.0,
                eps_final=0.05,
            )
            ep_offset = 0
            rewards_acumulados = []
            sucessos_acumulados = []
            rewards_por_pista_acumulados = {p: [] for p in PISTAS_TREINO}

            # O decaimento cobre 80% do orçamento TOTAL planejado para esta sessão.
            # Se o usuário continuar depois, o offset garante posicionamento correto.
            eps_decai_em = int(0.8 * n_episodios_novos)
            eps_inicio_continuacao = None  # treino do zero: schedule parte de 1.0
            config = {
                "alpha": agente.alpha,
                "gamma": agente.gamma,
                "eps_inicial": 1.0,
                "eps_final": agente.eps_final,
                "eps_decai_em": eps_decai_em,
                "max_passos": args.max_passos,
            }
            print(f"\nTreinamento do zero. Orçamento desta sessão: {n_episodios_novos} episódios.")

        # ── Roda o treinamento ─────────────────────────────────────────
        rewards_novos, sucessos_novos, rewards_pista_novos = treinar_round_robin(
            PISTAS_TREINO,
            agente,
            n_episodios_novos=n_episodios_novos,
            max_passos=args.max_passos,
            ep_offset=ep_offset,
            eps_decai_em=eps_decai_em,
            eps_inicio_continuacao=eps_inicio_continuacao,
            verbose=True,
        )

        # ── Acumula históricos ─────────────────────────────────────────
        rewards_acumulados.extend(rewards_novos)
        sucessos_acumulados.extend(sucessos_novos)
        for p in PISTAS_TREINO:
            rewards_por_pista_acumulados[p].extend(rewards_pista_novos.get(p, []))

        n_total_acumulado = ep_offset + n_episodios_novos

        # Atualiza config com os valores desta sessão
        config.update({
            "max_passos": args.max_passos,
            "eps_decai_em": eps_decai_em,
        })

        # ── Salva modelo ───────────────────────────────────────────────
        modelo = montar_modelo(
            agente,
            n_total_acumulado,
            rewards_acumulados,
            sucessos_acumulados,
            rewards_por_pista_acumulados,
            config,
        )
        salvar_modelo(arquivo_pkl, modelo)

        # ── Reconstrói agente para avaliação (ε=0) ────────────────────
        agente_avaliacao = AgenteQLearning.from_modelo(modelo)
        agente_avaliacao.eps = 0.0
        pistas_avaliar = PISTAS_HOLDOUT

    # ── Avaliação holdout (ou pista especificada com --avaliar) ─────────
    _avaliar_e_escrever(pistas_avaliar, agente_avaliacao, modelo, args)
    print("\nPronto.")


def _avaliar_e_escrever(
    pistas_avaliar: list[str],
    agente: AgenteQLearning,
    modelo: dict,
    args: argparse.Namespace,
) -> None:
    """
    Avalia o agente nas pistas listadas e escreve os arquivos de saída.
    Fatorado para evitar duplicação entre o branch de treino e o de --avaliar.
    """
    print(
        f"\nAvaliando em {len(pistas_avaliar)} pista(s) com ε=0 "
        f"({args.n_avaliacao} episódios cada)..."
    )
    for pista in pistas_avaliar:
        if not Path(pista).exists():
            print(f"  AVISO: {pista} não encontrada, pulando.")
            continue

        env = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        print(f"\n--- {pista} ---")
        resultado = avaliar(env, agente, n_episodios=args.n_avaliacao)

        nome_pista = Path(pista).stem               # "pista_17"
        arquivo_saida = f"q_learning_{nome_pista}.txt"

        escrever_saida(
            arquivo_saida,
            pista,
            resultado,
            n_episodios_treinados=modelo["n_episodes_trained"],
            n_estados_populados=modelo.get(
                "n_estados_populados", agente.n_estados_populados
            ),
        )


if __name__ == "__main__":
    main()