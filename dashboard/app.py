import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

AZUL = "#2a78d6"
CINZA = "#898781"
COR_POR_TIPO = {"bogo": "#2a78d6", "discount": "#1baf7a", "informational": "#eda100"}
ROTULO_TIPO = {"bogo": "BOGO", "discount": "Desconto", "informational": "Informacional"}

st.set_page_config(page_title="iFood | Otimização de Ofertas", page_icon="🎯", layout="wide")


def layout(titulo, altura=380):
    return {
        "title": {"text": titulo, "x": 0.0},
        "height": altura,
        "paper_bgcolor": "#fcfcfb",
        "plot_bgcolor": "#fcfcfb",
        "font": {"family": 'system-ui, "Segoe UI", sans-serif', "size": 12},
        "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
        "xaxis": {"gridcolor": "#e1e0d9"},
        "yaxis": {"gridcolor": "#e1e0d9"},
        "legend": {"orientation": "h", "y": 1.0, "x": 0.0, "yanchor": "bottom"},
    }


def percentual(valor, decimais=1):
    return f"{valor * 100:.{decimais}f}%".replace(".", ",")


def milhar(valor):
    return f"{valor:,.0f}".replace(",", ".")


def barras(rotulos, valores, cores, titulo, textos, altura=360, horizontal=False):
    eixo_dados = {"y": rotulos, "x": valores, "orientation": "h"} if horizontal else {"x": rotulos, "y": valores}
    figura = go.Figure(go.Bar(
        **eixo_dados, marker_color=cores, text=textos, textposition="outside", cliponaxis=False,
    ))
    figura.update_layout(**layout(titulo, altura), showlegend=False, bargap=0.45)
    figura.update_xaxes(visible=not horizontal) if horizontal else figura.update_yaxes(visible=False)
    return figura


@st.cache_data
def carregar():
    return {
        "dados": pd.read_parquet(PROCESSED / "dataset_ofertas.parquet"),
        "curvas": pd.read_parquet(PROCESSED / "curvas_modelo.parquet"),
        "lift": pd.read_parquet(PROCESSED / "lift_por_decil.parquet"),
        "importancias": pd.read_parquet(PROCESSED / "importancia_features.parquet"),
        "scores": pd.read_parquet(PROCESSED / "scores_ofertas.parquet"),
        "recomendacoes": pd.read_parquet(PROCESSED / "recomendacoes.parquet"),
        "metricas": json.loads((PROCESSED / "metricas_modelos.json").read_text(encoding="utf-8")),
    }


if not (PROCESSED / "metricas_modelos.json").exists():
    st.error("Arquivos não encontrados em data/processed. Execute os notebooks 1 e 2 antes de abrir o dashboard.")
    st.stop()

base = carregar()
dados, metricas = base["dados"], base["metricas"]

st.sidebar.title("🎯 iFood | Ofertas")
pagina = st.sidebar.radio("Navegação", ["Visão geral", "Performance do modelo", "Simulador de estratégia", "Recomendações"])
st.sidebar.caption(
    "Teste com 17 mil clientes e 306 mil eventos. O modelo escora as 10 ofertas do portfólio "
    "para cada cliente e a política envia a melhor."
)


def pagina_visao_geral():
    st.title("Visão geral do teste")
    st.caption("Resultados observados no experimento (envio aleatório). Conversão = oferta vista e completada na janela.")

    kpis = st.columns(5)
    kpis[0].metric("Clientes no teste", milhar(dados["id_cliente"].nunique()))
    kpis[1].metric("Ofertas enviadas", milhar(len(dados)))
    kpis[2].metric("Taxa de conversão", percentual(dados["converteu"].mean()))
    kpis[3].metric("Taxa de visualização", percentual(dados["visualizou"].mean()))
    kpis[4].metric("Conversões acidentais", percentual(dados["conversao_acidental"].mean()))

    tipo = dados.groupby("tipo_oferta")["converteu"].mean().sort_values(ascending=False)
    canais = {"E-mail": "canal_email", "Mobile": "canal_mobile", "Web": "canal_web", "Social": "canal_social"}
    canal = pd.Series({nome: dados.loc[dados[coluna] == 1, "converteu"].mean() for nome, coluna in canais.items()}).sort_values(ascending=False)

    esquerda, direita = st.columns(2)
    esquerda.plotly_chart(barras(
        [ROTULO_TIPO[t] for t in tipo.index], tipo.values, [COR_POR_TIPO[t] for t in tipo.index],
        "Taxa de conversão por tipo de oferta", [percentual(v) for v in tipo.values],
    ), use_container_width=True)
    direita.plotly_chart(barras(
        canal.index, canal.values, AZUL,
        "Taxa de conversão dos envios que incluem cada canal", [percentual(v) for v in canal.values],
    ), use_container_width=True)

    funil_valores = [len(dados), int(dados["visualizou"].sum()), int(dados["converteu"].sum())]
    onda = dados.groupby("dia_recebimento")["converteu"].mean()

    esquerda, direita = st.columns(2)
    esquerda.plotly_chart(barras(
        ["Convertidas", "Vistas", "Recebidas"], funil_valores[::-1], ["#1c5cab", "#5598e7", "#86b6ef"],
        "Funil geral da oferta", [milhar(v) for v in funil_valores[::-1]], horizontal=True,
    ), use_container_width=True)
    direita.plotly_chart(barras(
        [f"dia {int(d)}" for d in onda.index], onda.values, AZUL,
        "Taxa de conversão por onda de envio", [percentual(v) for v in onda.values],
    ), use_container_width=True)

    oferta = (
        dados.groupby(["tipo_oferta", "valor_minimo", "valor_desconto", "duracao_dias"])["converteu"]
        .mean().reset_index().sort_values("converteu")
    )
    oferta["rotulo"] = oferta.apply(
        lambda linha: f"{ROTULO_TIPO[linha['tipo_oferta']]} · mín R$ {linha['valor_minimo']} · "
        f"R$ {linha['valor_desconto']} off · {int(linha['duracao_dias'])}d", axis=1,
    )
    st.plotly_chart(barras(
        oferta["rotulo"], oferta["converteu"], [COR_POR_TIPO[t] for t in oferta["tipo_oferta"]],
        "Taxa de conversão por oferta do portfólio", [percentual(v) for v in oferta["converteu"]],
        altura=420, horizontal=True,
    ), use_container_width=True)


def pagina_performance():
    st.title("Performance do modelo")
    st.caption("LightGBM de propensão por (cliente, oferta), avaliado em clientes totalmente fora do treino.")

    lgbm = metricas["modelos"]["lightgbm"]
    baseline = metricas["modelos"]["baseline taxa por tipo"]
    kpis = st.columns(4)
    kpis[0].metric("ROC-AUC", f"{lgbm['roc_auc']:.3f}", delta=f"+{lgbm['roc_auc'] - baseline['roc_auc']:.3f} vs baseline")
    kpis[1].metric("PR-AUC", f"{lgbm['pr_auc']:.3f}", delta=f"+{lgbm['pr_auc'] - baseline['pr_auc']:.3f} vs baseline")
    kpis[2].metric("Log loss", f"{lgbm['log_loss']:.3f}")
    kpis[3].metric("Brier score", f"{lgbm['brier']:.3f}")

    tabela = pd.DataFrame(metricas["modelos"]).T
    tabela.columns = ["ROC-AUC", "PR-AUC", "Log loss", "Brier"]
    st.dataframe(tabela.style.format("{:.4f}"), use_container_width=True)

    taxa_base = metricas["geral"]["taxa_conversao_teste"]
    curvas = base["curvas"]
    graficos = st.columns(2)
    for coluna, (nome, titulo, referencia, eixos) in zip(
        graficos,
        [
            ("roc", f"Curva ROC (AUC {lgbm['roc_auc']:.3f})", ([0, 1], [0, 1]), ("taxa de falsos positivos", "taxa de verdadeiros positivos")),
            ("precision_recall", f"Curva Precision-Recall (AP {lgbm['pr_auc']:.3f})", ([0, 1], [taxa_base, taxa_base]), ("recall", "precisão")),
        ],
    ):
        recorte = curvas[curvas["curva"] == nome]
        figura = go.Figure([
            go.Scatter(x=referencia[0], y=referencia[1], mode="lines", line={"color": CINZA, "width": 1}, name="Aleatório"),
            go.Scatter(x=recorte["x"], y=recorte["y"], mode="lines", line={"color": AZUL, "width": 2}, name="LightGBM"),
        ])
        figura.update_layout(**layout(titulo))
        figura.update_xaxes(title=eixos[0])
        figura.update_yaxes(title=eixos[1])
        coluna.plotly_chart(figura, use_container_width=True)

    calibracao = curvas[curvas["curva"] == "calibracao"]
    figura_cal = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": CINZA, "width": 1}, name="Calibração perfeita"),
        go.Scatter(x=calibracao["x"], y=calibracao["y"], mode="lines+markers", line={"color": AZUL, "width": 2}, name="LightGBM"),
    ])
    figura_cal.update_layout(**layout("Calibração: o score é probabilidade real"))
    figura_cal.update_xaxes(title="score previsto", range=[0, 1])
    figura_cal.update_yaxes(title="taxa observada", range=[0, 1])

    lift = base["lift"]
    figura_lift = barras(
        [str(int(d)) for d in lift["decil"]], lift["taxa_conversao"], AZUL,
        "Conversão por decil de score (1 = maiores scores)", [percentual(v, 0) for v in lift["taxa_conversao"]],
    )
    figura_lift.add_hline(y=taxa_base, line_color=CINZA, line_width=1)

    esquerda, direita = st.columns(2)
    esquerda.plotly_chart(figura_cal, use_container_width=True)
    direita.plotly_chart(figura_lift, use_container_width=True)

    top15 = base["importancias"].head(15).iloc[::-1]
    st.plotly_chart(barras(
        top15["feature"], top15["importancia_relativa"], AZUL,
        "Principais features (ganho relativo do LightGBM)", [percentual(v) for v in top15["importancia_relativa"]],
        altura=480, horizontal=True,
    ), use_container_width=True)
    st.caption(
        "O comportamento transacional e o perfil do cliente dominam; os atributos da oferta "
        "(desconto, gasto mínimo, canais) são o que permite ranquear ofertas para o mesmo cliente."
    )


def pagina_simulador():
    st.title("Simulador de estratégia")
    st.caption(
        "Compara a política do modelo (melhor oferta por cliente) com o envio aleatório do teste. "
        "Conversões esperadas usam os scores calibrados; receita líquida = score x (gasto médio por conversão - desconto)."
    )

    simulacao = metricas["simulacao"]
    taxa_aleatoria = simulacao["taxa_conversao_politica_aleatoria"]
    receita_aleatoria = simulacao["receita_liquida_por_envio_aleatoria"]

    filtros = st.columns([1.2, 1.4, 1.4])
    objetivo = filtros[0].radio("Objetivo da política", ["Maximizar conversão", "Maximizar receita líquida"])
    tipos = filtros[1].multiselect("Tipos de oferta permitidos", list(ROTULO_TIPO.values()), default=list(ROTULO_TIPO.values()))
    fracao_alvo = filtros[2].slider("% da base a impactar", 5, 100, 100, step=5) / 100

    if not tipos:
        st.info("Selecione ao menos um tipo de oferta.")
        return

    criterio = "score" if objetivo == "Maximizar conversão" else "receita_esperada"
    permitidos = [t for t, r in ROTULO_TIPO.items() if r in tipos]
    melhores = (
        base["scores"][base["scores"]["tipo_oferta"].isin(permitidos)]
        .sort_values(["id_cliente", criterio, "eh_informacional"], ascending=[True, False, True])
        .groupby("id_cliente").head(1)
        .sort_values(criterio, ascending=False)
        .reset_index(drop=True)
    )
    impactados = melhores.head(max(1, int(len(melhores) * fracao_alvo)))
    n = len(impactados)

    kpis = st.columns(4)
    kpis[0].metric("Clientes impactados", milhar(n))
    kpis[1].metric(
        "Conversões esperadas", milhar(impactados["score"].sum()),
        delta=f"+{milhar(impactados['score'].sum() - taxa_aleatoria * n)} vs aleatório",
    )
    kpis[2].metric(
        "Taxa de conversão esperada", percentual(impactados["score"].mean()),
        delta=f"{(impactados['score'].mean() / taxa_aleatoria - 1) * 100:+.0f}% vs aleatório",
    )
    kpis[3].metric(
        "Receita líquida esperada", f"R$ {milhar(impactados['receita_esperada'].sum())}",
        delta=f"{(impactados['receita_esperada'].sum() / (receita_aleatoria * n) - 1) * 100:+.0f}% vs aleatório",
    )

    fracao = np.arange(1, len(melhores) + 1) / len(melhores)
    referencia = taxa_aleatoria if criterio == "score" else receita_aleatoria
    figura = go.Figure([
        go.Scatter(x=fracao, y=[referencia] * len(fracao), mode="lines", line={"color": CINZA, "width": 1.5}, name="Envio aleatório"),
        go.Scatter(x=fracao, y=melhores[criterio].expanding().mean(), mode="lines", line={"color": AZUL, "width": 2}, name="Política do modelo"),
    ])
    figura.add_vline(x=fracao_alvo, line_color=CINZA, line_width=1)
    figura.update_layout(**layout("Valor médio por envio ao impactar os top X% da base", 400))
    figura.update_xaxes(title="% da base impactada (ordenada pelo critério)", tickformat=".0%")
    figura.update_yaxes(tickformat=".0%" if criterio == "score" else ",.2f")

    mix = impactados["tipo_oferta"].value_counts()
    figura_mix = barras(
        [ROTULO_TIPO[t] for t in mix.index], mix.values, [COR_POR_TIPO[t] for t in mix.index],
        "Mix de ofertas no público impactado", [milhar(v) for v in mix.values], altura=400,
    )

    esquerda, direita = st.columns([1.5, 1])
    esquerda.plotly_chart(figura, use_container_width=True)
    direita.plotly_chart(figura_mix, use_container_width=True)
    st.caption(
        "Projeções baseadas em scores calibrados e no gasto médio por conversão do treino. "
        "A confirmação do incremento exige teste A/B da política."
    )


def pagina_recomendacoes():
    st.title("Recomendações por cliente")
    st.caption("Melhor oferta por cliente segundo o modelo, com a alternativa que maximiza receita líquida.")

    tabela = base["recomendacoes"].copy()
    tabela["tipo_recomendado"] = tabela["tipo_recomendado"].map(ROTULO_TIPO)
    tabela["tipo_maior_receita"] = tabela["tipo_maior_receita"].map(ROTULO_TIPO)

    filtros = st.columns([1.4, 1.2, 1.6, 1.2])
    tipos = filtros[0].multiselect("Tipo recomendado", sorted(tabela["tipo_recomendado"].unique()), default=sorted(tabela["tipo_recomendado"].unique()))
    generos = filtros[1].multiselect("Gênero", sorted(tabela["genero"].unique()), default=sorted(tabela["genero"].unique()))
    idades = filtros[2].slider("Faixa etária", 18, 101, (18, 101))
    incluir_incompletos = filtros[3].checkbox("Incluir cadastro incompleto", value=True)

    na_faixa = tabela["idade"].between(*idades) | (incluir_incompletos & tabela["idade"].isna())
    filtradas = tabela[
        tabela["tipo_recomendado"].isin(tipos) & tabela["genero"].isin(generos) & na_faixa
    ].sort_values("score_conversao", ascending=False)

    resumo = st.columns(3)
    resumo[0].metric("Clientes filtrados", milhar(len(filtradas)))
    resumo[1].metric("Score médio de conversão", percentual(filtradas["score_conversao"].mean()) if len(filtradas) else "0%")
    resumo[2].metric("Receita esperada média", f"R$ {filtradas['receita_esperada'].mean():.2f}".replace(".", ",") if len(filtradas) else "R$ 0")

    colunas = {
        "id_cliente": "Cliente", "genero": "Gênero", "idade": "Idade", "limite_cartao": "Limite do cartão",
        "tempo_cadastro_dias": "Dias de cadastro", "num_transacoes_antes": "Compras no teste",
        "taxa_conversao_previa": "Conversão prévia", "tipo_recomendado": "Tipo recomendado",
        "oferta_recomendada": "Oferta recomendada", "score_conversao": "Score de conversão",
        "tipo_maior_receita": "Tipo ótimo em receita", "receita_esperada": "Receita esperada (R$)",
    }
    st.dataframe(
        filtradas[list(colunas)].rename(columns=colunas),
        use_container_width=True, height=430, hide_index=True,
        column_config={
            "Score de conversão": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
            "Conversão prévia": st.column_config.NumberColumn(format="percent"),
            "Limite do cartão": st.column_config.NumberColumn(format="localized"),
            "Receita esperada (R$)": st.column_config.NumberColumn(format="localized"),
            "Idade": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.download_button(
        "Baixar recomendações filtradas (CSV)",
        data=filtradas[list(colunas)].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="recomendacoes_ifood.csv",
        mime="text/csv",
    )


PAGINAS = {
    "Visão geral": pagina_visao_geral,
    "Performance do modelo": pagina_performance,
    "Simulador de estratégia": pagina_simulador,
    "Recomendações": pagina_recomendacoes,
}
PAGINAS[pagina]()
