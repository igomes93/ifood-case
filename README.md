# Case Técnico Data Science iFood: Otimização de Distribuição de Cupons

Solução para decidir qual oferta enviar para cada cliente, a partir de um teste com cerca de 17 mil clientes e 306 mil eventos.

## A solução

Um pipeline PySpark reconstrói a jornada de cada oferta enviada (recebida, vista e completada dentro da janela de validade) e gera um dataset no nível (cliente, oferta recebida). Um modelo LightGBM de propensão escora as 10 ofertas do portfólio para cada cliente, e a política de envio escolhe a melhor por conversão ou por receita líquida esperada. Um dashboard Streamlit apresenta os resultados, a performance do modelo e um simulador interativo de estratégia.

## Resultados

| Métrica | Valor |
|---|---|
| Conversão observada no teste (envio aleatório) | 38,3% |
| ROC-AUC / PR-AUC do modelo (clientes fora do treino) | 0,801 / 0,698 |
| Calibração (score médio previsto vs taxa observada) | 38,5% vs 38,3% |
| Conversão esperada da política do modelo | 62,5% (+63% relativo) |
| Receita líquida esperada por envio | de R$ 16,42 para R$ 35,37 (+115%) |

Os números da política são projeções com scores calibrados. A confirmação do incremento causal exige teste A/B (ver limitações no notebook 2).

## Estrutura

```
ifood-case/
├── data/
│   ├── raw/                      # três JSONs originais
│   └── processed/                # saídas dos notebooks
├── notebooks/
│   ├── 1_data_processing.ipynb   # PySpark: EDA, limpeza, jornada da oferta, features
│   └── 2_modeling.ipynb          # modelagem, avaliação e simulação de impacto
├── dashboard/
│   └── app.py                    # dashboard Streamlit
├── README.md
└── requirements.txt
```

## Decisões técnicas

- Grão do dataset é o evento de recebimento: o mesmo par (cliente, oferta) se repete, e cada envio tem janela e jornada próprias.
- Atribuição de último toque: cada visualização ou conclusão vai para o envio mais recente cuja janela a contém e conta uma única vez.
- Conversão acidental não é sucesso: 16% dos envios de BOGO e desconto tiveram o cupom creditado sem o cliente ver a oferta; recebem `converteu = 0`.
- Ofertas informacionais não têm evento de conclusão: convertem com visualização seguida de transação na janela (premissa documentada).
- Sem vazamento: features usam apenas eventos anteriores a cada envio, e o split é estratificado e separado por cliente (`StratifiedGroupKFold`, zero clientes em comum).
- LightGBM escolhido contra baselines e Regressão Logística por vencer em todas as métricas, com scores calibrados (essencial para a simulação).
- Escrita do dataset em Parquet único via pandas: mantém o processamento em Spark e dispensa o winutils exigido pela escrita nativa no Windows.

As premissas completas estão na última célula do notebook 1.



### 1. Notebooks (na ordem)

Execute `notebooks/1_data_processing.ipynb` e depois `notebooks/2_modeling.ipynb` no Jupyter ou VS Code, ou pela linha de comando:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/1_data_processing.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/2_modeling.ipynb
```

O notebook 1 cria a sessão Spark local automaticamente e grava as saídas em `data/processed/`. No Databricks Community Edition, importe os notebooks e ajuste apenas os caminhos para o DBFS.

### 2. Dashboard

```bash
streamlit run dashboard/app.py
```

Quatro páginas: Visão geral (KPIs e conversão por tipo, canal e onda), Performance do modelo (ROC, PR, calibração, lift e importâncias), Simulador de estratégia (% da base, tipos permitidos e objetivo, comparando a política do modelo com o envio aleatório) e Recomendações (tabela filtrável por segmento com download em CSV).
