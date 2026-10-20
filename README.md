# Tech Challenge Fase 4: Amazon Product Performance & Customer Insights

Solução analítica em Power BI que responde à pergunta: **quais produtos, categorias e estratégias comerciais merecem maior atenção, e quais oportunidades os dados revelam?** A análise segue a cadeia Preço → Desconto → Engajamento → Satisfação.

## Entregáveis

| Entregável | Arquivo |
|---|---|
| Dashboard Power BI | [AmazonInsights.pbix](AmazonInsights.pbix) |
| Apresentação do vídeo executivo (11 slides) | [Apresentacao_Executiva.pptx](Apresentacao_Executiva.pptx) |
| Roteiro do vídeo executivo (≤ 7 min) | [docs/Roteiro_Video_Executivo.md](docs/Roteiro_Video_Executivo.md) |
| Preparação dos comentários (sentimento e temas) | [scripts/01_customer_insights_nlp.py](scripts/01_customer_insights_nlp.py) |
| Análise exploratória de validação | [scripts/02_analise_exploratoria.py](scripts/02_analise_exploratoria.py) · [saída](docs/saida_analise_exploratoria.txt) |

## Estrutura do dashboard

Cada página tem poucos visuais, e cada título já diz qual decisão o gráfico apoia. As cores seguem uma regra única: vermelho = agir/revisar, azul = oportunidade, verde = saudável.

1. **Visão Executiva:** 5 indicadores-chave, concentração do engajamento por categoria, portfólio por quadrante Engajamento × Satisfação e % de produtos mal avaliados por faixa de desconto.
2. **Produtos e Categorias:** dispersão Engajamento × Satisfação por produto, matriz categoria → subcategoria e ranking dos produtos mais avaliados. A nota aparece em vermelho quando está abaixo de 4,0 e em verde quando é de 4,3 ou mais.
3. **Preços e Estratégia Comercial:** preço original × com desconto, desconto efetivo, engajamento e % de notas baixas por faixa de desconto, e satisfação por faixa de preço.
4. **Customer Insights:** sentimento dos comentários, temas citados com % positivo/negativo, problemas mais frequentes e a "voz do cliente".
5. **Recomendações:** plano de ação priorizado com indicadores de acompanhamento, lista de produtos para revisar e lista de joias escondidas para dar visibilidade.

**Filtros e interações:** segmentações por categoria, subcategoria, marca, faixa de preço, faixa de desconto, quadrante, tema e sentimento; filtro cruzado entre visuais; navegação entre páginas pelo cabeçalho (Ctrl + clique no Power BI Desktop).

**Moeda:** os valores estão em rúpias indianas (₹), a moeda original da base. Foram mantidos sem conversão para não introduzir uma suposição de câmbio. Percentuais, notas e conclusões não dependem da moeda.

## Preparação e tratamento dos dados

**Fonte:** [Kaggle: karkavelrajaj/amazon-sales-dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset), 1.465 linhas e 16 colunas, em `dados_brutos/amazon.csv`.

### Power Query (tabela `Produtos`, etapas visíveis no .pbix)

| Problema encontrado | Tratamento |
|---|---|
| Preços como texto (`₹1,099`) | Remoção do símbolo e do separador de milhar; conversão para número |
| Desconto como texto (`64%`) | Conversão para decimal (0,64) |
| `rating` com valor inválido `\|` (1 produto) | Convertido para nulo e excluído das médias de nota |
| `rating_count` vazio (2 produtos) | Considerado 0 avaliações |
| 114 linhas com `product_id` repetido (mesmo produto capturado em momentos diferentes) | Mantido o registro com maior `rating_count`; resultado: **1.351 produtos únicos** |
| Categoria hierárquica (`Electronics\|...\|In-Ear`) | Separada em Categoria (traduzida), Categoria Nível 2, Subcategoria e Hierarquia |
| Sem coluna de marca | Marca = primeira palavra do nome, com normalização de Amazon Basics e Solimo |
| Nomes de produto longos e repetidos | Rótulo curto e único: nome resumido + ID |
| Variáveis contínuas | Faixas de preço, desconto e nota, com colunas de ordenação |

**Colunas calculadas (DAX):**
- **Quadrante:** corte na mediana de avaliações (4.736) e na nota 4,0.
- **Nota Bayesiana:** nota ajustada pelo volume de avaliações, para não superestimar produtos com poucas avaliações.
- **% de comentários negativos por produto.**

### Comentários dos clientes: `scripts/01_customer_insights_nlp.py`

- Cada linha da base traz até 8 avaliações concatenadas por vírgula, e o próprio texto também contém vírgulas. Por isso a unidade de análise é o **trecho de comentário**: **33.635 trechos**.
- **Sentimento:** VADER, um léxico para textos curtos em inglês, com score de −1 a +1. Os limiares são os padrão: ≥ 0,05 positivo, ≤ −0,05 negativo, entre eles neutro.
- **Temas:** dicionário de 14 temas com palavras-gatilho. Um trecho pode citar mais de um tema.
- **Validação:** o sentimento dos comentários acompanha a nota do produto (correlação de 0,36).

## Modelo de dados

Esquema estrela:
- `Produtos` → `Trechos` e `Temas Trechos`.
- `Temas` → `Temas Trechos`.
- `Sentimentos` → `Trechos`, `Temas Trechos` e `Palavras`.
- `Recomendações` é uma tabela de apoio do plano de ação.
- `_Medidas` reúne as medidas DAX, organizadas em pastas.

| Pasta | Principais medidas |
|---|---|
| Portfólio | Qtd Produtos, Qtd Categorias, Qtd Subcategorias, Qtd Marcas |
| Satisfação | Nota Média, Nota Ponderada (por volume), Nota Bayesiana, % Produtos Bem Avaliados, % Produtos Abaixo de 4 |
| Engajamento | Total Avaliações, Mediana Avaliações, % do Engajamento, Concentração Top 10%, Correlação Engajamento × Nota |
| Preço e Desconto | Preço Original/com Desconto Médio, Economia Média, Desconto Médio, Desconto Efetivo Ponderado, % Produtos com Desconto 50%+, Correlação Desconto × Nota |
| Quadrantes | Qtd Campeões / Risco / Joias, % Engajamento em Risco / em Joias, cores de alerta |
| Customer Insights | % Comentários Positivos/Negativos, Net Sentiment Score, Menções por Tema, % Menções Negativas, Top termos |

## Principais descobertas

| Achado | Número |
|---|---|
| Os 10% produtos mais avaliados concentram | 61,6% das avaliações |
| Eletrônicos responde por | 59,7% do engajamento |
| Risco de reputação (alto engajamento, nota < 4,0) | 112 produtos, 11,7% das avaliações; 20 são fones In-Ear |
| Joias escondidas (nota ≥ 4,0, baixo engajamento) | 446 produtos com só 3,1% das avaliações |
| Correlação desconto × nota | −0,16 |
| Produtos com nota < 4,0: até 20% de desconto vs acima de 80% | 16,6% vs 42,0% |
| Produtos acima de ₹20 mil | Melhor nota (4,23), só 5,3% abaixo de 4 e o menor desconto |
| Tema mais valorizado | Custo-benefício (81% positivo) |
| Tema mais crítico | Defeito e durabilidade (46% negativo; 57% em Casa e Cozinha) |

## Limitações

- A base não tem vendas, datas, devoluções nem margem, então o engajamento (quantidade de avaliações) funciona como indicador de popularidade.
- 349 produtos compartilham a mesma contagem de avaliações com outras variantes do mesmo anúncio (cores/versões), o que gera dupla contagem no total.
- As avaliações concatenadas não se separam com precisão total, por isso a análise textual é feita por trecho.

## Como atualizar

1. `py -m pip install pandas vaderSentiment`
2. `py -I scripts/01_customer_insights_nlp.py`, que gera `dados_tratados/*.csv`.
3. Abra o `AmazonInsights.pbix` e clique em **Atualizar**. Se mover a pasta, ajuste o parâmetro **PastaProjeto** (Transformar dados → Gerenciar parâmetros).
