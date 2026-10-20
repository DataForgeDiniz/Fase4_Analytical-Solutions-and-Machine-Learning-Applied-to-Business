"""
Tech Challenge Fase 4 - Amazon Product Performance & Customer Insights
Etapa de Customer Insights (NLP) sobre os comentários dos consumidores.

Entrada : dados_brutos/amazon.csv (Kaggle - karkavelrajaj/amazon-sales-dataset)
Saídas  : dados_tratados/
    trechos_comentarios.csv  -> 1 linha por trecho de comentário (título ou conteúdo)
                                com sentimento (VADER) e classificação
    trechos_temas.csv        -> trecho x tema (um trecho pode citar vários temas), com produto e sentimento
    palavras_chave.csv       -> frequência de termos (unigramas/bigramas) por sentimento
    dim_tema.csv             -> dicionário de temas com descrição e palavras-gatilho

Observação metodológica:
    Na base original cada linha traz até 8 avaliações concatenadas por vírgula nos campos
    review_title e review_content. Como o próprio texto também contém vírgulas, não é possível
    separar as avaliações individuais com 100% de precisão (os títulos só alinham com o número
    de review_id em ~67% das linhas). Por isso a unidade de análise textual é o "trecho":
    segmentos de texto obtidos quebrando títulos e conteúdos por vírgula/ponto final.

Uso: py -I scripts/01_customer_insights_nlp.py   (a partir da pasta do projeto)
"""
from pathlib import Path
import re
from collections import Counter

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / "dados_brutos" / "amazon.csv"
SAIDA = BASE / "dados_tratados"
SAIDA.mkdir(exist_ok=True)

# Limiares padrão do VADER para classificação do compound score
LIMIAR_POS = 0.05
LIMIAR_NEG = -0.05

# Dicionário de temas: regex aplicada ao trecho em minúsculas
TEMAS = {
    "Qualidade e acabamento": (
        "Percepção de qualidade, material e construção",
        r"\bquality\b|\bbuild\b|\bsturdy\b|\bmaterial\b|\bpremium\b|\bsolid\b|\bflimsy\b|\bplastic\b|\bfinish",
    ),
    "Custo-benefício": (
        "Preço justo, valor pelo dinheiro",
        r"value for money|\bworth\b|\bprice\b|\bbudget\b|\baffordable\b|\bcheap\b|\bmoney\b|\bvfm\b|\bcostly\b|\bexpensive\b",
    ),
    "Defeito e durabilidade": (
        "Parou de funcionar, quebrou, veio com defeito",
        r"stopped working|not working|\bdefective\b|\bdamaged?\b|\bbroke|\bfaulty\b|\bdead\b|\bdied\b|\bwaste\b|"
        r"\bworst\b|within (a|one|two|\d+) (week|month|day)s?|\blasting\b|\bdurab|\blong lasting|\bfail",
    ),
    "Carregamento e energia": (
        "Velocidade de carga, carregador, potência",
        r"\bcharg|\bwatt|\b\d+w\b|\bpower\b|\bcurrent\b",
    ),
    "Bateria": (
        "Duração e autonomia da bateria",
        r"\bbattery\b|\bbackup\b|\bdrain",
    ),
    "Som e áudio": (
        "Qualidade sonora, graves, microfone",
        r"\bsound\b|\bbass\b|\baudio\b|\bvolume\b|\bmic\b|\bmicrophone\b|\bnoise\b|\bspeaker",
    ),
    "Tela e imagem": (
        "Display, resolução, brilho, imagem",
        r"\bdisplay\b|\bscreen\b|\bpicture\b|\bbrightness\b|\bresolution\b|\bimage\b|\bcolou?rs? (are|is)\b",
    ),
    "Conectividade": (
        "Bluetooth, Wi-Fi, pareamento, compatibilidade",
        r"\bbluetooth\b|\bwi-?fi\b|\bconnect|\bpair|\bcompatib|\bsignal\b|\bnetwork\b",
    ),
    "Desempenho": (
        "Velocidade, funcionamento, performance",
        r"\bperformance\b|\bspeed\b|\bfast\b|\bslow\b|\blag|\bworks? (well|fine|perfectly|great|good)\b|\bworking (fine|well|good)\b",
    ),
    "Facilidade de uso": (
        "Instalação, configuração, praticidade",
        r"\beasy\b|\binstall|\bsetup\b|\bset up\b|\buser friendly\b|\bconvenient\b|\bhandy\b|\bsimple\b",
    ),
    "Design e tamanho": (
        "Aparência, tamanho, comprimento, cor",
        r"\bdesign\b|\blooks?\b|\bsize\b|\bcompact\b|\blength\b|\bcolou?r\b|\bstylish\b|\bweight\b|\blight ?weight\b|\bfit\b",
    ),
    "Entrega e embalagem": (
        "Prazo de entrega, embalagem, estado de chegada",
        r"\bdelivery\b|\bdelivered\b|\bpackag|\bpacking\b|\bbox\b|\bshipping\b|\barrived\b",
    ),
    "Atendimento e garantia": (
        "Suporte, garantia, troca, devolução",
        r"\bservice\b|\bwarranty\b|\bguarantee\b|\bsupport\b|\breturn|\brefund|\breplace|\bcustomer care\b",
    ),
    "Aquecimento": (
        "Superaquecimento do produto",
        r"\bheat|\bhot\b|\btemperature\b|\bwarm\b",
    ),
}
TEMAS_RE = {nome: re.compile(rx) for nome, (_, rx) in TEMAS.items()}

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also get got one product amazon item use used using bought buy dont
don't it's im i'm ive i've thats that's even well much like really go us still will can't cant didn't doesnt doesn't
isn't wasn't 2 1 3 4 5 good nice ok okay great
""".split())
# "good/nice/great" são retirados apenas da nuvem de termos (dominariam o ranking sem agregar informação)


def limpar_texto(s: str) -> str:
    s = re.sub(r"https?://\S+", " ", str(s))          # links de imagens dentro das reviews
    s = s.replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def segmentar(texto: str) -> list[str]:
    """Quebra o bloco concatenado em trechos (vírgula seguida de não-espaço, ou fim de frase)."""
    texto = limpar_texto(texto)
    partes = re.split(r",(?=\S)|(?<=[.!?])\s+(?=[A-Z])|(?<=[a-z])(?=[A-Z][a-z])", texto)
    return [p.strip(" .,-") for p in partes if len(p.strip(" .,-")) >= 3]


def classificar(compound: float) -> str:
    if compound >= LIMIAR_POS:
        return "Positivo"
    if compound <= LIMIAR_NEG:
        return "Negativo"
    return "Neutro"


def main() -> None:
    df = pd.read_csv(ENTRADA, dtype=str)
    # mesma regra de deduplicação do Power Query: manter o registro com maior rating_count
    df["_rc"] = pd.to_numeric(df["rating_count"].str.replace(",", ""), errors="coerce").fillna(0)
    df = df.sort_values("_rc", ascending=False).drop_duplicates("product_id", keep="first")

    analisador = SentimentIntensityAnalyzer()
    linhas = []
    for _, r in df.iterrows():
        for origem, campo in (("Título", "review_title"), ("Comentário", "review_content")):
            for seq, trecho in enumerate(segmentar(r[campo]), start=1):
                comp = analisador.polarity_scores(trecho)["compound"]
                linhas.append({
                    "product_id": r["product_id"],
                    "origem": origem,
                    "sequencia": seq,
                    "trecho": trecho[:500],
                    "qtd_palavras": len(trecho.split()),
                    "sentimento_score": round(comp, 4),
                    "sentimento": classificar(comp),
                })
    trechos = pd.DataFrame(linhas)
    # remove repetições exatas do mesmo trecho no mesmo produto (título e comentário iguais)
    trechos = trechos.drop_duplicates(["product_id", "trecho"]).reset_index(drop=True)
    trechos.insert(0, "trecho_id", range(1, len(trechos) + 1))

    ponte = []
    for t in trechos.itertuples():
        txt = t.trecho.lower()
        for nome, rx in TEMAS_RE.items():
            if rx.search(txt):
                # denormalizado com produto e sentimento para relacionar direto com Produtos no modelo
                ponte.append({"trecho_id": t.trecho_id, "product_id": t.product_id, "tema": nome, "trecho": t.trecho,
                              "sentimento": t.sentimento, "sentimento_score": t.sentimento_score})
    ponte = pd.DataFrame(ponte)

    # frequência de termos por sentimento (unigramas + bigramas)
    contagem = {s: Counter() for s in ("Positivo", "Negativo", "Neutro")}
    for t in trechos.itertuples():
        toks = [w for w in re.findall(r"[a-z][a-z']+", t.trecho.lower()) if w not in STOPWORDS and len(w) > 2]
        contagem[t.sentimento].update(toks)
        contagem[t.sentimento].update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    palavras = []
    for sent, cnt in contagem.items():
        for termo, freq in cnt.most_common(150):
            palavras.append({"sentimento": sent, "termo": termo, "frequencia": freq,
                             "tipo": "Bigrama" if " " in termo else "Palavra"})
    palavras = pd.DataFrame(palavras)

    dim_tema = pd.DataFrame(
        [{"tema": n, "descricao": d, "palavras_gatilho": rx.replace("\\b", "")} for n, (d, rx) in TEMAS.items()]
    )

    trechos.to_csv(SAIDA / "trechos_comentarios.csv", index=False, encoding="utf-8")
    ponte.to_csv(SAIDA / "trechos_temas.csv", index=False, encoding="utf-8")
    palavras.to_csv(SAIDA / "palavras_chave.csv", index=False, encoding="utf-8")
    dim_tema.to_csv(SAIDA / "dim_tema.csv", index=False, encoding="utf-8")

    print(f"Produtos: {df.product_id.nunique()} | Trechos: {len(trechos)} | Trechos com tema: {ponte.trecho_id.nunique()}")
    print(trechos.sentimento.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
