"""
Análise exploratória de apoio: replica as regras de tratamento do Power Query para
validar os números do dashboard e levantar os insights do relatório executivo.

Uso: py -I scripts/02_analise_exploratoria.py > docs/saida_analise_exploratoria.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)


def num(s):
    return pd.to_numeric(s.str.replace("₹", "").str.replace(",", "").str.replace("%", ""), errors="coerce")


def carregar() -> pd.DataFrame:
    df = pd.read_csv(BASE / "dados_brutos" / "amazon.csv", dtype=str)
    df["preco_desconto"] = num(df.discounted_price)
    df["preco_original"] = num(df.actual_price)
    df["desconto_pct"] = num(df.discount_percentage) / 100
    df["nota"] = pd.to_numeric(df.rating, errors="coerce")
    df["qtd_avaliacoes"] = num(df.rating_count).fillna(0)
    df = df.sort_values("qtd_avaliacoes", ascending=False).drop_duplicates("product_id")
    cat = df.category.str.split("|")
    df["cat_n1"] = cat.str[0]
    df["cat_n2"] = cat.str[1]
    df["subcategoria"] = cat.str[-1]
    df["economia"] = df.preco_original - df.preco_desconto
    df["faixa_preco"] = pd.cut(df.preco_original, [0, 500, 1000, 5000, 20000, 1e9],
                               labels=["Até ₹500", "₹500 a ₹1 mil", "₹1 mil a ₹5 mil", "₹5 mil a ₹20 mil", "Acima de ₹20 mil"], right=False)
    df["faixa_desconto"] = pd.cut(df.desconto_pct, [-0.01, 0.2, 0.4, 0.6, 0.8, 1],
                                  labels=["0-20%", "21-40%", "41-60%", "61-80%", "81%+"])
    m = df.qtd_avaliacoes.median()
    C = df.nota.mean()
    df["nota_bayes"] = (df.qtd_avaliacoes / (df.qtd_avaliacoes + m)) * df.nota + (m / (df.qtd_avaliacoes + m)) * C
    alto_eng = df.qtd_avaliacoes >= m
    alta_nota = df.nota >= 4
    df["quadrante"] = np.select(
        [alto_eng & alta_nota, alto_eng & ~alta_nota, ~alto_eng & alta_nota],
        ["Campeões", "Risco de reputação", "Joias escondidas"], "Em observação")
    return df


def main():
    df = carregar()
    t = pd.read_csv(BASE / "dados_tratados" / "trechos_comentarios.csv")
    print("== KPIs gerais")
    print(f"produtos={len(df)} cat_n1={df.cat_n1.nunique()} subcat={df.subcategoria.nunique()}")
    print(f"nota média={df.nota.mean():.3f} mediana={df.nota.median()} ponderada={np.average(df.nota.fillna(C:=df.nota.mean()), weights=df.qtd_avaliacoes):.3f}")
    print(f"total avaliações={df.qtd_avaliacoes.sum():,.0f} mediana={df.qtd_avaliacoes.median():,.0f} média={df.qtd_avaliacoes.mean():,.0f}")
    print(f"desconto médio={df.desconto_pct.mean():.3f} mediano={df.desconto_pct.median():.3f} ponderado preço={1-df.preco_desconto.sum()/df.preco_original.sum():.3f}")
    print(f"% produtos desconto>=50%={(df.desconto_pct>=0.5).mean():.3f}  >=70%={(df.desconto_pct>=0.7).mean():.3f}")
    print(f"preço original médio={df.preco_original.mean():,.0f} c/ desc médio={df.preco_desconto.mean():,.0f} mediana orig={df.preco_original.median():,.0f}")
    print(f"% nota>=4={(df.nota>=4).mean():.3f} % nota<3.5={(df.nota<3.5).mean():.3f} min={df.nota.min()} max={df.nota.max()}")
    top10 = df.nlargest(int(len(df)*0.1), "qtd_avaliacoes").qtd_avaliacoes.sum()/df.qtd_avaliacoes.sum()
    print(f"concentração: top10% produtos = {top10:.3f} das avaliações")
    print(f"sentimento trechos: {t.sentimento.value_counts(normalize=True).round(3).to_dict()}")

    print("\n== Categoria nível 1")
    g = df.groupby("cat_n1").agg(produtos=("product_id", "count"), nota=("nota", "mean"), aval=("qtd_avaliacoes", "sum"),
                                 desc=("desconto_pct", "mean"), preco=("preco_original", "mean")).sort_values("aval", ascending=False)
    g["share_aval"] = g.aval / g.aval.sum()
    print(g.round(3))

    print("\n== Subcategorias (>=10 produtos)")
    s = df.groupby("subcategoria").agg(produtos=("product_id", "count"), nota=("nota", "mean"), aval=("qtd_avaliacoes", "sum"),
                                       desc=("desconto_pct", "mean"), preco=("preco_original", "median"))
    s = s[s.produtos >= 10].sort_values("aval", ascending=False)
    print(s.round(3).head(20))
    print("-- piores notas:"); print(s.sort_values("nota").head(8).round(3))
    print("-- melhores notas:"); print(s.sort_values("nota", ascending=False).head(8).round(3))
    print("-- maior desconto:"); print(s.sort_values("desc", ascending=False).head(8).round(3))

    print("\n== Desconto x nota / engajamento")
    print("corr desconto-nota", round(df.desconto_pct.corr(df.nota), 3), "| spearman", round(df.desconto_pct.rank().corr(df.nota.rank()), 3))
    print("corr desconto-log(aval)", round(df.desconto_pct.corr(np.log1p(df.qtd_avaliacoes)), 3))
    print("corr preço-nota", round(np.log(df.preco_original).corr(df.nota), 3), "| log preço-log aval", round(np.log(df.preco_original).corr(np.log1p(df.qtd_avaliacoes)), 3))
    print("corr log aval-nota", round(np.log1p(df.qtd_avaliacoes).corr(df.nota), 3))
    print(df.groupby("faixa_desconto", observed=True).agg(produtos=("product_id", "count"), nota=("nota", "mean"),
          aval_med=("qtd_avaliacoes", "median"), preco=("preco_original", "median"), pct_baixa=("nota", lambda x: (x < 4).mean())).round(3))
    print(df.groupby("faixa_preco", observed=True).agg(produtos=("product_id", "count"), nota=("nota", "mean"),
          aval_med=("qtd_avaliacoes", "median"), desc=("desconto_pct", "mean")).round(3))
    print(pd.crosstab(df.cat_n1, df.faixa_desconto).loc[["Electronics", "Computers&Accessories", "Home&Kitchen", "OfficeProducts"]])

    print("\n== Quadrantes (corte: mediana de avaliações e nota 4,0)")
    q = df.groupby("quadrante").agg(produtos=("product_id", "count"), nota=("nota", "mean"), aval=("qtd_avaliacoes", "sum"), desc=("desconto_pct", "mean"))
    q["share_aval"] = q.aval / q.aval.sum()
    print(q.round(3), "\nmediana aval:", df.qtd_avaliacoes.median())
    cols = ["product_name", "subcategoria", "nota", "qtd_avaliacoes", "desconto_pct", "preco_desconto"]
    print("-- Risco de reputação (mais avaliações):")
    print(df[df.quadrante == "Risco de reputação"].nlargest(10, "qtd_avaliacoes")[cols].to_string())
    print("-- Joias escondidas (maior nota):")
    print(df[df.quadrante == "Joias escondidas"].sort_values(["nota", "qtd_avaliacoes"], ascending=False).head(10)[cols].to_string())
    print("-- Top 10 engajamento:"); print(df.nlargest(10, "qtd_avaliacoes")[cols].to_string())
    print("-- Top 10 nota bayesiana:"); print(df.nlargest(10, "nota_bayes")[cols + ["nota_bayes"]].to_string())
    print("-- Desconto>=70% e nota<4:", ((df.desconto_pct >= .7) & (df.nota < 4)).sum(), "de", (df.desconto_pct >= .7).sum())

    print("\n== Sentimento por categoria / quadrante")
    tp = t.merge(df[["product_id", "cat_n1", "subcategoria", "quadrante", "nota", "faixa_desconto"]], on="product_id")
    tp["neg"] = tp.sentimento.eq("Negativo"); tp["pos"] = tp.sentimento.eq("Positivo")
    for k in ["cat_n1", "quadrante", "faixa_desconto"]:
        print(tp.groupby(k, observed=True).agg(trechos=("trecho_id", "count"), pos=("pos", "mean"), neg=("neg", "mean"), score=("sentimento_score", "mean")).round(3))
    sp = tp.groupby("product_id").agg(neg=("neg", "mean"), score=("sentimento_score", "mean")).join(df.set_index("product_id")[["nota"]])
    print("corr nota x %neg trechos", round(sp.nota.corr(sp.neg), 3), "| nota x score", round(sp.nota.corr(sp.score), 3))
    s2 = tp.groupby("subcategoria").agg(trechos=("trecho_id", "count"), neg=("neg", "mean")).query("trechos>=150").sort_values("neg", ascending=False)
    print(s2.head(10).round(3))
    p = pd.read_csv(BASE / "dados_tratados" / "trechos_temas.csv", usecols=["trecho_id", "tema"]).merge(tp, on="trecho_id")
    print("-- tema x cat_n1 (% negativo)")
    print(p.pivot_table(index="tema", columns="cat_n1", values="neg", aggfunc="mean").round(3)[["Electronics", "Computers&Accessories", "Home&Kitchen"]])
    print("-- temas em produtos com nota<4 vs >=4.3 (share de trechos)")
    p["grupo"] = np.where(p.nota < 4, "nota<4", np.where(p.nota >= 4.3, "nota>=4.3", "meio"))
    sh = p.groupby("grupo").tema.value_counts(normalize=True).unstack(0).round(3)
    print(sh.sort_values("nota<4", ascending=False))
    print("-- exemplos de trechos negativos sobre defeito:")
    print(p[(p.tema == "Defeito e durabilidade") & p.neg].trecho.sample(8, random_state=3).to_string())


if __name__ == "__main__":
    main()
