#!/usr/bin/env python3
"""Build static HTML EDA report for avito parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

ITEMS_PATH = ROOT / "items_with_attrs.parquet"
USERS_PATH = ROOT / "users_with_history.parquet"

# fallback if files live in repo root
if not ITEMS_PATH.exists():
    ITEMS_PATH = ROOT / "items_with_attrs.parquet"
if not USERS_PATH.exists():
    USERS_PATH = ROOT / "users_with_history.parquet"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_parquet(ITEMS_PATH)
    users = pd.read_parquet(USERS_PATH)
    if users["contact_date"].dtype == object:
        users["contact_date"] = pd.to_datetime(users["contact_date"])
    return items, users


def fig_to_html(fig: plt.Figure, div_id: str) -> str:
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img id="{div_id}" src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0;" />'


def plot_price(items: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    price = items["price"].dropna()
    price = price[price > 0]
    ax.hist(price, bins=60, color="#2a6fbb", edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Цена, ₽ (log scale)")
    ax.set_ylabel("Число показов в SERP")
    ax.set_title("Распределение цен объявлений")
    return fig


def plot_block(items: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = items["block"].value_counts()
    ax.barh(counts.index.astype(str), counts.values, color="#e07a2f")
    ax.set_xlabel("Количество строк")
    ax.set_title("Тип блока в выдаче (block)")
    return fig


def plot_brands(items: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    top = items["brand"].value_counts().head(top_n)
    ax.barh(top.index[::-1], top.values[::-1], color="#3d9a5b")
    ax.set_xlabel("Показов в SERP")
    ax.set_title(f"Топ-{top_n} марок")
    return fig


def plot_serp_size(items: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = items.groupby("serp_x").size()
    ax.hist(sizes, bins=40, color="#7b5ea7", edgecolor="white")
    ax.set_xlabel("Число объявлений в одной выдаче (serp_x)")
    ax.set_ylabel("Число выдач")
    ax.set_title("Размер SERP (кандидатов на сессию)")
    return fig


def plot_category(items: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    cat = items["query_infm_logical_category"].value_counts()
    ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Категория запроса")
    return fig


def plot_engagement(items: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(
        axes,
        ["clicks_daily", "contacts_daily"],
        ["Клики (daily)", "Контакты (daily)"],
    ):
        vals = items[col].dropna()
        vals = vals[vals >= 0]
        ax.hist(vals.clip(upper=vals.quantile(0.99)), bins=50, color="#c44e52")
        ax.set_title(title)
        ax.set_xlabel(col)
    fig.suptitle("Вовлечённость (обрезка 99 перцентиля)")
    fig.tight_layout()
    return fig


def plot_user_history(users: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    per_user = users.groupby("user_id").size()
    axes[0].hist(per_user, bins=30, color="#4c72b0", edgecolor="white")
    axes[0].set_xlabel("Событий contact на пользователя")
    axes[0].set_ylabel("Число пользователей")
    axes[0].set_title("Длина истории контактов")

    daily = users.groupby(users["contact_date"].dt.date).size()
    axes[1].plot(daily.index, daily.values, marker="o", markersize=3)
    axes[1].set_title("Контакты по дням")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_missing(items: pd.DataFrame, users: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, df, title in [
        (axes[0], items, "items: доля пропусков"),
        (axes[1], users, "users: доля пропусков"),
    ]:
        miss = df.isnull().mean().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if miss.empty:
            ax.text(0.5, 0.5, "Нет пропусков", ha="center", va="center")
        else:
            ax.barh(miss.index[::-1], miss.values[::-1], color="#888")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Доля NaN")
        ax.set_title(title)
    fig.tight_layout()
    return fig


def build_summary(items: pd.DataFrame, users: pd.DataFrame) -> dict:
    serp_sizes = items.groupby("serp_x").size()
    return {
        "items_rows": int(len(items)),
        "items_cols": int(len(items.columns)),
        "unique_serp": int(items["serp_x"].nunique()),
        "unique_item_id": int(items["item_id"].nunique()),
        "unique_sellers_in_items": int(items["user_id"].nunique()),
        "serp_size_mean": float(serp_sizes.mean()),
        "serp_size_max": int(serp_sizes.max()),
        "users_rows": int(len(users)),
        "unique_users_history": int(users["user_id"].nunique()),
        "contact_date_min": str(users["contact_date"].min().date()),
        "contact_date_max": str(users["contact_date"].max().date()),
        "overlap_user_id": int(
            len(set(users["user_id"]) & set(items["user_id"].dropna().astype(int)))
        ),
        "overlap_item_id": int(len(set(users["item_id"]) & set(items["item_id"]))),
        "blocks": items["block"].value_counts().to_dict(),
        "categories": items["query_infm_logical_category"].value_counts().to_dict(),
    }


def render_html(summary: dict, figures: list[tuple[str, str]]) -> str:
    fig_blocks = "\n".join(
        f"<section><h3>{title}</h3>{img}</section>" for title, img in figures
    )
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Avito Auto — EDA</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; }}
    h1 {{ color: #1a1a1a; }}
    pre {{ background: #f4f4f4; padding: 12px; overflow: auto; font-size: 13px; }}
    section {{ margin-bottom: 28px; border-bottom: 1px solid #eee; padding-bottom: 12px; }}
    .note {{ background: #fff8e6; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>EDA: Avito «Авто» (parquet)</h1>
  <div class="note">
    <p><b>items_with_attrs</b> — показы объявлений в поисковой выдаче (SERP): одна строка = одно объявление в конкретной выдаче <code>serp_x</code>.</p>
    <p><b>users_with_history</b> — история контактов пользователя с объявлениями (событие contact + атрибуты авто).</p>
    <p>Откройте также <code>notebooks/01_eda_avito_data.ipynb</code> для интерактивного просмотра таблиц.</p>
  </div>
  <h2>Сводка</h2>
  <pre>{summary_json}</pre>
  <h2>Графики</h2>
  {fig_blocks}
</body>
</html>
"""


def main() -> None:
    sns.set_theme(style="whitegrid")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    items, users = load_data()
    summary = build_summary(items, users)

    figures = [
        ("Распределение цен", fig_to_html(plot_price(items), "price")),
        ("Тип блока в выдаче", fig_to_html(plot_block(items), "block")),
        ("Топ марок", fig_to_html(plot_brands(items), "brands")),
        ("Размер SERP", fig_to_html(plot_serp_size(items), "serp")),
        ("Категория запроса", fig_to_html(plot_category(items), "cat")),
        ("Клики и контакты", fig_to_html(plot_engagement(items), "eng")),
        ("История пользователей", fig_to_html(plot_user_history(users), "users")),
        ("Пропуски", fig_to_html(plot_missing(items, users), "miss")),
    ]

    html = render_html(summary, figures)
    out = REPORT_DIR / "eda_report.html"
    out.write_text(html, encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out}")
    print(f"Summary: overlap item_id={summary['overlap_item_id']} (0 = разные срезы данных)")


if __name__ == "__main__":
    main()
