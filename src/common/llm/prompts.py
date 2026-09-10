"""Prompts from UR4Rec §3.1 (COLING 2025) — user preference & item knowledge."""

from __future__ import annotations


def build_user_preference_prompt_ml1m(
    user_id: int,
    history_titles: list[str],
    history_genres: list[str],
) -> str:
    """§3.1.1 — adapted from paper Steam example to MovieLens movies."""
    lines = []
    for title, genres in zip(history_titles, history_genres):
        lines.append(f'"{title}", a {genres} movie')
    hist = "; ".join(lines) if lines else "(empty history)"
    return (
        "Your task is to use some keywords to summarize the user's preference "
        "based on historical interactions. "
        f"Given a user (id {user_id}) whose history of watched movies over time "
        f"is listed below: [{hist}]. "
        "Please analyze the user's preferences on movies about factors like genre, "
        "director, actors, time period, country, character, mood/tone, and themes. "
        "Please give an analysis that incorporates specific elements from the user's "
        "history of movie interactions and other relevant factors."
    )


def build_user_preference_prompt_amazon(
    user_id: int,
    history_titles: list[str],
    history_cats: list[str],
) -> str:
    """§3.1.1 style — Amazon-Books (book title + category)."""
    lines = []
    for title, cat in zip(history_titles, history_cats):
        lines.append(f'"{title}", a {cat} book')
    hist = "; ".join(lines) if lines else "(empty history)"
    return (
        "Your task is to use some keywords to summarize the user's preference "
        "based on historical interactions. "
        f"Given a user (id {user_id}) whose history of read books over time "
        f"is listed below: [{hist}]. "
        "Please analyze the user's preferences on books about factors like genre, "
        "author, publisher, time period, country, themes, and writing style. "
        "Please give an analysis that incorporates specific elements from the user's "
        "history of book interactions and other relevant factors."
    )


def build_item_knowledge_prompt_amazon(title: str, category: str, brand: str = "") -> str:
    """§3.1.2 — book overview (title + category/brand)."""
    by = f' by {brand}' if brand and brand != "unknown" else ""
    return (
        f'Your task is to craft a detailed overview of the book "{title}"{by} '
        f"({category}), covering aspects such as genre, author, publisher, plot tone, "
        "mood, themes, time period, and audience appeal."
    )


def build_user_preference_prompt_steam(
    user_id: int,
    history_titles: list[str],
    history_genres: list[str],
) -> str:
    """§3.1.1 — paper Steam example wording."""
    lines = []
    for title, genres in zip(history_titles, history_genres):
        lines.append(f'"{title}", a {genres} game')
    hist = "; ".join(lines) if lines else "(empty history)"
    return (
        "Your task is to use some keywords to summarize the user's preference "
        "based on historical interactions. "
        f"Given a user (id {user_id}) whose history of playing games over time "
        f"is listed below: [{hist}]. "
        "Please analyze the user's preferences on games about factors like genre, "
        "development company, graphics quality, game plot, and storyline. "
        "Please give an analysis that incorporates specific elements from the user's "
        "history of game interactions and other relevant factors."
    )


def build_item_knowledge_prompt_steam(title: str, genres: str, developer: str = "") -> str:
    """§3.1.2 — paper Half-Life style."""
    by = f' by {developer}' if developer and developer != "unknown" else ""
    return (
        f'Your task is to craft a detailed overview of the game "{title}"{by}, '
        f"covering aspects such as genre, publish company, graphics quality, price, "
        f"game plot, mood, themes, and audience appeal. Game genres: {genres}."
    )


def build_item_knowledge_prompt_ml1m(title: str, genres: str) -> str:
    """§3.1.2 — item knowledge from title + category (genres)."""
    return (
        f'Your task is to craft a detailed overview of the movie "{title}" '
        f"({genres}), covering aspects such as genre, director, actors, plot tone, "
        "mood, themes, time period, and audience appeal."
    )


def build_user_preference_prompt_avito(
    session_id: int,
    query_category: str,
    query_loc: str,
    history_listings: list[str] | None = None,
) -> str:
    """§3.1.1 style for Avito Auto search sessions."""
    extra = ""
    if history_listings:
        extra = " Contact history listings: " + "; ".join(history_listings[:10]) + "."
    return (
        "Your task is to use some keywords to summarize the user's preference "
        "based on search context and interactions. "
        f"Given a car buyer search session (id {session_id}) for {query_category} "
        f"in region {query_loc}.{extra} "
        "Please analyze preferences on factors like brand, model, price range, "
        "mileage, fuel type, body type, year, and seller type. "
        "Give a structured analysis with numbered points."
    )


def build_item_knowledge_prompt_avito(
    title: str,
    brand: str,
    attrs: str,
    price: float | None = None,
) -> str:
    """§3.1.2 — title + brand as in paper."""
    price_s = f" Price: about {int(price)}." if price and price > 0 else ""
    return (
        f'Your task is to craft a detailed overview of the car listing "{title}" '
        f"by {brand}, covering aspects such as brand, model, mileage, fuel, gearbox, "
        f"body type, price, condition, and target buyer.{price_s} "
        f"Additional attributes: {attrs}."
    )


# Legacy templates (debug only — NOT paper reproduction)
def template_user_preference(user_id: int, history_titles: list[str], history_genres: list[str]) -> str:
    genres_flat = ", ".join(sorted({g for gs in history_genres for g in gs.split("|") if g}))
    titles = "; ".join(history_titles[-5:])
    return (
        f"User preference summary for user {user_id}. "
        f"Recent movies: {titles}. "
        f"Preferred genres: {genres_flat or 'unknown'}."
    )


def template_item_knowledge(title: str, genres: str) -> str:
    return f"Item knowledge for '{title}'. Genres: {genres}."
