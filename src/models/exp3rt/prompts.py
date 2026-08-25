"""Qwen chat prompts for Exp3RT stages (ported from official Llama-3 templates)."""

from __future__ import annotations

from typing import Any


def _item_labels(dataset: str) -> tuple[str, str]:
    if dataset == "imdb":
        return "movie", "Movie"
    return "book", "Book"


def _process_user_rating(text: str, dataset: str) -> str:
    parts = text.split("[Average Rating]")
    if len(parts) <= 1:
        return text
    try:
        rating = float(parts[-1].strip())
        if dataset == "imdb":
            rating = max(0.0, rating - 1.0)
        return f"{parts[0]}[User Average Rating]\n{rating:.1f}"
    except ValueError:
        return text


def _process_item_rating(text: str, dataset: str) -> str:
    parts = text.split("[Average Rating]")
    if len(parts) <= 1:
        return text
    try:
        rating = float(parts[-1].strip())
        if dataset == "imdb":
            rating = max(0.0, rating - 1.0)
        return f"{parts[0]}[Item Average Rating]\n{rating:.1f}"
    except ValueError:
        return text


def rating_system_prompt(dataset: str) -> str:
    s_item, _ = _item_labels(dataset)
    if dataset == "imdb":
        rating_scale = (
            "Predicted User Rating: [Predict the user's rating as an integer from 0 to 9: "
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9. 0 indicates the user would strongly dislike the "
            f"{s_item}, while 9 indicates the user would highly enjoy and recommend it. "
            "Consider the average ratings provided for the user and the item in your prediction.]"
        )
    else:
        rating_scale = (
            "Predicted User Rating: [Predict the user's rating as an integer from 1 to 5: "
            "1, 2, 3, 4, 5. 1 indicates the user would strongly dislike the "
            f"{s_item}, while 5 indicates the user would highly enjoy and recommend it. "
            "Consider the average ratings provided for the user and the item in your prediction.]"
        )
    return (
        f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's "
        f"preferences and {s_item} characteristics provided, generate a recommendation reasoning "
        "and predict the user's rating.\n"
        "You must always generate a response in the following format whenever the user provides information:\n"
        "Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, "
        f"addressing at least three specific points of alignment or misalignment between the "
        f"user's preferences and the {s_item}'s characteristics.]\n"
        f"{rating_scale}\n"
        "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction "
        "based on the input provided."
    )


def rating_user_prompt(data_point: dict[str, Any], dataset: str) -> str:
    s_item, l_item = _item_labels(dataset)
    user_persona = _process_user_rating(str(data_point["user_persona"]), dataset)
    item_synopsis = _process_item_rating(str(data_point["item_synopsis"]), dataset)
    return (
        f"I need a recommendation for this {s_item}. Here's the information:\n"
        "User Preferences:\n"
        "<User Persona>\n"
        f"{user_persona}\n\n"
        f"{l_item} Characteristics:\n"
        f"<{l_item} Description>\n"
        f"{data_point['item_description']}\n\n"
        f"<{l_item} Synopsis>\n"
        f"{item_synopsis}\n\n"
        f"Based on this information, please provide a detailed reasoning for your recommendation "
        f"and predict a rating for this {s_item}. Follow the format specified in the system instructions."
    )


def rating_assistant_text(data_point: dict[str, Any], dataset: str) -> str:
    rating = data_point["score"]
    if dataset == "imdb":
        rating = int(rating) - 1
    return f"Reasoning: {data_point['rationale']}\nPredicted User Rating: {rating}"


def preference_system_prompt(dataset: str) -> str:
    s_item, _ = _item_labels(dataset)
    return (
        f"Given a review written by a user, list about the \"preference\" the user liked and "
        f"disliked about the {s_item}, under [Like] and [Dislike] in bullet points, respectively. "
        "If there is nothing to mention about like/dislike, simply write \"None.\" under the "
        "corresponding tag. DO NOT write any content that is not revealed in the review.\n"
        "### Output Format:\n"
        "[Like]\n"
        "- Encapsulate the \"preference\" user liked about the item in bullet points.\n"
        "[Dislike]\n"
        "- Encapsulate the \"preference\" user disliked about the item in bullet points."
    )


def preference_user_prompt(data_point: dict[str, Any]) -> str:
    return f"Here is the review written by the user:\n{data_point['input']}"


def preference_assistant_text(data_point: dict[str, Any]) -> str:
    return f"Preference: {data_point['output']}"


def profile_messages(data_point: dict[str, Any], kind: str, dataset: str) -> list[dict[str, str]]:
    s_item, _ = _item_labels(dataset)
    if kind == "user":
        noun = "movies" if dataset == "imdb" else "books"
        system = (
            f"These are the user's preferences about {noun}: {data_point['input']}\n"
            "Based on this preferences, point out the personality of the user under [Like] and [Dislike] "
            'in bullet point, respectively. If there is nothing to mention about like/dislike, '
            'simply write "None." under the corresponding tag.\n'
            "### Output Format:\n"
            "[Like]\n"
            "- Encapsulate the preferences of the user in bullet points.\n"
            "[Dislike]\n"
            "- Encapsulate the preferences of the user in bullet points."
        )
    else:
        system = (
            f"These are users' preferences about the {s_item}: {data_point['input']}\n"
            'Based on this preferences, point out the "preference" people liked and disliked about '
            'the item under [Like] and [Dislike] in bullet point, respectively. '
            'If there is nothing to mention about like/dislike, simply write "None." under the '
            "corresponding tag.\n"
            "### Output Format:\n"
            "[Like]\n"
            "- Encapsulate the preference people liked about the item in bullet points.\n"
            "[Dislike]\n"
            "- Encapsulate the preference people disliked about the item in bullet points."
        )
    return [{"role": "system", "content": system}]


def profile_assistant_text(data_point: dict[str, Any]) -> str:
    return f"Preference: {data_point['output']}"


def build_chat_messages(stage: str, data_point: dict[str, Any], dataset: str) -> list[dict[str, str]]:
    if stage == "rating":
        return [
            {"role": "system", "content": rating_system_prompt(dataset)},
            {"role": "user", "content": rating_user_prompt(data_point, dataset)},
        ]
    if stage == "preference":
        return [
            {"role": "system", "content": preference_system_prompt(dataset)},
            {"role": "user", "content": preference_user_prompt(data_point)},
        ]
    if stage in {"user", "item"}:
        return profile_messages(data_point, stage, dataset)
    raise ValueError(f"Unknown stage: {stage}")


def build_assistant_text(stage: str, data_point: dict[str, Any], dataset: str) -> str:
    if stage == "rating":
        return rating_assistant_text(data_point, dataset)
    if stage == "preference":
        return preference_assistant_text(data_point)
    if stage in {"user", "item"}:
        return profile_assistant_text(data_point)
    raise ValueError(f"Unknown stage: {stage}")


def format_chat(tokenizer, messages: list[dict[str, str]], assistant_text: str | None = None) -> tuple[str, str]:
    """Return (prompt_only, full_text) for label masking."""
    prompt_only = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_text is None:
        return prompt_only, prompt_only
    full_messages = messages + [{"role": "assistant", "content": assistant_text}]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_only, full_text
