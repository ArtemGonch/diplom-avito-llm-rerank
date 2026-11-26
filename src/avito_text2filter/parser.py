"""Extract conservative UsedCars filters from free text and the supplied catalog."""

import csv
import io
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile


_CLOSED = {"110000", "110001", "110003", "110005", "110006", "110007", "110008", "110010", "633"}
_NAMES = {"164669": "Год выпуска"}
_YEAR = "164669"
_LITERS = "110012"
_MONEY = "price"
_MILEAGE = "2687"
_POWER = "110011"
_MAKE = "110000"
_MODEL = "110001"

# Aliases point to catalog labels. They do not introduce IDs or break catalog ties.
_ALIASES = {
    "110000": {
        "Audi": "ауди", "BMW": "бмв|бэ эм вэ", "Chevrolet": "шевроле|шевролет",
        "Chery": "чери|черри", "Ford": "форд", "Geely": "джили", "Haval": "хавал|хавейл",
        "Honda": "хонда", "Hyundai": "хендай|хёндай|хундай|хюндай", "Infiniti": "инфинити",
        "Kia": "киа|кия", "Land Rover": "ленд ровер|ланд ровер", "Lexus": "лексус",
        "Mazda": "мазда", "Mercedes-Benz": "мерседес|мерс|mercedes", "Mitsubishi": "митсубиси|митсубиши",
        "Nissan": "ниссан", "Peugeot": "пежо", "Renault": "рено", "Skoda": "шкода|škoda",
        "Subaru": "субару", "Suzuki": "сузуки", "Toyota": "тойота|тоёта", "Volkswagen": "фольксваген|вольксваген|vw",
        "Volvo": "вольво", "ВАЗ (LADA)": "ваз|лада|lada",
    },
    "110001": {
        "2113 Samara": "2113", "2114 Samara": "2114", "2115 Samara": "2115",
        "A4": "а4", "A6": "а6", "Camry": "камри|кэмри", "Ceed": "сид|ceed|cee'd",
        "Corolla": "королла|корола", "Creta": "крета|грета", "CX-5": "сх5|cx5|cx 5|сх 5", "Duster": "дастер",
        "Elantra": "элантра",
        "Focus": "фокус", "Granta": "гранта", "Kalina": "калина", "Kodiaq": "кодиак",
        "Largus": "ларгус", "Logan": "логан", "Niva": "нива", "Octavia": "октавия",
        "Polo": "поло", "Priora": "приора|преора", "Q5": "ку5", "Q7": "ку7",
        "RAV4": "рав4|рав 4|rav 4", "Rio": "рио", "Santa Fe": "санта фе|сантафе",
        "Solaris": "солярис", "Sportage": "спортейдж|спортаж", "Tucson": "туссан|туксон", "Vesta": "веста", "Tiguan": "тигуан",
        "X-Trail": "икс трейл|икстрейл|xtrail|х трейл", "X5": "х5|x 5", "X6": "х6|x 6",
    },
    "110003": {
        "Седан": "седана|седане", "Внедорожник": "внедорожника|внедорожники", "Хетчбэк": "хэтчбек|хэтчбэк|хетчбек",
        "Универсал": "универсала|универсале", "Минивэн": "минивен|минивена|минивэна", "Пикап": "пикапа",
    },
    "110006": {
        "Бензин": "бензина|бензиновый|бензиновая|на бензине", "Дизель": "дизеля|дизеле|дизельный|дизельная|дизельное",
        "Гибрид": "гибрида|гибридный|гибридная", "Электро": "электромобиль|электрический|электрическая",
        "Газ": "газовый|газовая|на газу",
    },
    "110007": {
        "Полный": "полный привод|полного привода|полным приводом|полноприводный|полноприводная|4wd|awd|4x4|4х4",
        "Передний": "передний привод|переднего привода|передним приводом|переднеприводный|fwd",
        "Задний": "задний привод|заднего привода|задним приводом|заднеприводный|rwd",
    },
    "110008": {
        "Автомат": "акпп|акп|автомата|на автомате|автоматическая коробка|автоматическая коробка передач",
        "Механика": "мкпп|мкп|механики|на механике|механическая коробка|механическая коробка передач",
        "Робот": "робота|роботизированная коробка|ркпп|dsg", "Вариатор": "вариатора|cvt",
    },
    "633": {
        "Белый": "белая|белое|белого|белом", "Чёрный": "черная|черное|черного|черном",
        "Серый": "серая|серое|серого|сером", "Красный": "красная|красное|красного|красном",
        "Синий": "синяя|синее|синего|синем", "Серебряный": "серебристый|серебристая|серебряная",
    },
}

# Additional spellings used by the model linker. Targets are catalog labels, never IDs.
_MODEL_ALIASES = {
    _MAKE: {
        "Bestune": "бестун|бестуне", "Changan": "чанган|чаньган", "Daewoo": "дэу|деу|дайву",
        "Dongfeng": "донгфенг|донфенг", "EXEED": "эксид|ексид", "GAC": "гак", "Higer": "хайгер|хигер",
        "Jetour": "джетур", "Li Auto": "ли авто|лисян|li", "OMODA": "омода", "Rox": "рокс|рох",
        "Tank": "танк", "Volga": "волга", "Voyah": "воя|войя", "Zeekr": "зикр|зикер",
    },
    _MODEL: {
        "3 серия": "bmw 3|бмв 3", "5 серия": "bmw 5|бмв 5", "7 серия": "bmw 7|бмв 7",
        "Caddy": "кадди|кэдди", "Cerato": "церато|цератто|серато", "CLS-класс": "цлс|cls",
        "E-класс AMG": "e63 amg|е63 амг|e 63 amg", "GL-класс": "gl350|gl450|gl500|gl550|гл350",
        "GLS-класс": "gls350|gls400|gls450|gls500|gls580|глс500",
        "Grand Vitara": "гранд витара", "Jolion": "джолион|жолион",
        "L7": "li 7|ли 7", "L8": "li 8|ли 8", "L9": "li 9|ли 9",
        "Land Cruiser": "ленд крузер|ланд крузер|ленд круизер|крузак",
        "Land Cruiser Prado": "ленд крузер прадо|ланд крузер прадо|прадо",
        "Mark II": "марк 2|марк ii|mark 2", "Navara": "навара|наварра",
        "Nexia": "нексия", "Outlander": "аутлендер|аутлэндер", "Pajero": "паджеро",
        "RAV4": "раф 4|раф4|равчик", "Sorento": "соренто", "Sorento Prime": "соренто прайм", "Space Star": "спейс стар|спайс стар",
        "Sportage": "спортэйдж|спортейж|спортридж|спотридж", "Teramont": "терамонт|террамонт",
        "Teramont Pro": "терамонт про|террамонт про", "Teramont X": "терамонт икс|террамонт икс",
        "Tiggo 7 Pro": "тигго 7 про|tiggo 7 про", "Tiggo 7 Pro Max": "тигго 7 про макс|tiggo 7 про max|tiggo 7 про макс",
        "Tiggo 8 Pro": "тигго 8 про|tiggo 8 про", "Tiggo 8 Pro Max": "тигго 8 про макс|tiggo 8 про max|tiggo 8 про макс",
        "Tugella": "тугела|тугелла", "Venza": "венза",
    },
    "110006": {"Дизель": "дизельные|diesel|cdi|tdi", "Гибрид": "hybrid|e-power", "Газ": "с гбо|на гбо"},
    "110008": {"Автомат": "at", "Механика": "mt"},
}

# Common make/model relationships curated by name. The source CSV has no hierarchy.
_MODEL_MAKES = {
    "Audi": "A4|A6|Q5|Q7", "Bestune": "T77|T55|T99", "BMW": "3 серия|5 серия|7 серия|X3|X5|X6|X7",
    "Chery": "Tiggo 4|Tiggo 4 Pro|Tiggo 7|Tiggo 7 Pro|Tiggo 7 Pro Max|Tiggo 8|Tiggo 8 Pro|Tiggo 8 Pro Max",
    "Daewoo": "Matiz|Nexia", "Dodge": "Ram", "Ford": "Focus|Mondeo|Kuga", "Geely": "Atlas|Coolray|Monjaro|Tugella",
    "HAVAL": "Jolion|F7|F7x|Dargo", "Honda": "Accord|Civic|CR-V|Fit", "Hyundai": "Creta|Elantra|Santa Fe|Solaris|Tucson",
    "Kia": "Ceed|Cerato|K5|Optima|Rio|Sorento|Sorento Prime|Soul|Sportage",
    "Li Auto": "L7|L8|L9", "Mazda": "CX-30|CX-5|CX-50|CX-7|CX-9",
    "Mercedes-Benz": "CLS-класс|E-класс AMG|GL-класс|GLS-класс|GLE-класс|GLC-класс|M-класс",
    "Mitsubishi": "Outlander|Pajero|Space Star", "Nissan": "Almera|Juke|Navara|Qashqai|X-Trail",
    "Renault": "Duster|Kaptur|Logan|Sandero", "Skoda": "Fabia|Kodiaq|Octavia|Rapid|Superb",
    "Suzuki": "Grand Vitara|Jimny|Swift", "Toyota": "Camry|Corolla|Highlander|Land Cruiser|Land Cruiser Prado|Mark II|RAV4|Venza",
    "Volkswagen": "Caddy|Passat|Polo|Teramont|Teramont Pro|Teramont X|Tiguan|Touareg",
    "Volga": "C50|K40|K50", "Volvo": "XC40|XC60|XC70|XC90",
    "ВАЗ (LADA)": "2113 Samara|2114 Samara|2115 Samara|Granta|Kalina|Largus|Priora|Vesta|XRAY",
}

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_GLUE = re.compile(r"[\s.()\[\]’'\-‐‑‒–—]*\Z")
_FILLERS = set("а в во с со к на по за для и у мне мой хочу ищу купить машина машину авто автомобиль автомобили нужна нужен новое новый бу б у только пожалуйста лучше что нужен нужна можно без".split())
_CUES = {
    "марка": "110000", "бренд": "110000", "модель": "110001", "комплектация": "110010",
    "поколение": "110005", "кузов": "110003", "цвет": "633", "коробка": "110008",
    "привод": "110007", "двигатель": "110006", "топливо": "110006",
}
_ALT_WORD = r"\b(?:или|либо|or)\b"
_ALT = re.compile(rf"{_ALT_WORD}|(?<=\w)\s*[/|]\s*(?=\w)", re.I)
_SPLIT = re.compile(r"(?<!\d),|,(?!\d)|[;!?\n]|\.\s+(?=[A-ZА-ЯЁ])|\b(?i:но|зато)\b")
_CONJUNCTION = re.compile(r"\bи\b|&", re.I)
_UNIT_SLASH = re.compile(r"\b(?:л\.?\s*/\s*с\.?|км\s*/\s*ч)\b", re.I)
_UNSUPPORTED = re.compile(
    r"\b(?:без\s+(?:дтп|аварий|сколов|царапин|вложений|окрасов?|ржавчины|ремонта|залогов|ограничений|пробега|кредита|торга)|"
    r"не\s+(?:бит[а-яё]*|крашен[а-яё]*|китай[а-яё]*)|один\s+(?:хозяин|владелец)|"
    r"от\s+(?:собственника|владельца)|кредит(?:а|е)?|рассрочк[а-яё]+|после\s+дтп)\b", re.I,
)
_PART_REQUEST = re.compile(
    r"\b(?:запчаст[а-яё]*|разборк[а-яё]*|реш[её]тк[а-яё]*|бампер[а-яё]*|амортизатор[а-яё]*|"
    r"радиатор[а-яё]*|диск[а-яё]*|колодк[а-яё]*|подшипник[а-яё]*|электрокорректор[а-яё]*|"
    r"фар(?:а|ы|у|ой|ами)|(?:ремонт|замена)\s+(?:двигател[а-яё]*|коробк[а-яё]*))\b", re.I,
)
_CODE_LETTERS = str.maketrans("авекмнорстух", "abekmhopctyx")
_CODE = re.compile(r"(?=.{2,9}$)(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_CONTEXTUAL_TERMS = {"at", "mt", "li", "волга"}
_VEHICLE_IDENTITY = {_MAKE, _MODEL}
_MODEL_OPTION_LIMIT = 12

_NUM = r"(?:[0-9]{1,3}(?P<group>[.,])(?:[0-9]{3}(?P=group)){1,3}[0-9]{3}|(?:[0-9]{1,3}(?:[ \u00a0\u202f][0-9]{3})+|[0-9]{1,12})(?:[.,][0-9]{1,6})?)"
_SCALE = r"(?:миллиард(?:а|ов)?|млрд\.?|миллион(?:а|ов)?|млн\.?|тысяч(?:а|и|у)?|тыс\.?|[kк])"
_CURRENCY = r"(?:руб(?:\.|ль|ля|лей)?|₽|р\.|р|rub)"
_FOREIGN = r"(?:евро|euro|eur|€|доллар(?:а|ов)?|долл\.?|usd|\$|£)"
_DISTANCE = r"(?:км|km|километр(?:а|ов)?)"
_HORSEPOWER = r"(?:л\s*[./]?\s*с\.?|лошадин(?:ые|ых|ая)\s+сил(?:а|ы)?|hp)"
_VOLUME = r"(?:литр(?:а|ов)?|л\.?|l)"
_YEAR_UNIT = r"(?:года\s+выпуска|год(?:а|ов)?|г\.\s*в\.?|г\.?)"
_UNSUPPORTED_UNIT = r"(?:км\s*/\s*ч|квт|kw|куб\.\s*см|см[³3])"
_AMOUNT = re.compile(
    rf"(?<![\w.,])(?:(?P<presymbol>[₽$€£])[^\S\n]*)?(?P<number>{_NUM})(?:[^\S\n]*(?P<scale>{_SCALE})(?!\w))?"
    rf"(?:[^\S\n]*(?:(?P<money>{_CURRENCY})|(?P<unsupported>{_UNSUPPORTED_UNIT}|{_FOREIGN})|(?P<mileage>{_DISTANCE})|"
    rf"(?P<power>{_HORSEPOWER})|(?P<liters>{_VOLUME})|(?P<year>{_YEAR_UNIT}))(?!\w))?(?![\w.,]\d|\w)", re.I,
)
_FIELD = r"(?:цена|стоимость|бюджет|пробег(?:ом|а)?|год(?:у|ом)?(?:\s+выпуска)?|об[ъь][её]м(?:ом)?(?:\s+двигателя)?|мощность(?:ю)?)"
_BETWEEN = re.compile(r"\s*(?:[-–—−]|до|по)\s*\Z", re.I)
_UPPER = {"до", "по", "не дороже", "не более", "не больше", "не выше", "не моложе", "не позднее", "не позже", "как максимум", "<=", "≤"}
_LOWER = {"от", "с", "не дешевле", "не менее", "не меньше", "не ниже", "не старше", "не ранее", "не раньше", "как минимум", ">=", "≥"}
_STRICT_UPPER = {"дешевле", "старше", "ниже", "меньше", "менее", "<"}
_STRICT_LOWER = {"дороже", "моложе", "более", "больше", "выше", "после", ">"}
_UNITS = {"money": _MONEY, "mileage": _MILEAGE, "power": _POWER, "liters": _LITERS, "year": _YEAR}


def _operator_pattern(operators):
    return "|".join(r"\s+".join(map(re.escape, operator.split())) for operator in sorted(operators, key=lambda item: (-len(item), item)))


_OP = _operator_pattern(_UPPER | _LOWER | _STRICT_UPPER | _STRICT_LOWER | {"между"})
_POSITIVE_AFTER_NE = _operator_pattern({operator[3:] for operator in _UPPER | _LOWER if operator.startswith("не ")})
_NEG = re.compile(
    rf"(?<!\w)(?:без|кроме|исключая|исключить|исключите|нет|ни|не(?!\s+(?:{_POSITIVE_AFTER_NE})\b))(?!\w)", re.I,
)
_PREFIX = re.compile(rf"(?<!\w)(?:(?P<field>{_FIELD})\s*[:=]?\s*(?:(?P<operator>{_OP})\s*)?|(?P<bound>{_OP})\s*)\Z", re.I)


def _normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")


def _key(text):
    return tuple(_normalize(match.group()) for match in _TOKEN.finditer(text))


def _model_code(words):
    code = "".join(words).translate(_CODE_LETTERS)
    return code if _CODE.fullmatch(code) else None


def _span(query, start, end):
    return {"evidence": query[start:end], "start": start, "end": end}


def _unresolved(query, start, end, reason):
    return {**_span(query, start, end), "reason": reason}


@dataclass(frozen=True, order=True)
class _Value:
    attr_id: str
    attr_name: str
    value_id: str
    value_name: str

    def grounded(self, query, start, end):
        return {**_span(query, start, end), **vars(self)}


@dataclass(frozen=True)
class _Clause:
    start: int
    end: int
    alternative: bool
    negative_from: int | None
    contains_negative: bool


def _clauses(query):
    start = 0
    for separator in [*_SPLIT.finditer(query), None]:
        end = separator.start() if separator else len(query)
        text = query[start:end]
        # "White, or black" is one disjunction. Commas between digits are decimal punctuation.
        if separator and separator.group() == "," and re.match(r"\s*(?:или|либо|or)\b", query[end + 1 :], re.I):
            continue
        unknown = list(_UNSUPPORTED.finditer(text))
        negatives = list(_NEG.finditer(text))
        negative_from = next((start + marker.start() for marker in negatives if not any(
            item.start() <= marker.start() < item.end() for item in unknown
        )), None)
        alternative = bool(_ALT.search(_UNIT_SLASH.sub(" ", text)))
        yield _Clause(start, end, alternative, negative_from, bool(negatives))
        start = separator.end() if separator else len(query)


def _model_blocked(clauses, start):
    return any(clause.start <= start < clause.end and (clause.alternative or clause.contains_negative) for clause in clauses)


def _blocked_evidence(query, clauses):
    for clause in clauses:
        if clause.alternative:
            fragment = query[clause.start : clause.end]
            start = clause.start + len(fragment) - len(fragment.lstrip())
            yield _unresolved(query, start, clause.start + len(fragment.rstrip()), "alternative")
        elif clause.negative_from is not None:
            end = clause.negative_from + len(query[clause.negative_from : clause.end].rstrip())
            yield _unresolved(query, clause.negative_from, end, "negated")


def _remaining_evidence(query, start, end):
    for conjunction in [*_CONJUNCTION.finditer(query, start, end), None]:
        end_of_fragment = conjunction.start() if conjunction else end
        tokens = [
            match for match in _TOKEN.finditer(query, start, end_of_fragment)
            if _normalize(match.group()) not in _FILLERS and _normalize(match.group()) not in _CUES
        ]
        if tokens:
            yield _unresolved(query, tokens[0].start(), tokens[-1].end(), "unrecognized")
        if conjunction:
            start = conjunction.end()


def _intersects(row, spans):
    return any(row["start"] < other["end"] and other["start"] < row["end"] for other in spans)


def _edits(token):
    return {token, *(token[:index] + token[index + 1 :] for index in range(len(token)))}


def _one_edit(left, right):
    if len(left) == len(right):
        differences = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        if len(differences) < 2:
            return bool(differences)
        first, second = differences[:2]
        return len(differences) == 2 and second == first + 1 and (left[first], left[second]) == (right[second], right[first])
    return min(left, right, key=len) in _edits(max(left, right, key=len))


class Catalog:
    """Read the source names and string IDs directly from the UsedCars archive."""

    def __init__(self, rows):
        self._names = _NAMES.copy()
        fields = defaultdict(set)
        for row in rows:
            attr_id, attr_name = row["attribute_id"], row["attribute_name"]
            value_id, value_name = row["value_id"], row["value_name"]
            self._names.setdefault(attr_id, attr_name)
            if attr_id in _CLOSED and value_id and value_name:
                fields[attr_id].add(_Value(attr_id, attr_name, value_id, value_name))

        labels = defaultdict(set)
        for attr_id, values in fields.items():
            aliases = {_key(label): variants for label, variants in _ALIASES.get(attr_id, {}).items()}
            for value in values:
                label = _key(value.value_name)
                labels[label].add(value)
                for alias in aliases.get(label, "").split("|"):
                    if alias:
                        labels[_key(alias)].add(value)
        self._terms = defaultdict(dict)
        self._nearby = defaultdict(set)
        for label, values in labels.items():
            if not label:
                continue
            options = tuple(sorted(values))
            self._terms[label[0]][label] = options
            word = label[0]
            if len(label) == 1 and len(word) >= 5 and word.isalpha() and len(options) == 1 and options[0].attr_id in {"110000", "110001"}:
                for deletion in _edits(word):
                    self._nearby[deletion].add(word)
                for index in range(len(word) - 1):
                    swapped = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
                    self._nearby[swapped].add(word)

        expanded = defaultdict(set)
        codes = defaultdict(set)
        for term, values in labels.items():
            expanded[term].update(value for value in values if value.attr_id not in {"110005", "110010"})
        for attr_id, aliases in _MODEL_ALIASES.items():
            variants = {_key(label): aliases for label, aliases in aliases.items()}
            for value in fields[attr_id]:
                for alias in variants.get(_key(value.value_name), "").split("|"):
                    if alias:
                        expanded[_key(alias)].add(value)
        self._model_terms = defaultdict(dict)
        for term, values in expanded.items():
            if not term or not values:
                continue
            self._model_terms[term[0]][term] = tuple(sorted(values))
            code = _model_code(term)
            if code:
                codes[code].update(value for value in values if value.attr_id == _MODEL)
        self._model_codes = {code: tuple(sorted(values)) for code, values in codes.items() if values}
        makes = defaultdict(set)
        for value in fields[_MAKE]:
            makes[_key(value.value_name)].add(value)
        self._model_makes = {}
        for make, models in _MODEL_MAKES.items():
            if len(makes[_key(make)]) == 1:
                for model in models.split("|"):
                    self._model_makes[_key(model)] = next(iter(makes[_key(make)]))

    @classmethod
    def from_zip(cls, path: str | Path):
        with ZipFile(path) as archive, archive.open("attribute_values.csv") as member:
            with io.TextIOWrapper(member, encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = {"attribute_id", "attribute_name", "value_id", "value_name"}
                if not fields.issubset(reader.fieldnames or ()):
                    raise ValueError("Invalid attribute_values.csv columns")
                return cls(reader)

    def _closed_mentions(self, query, *, expanded=False):
        lookup = self._model_terms if expanded else self._terms
        tokens = list(_TOKEN.finditer(query))
        normalized = [_normalize(token.group()) for token in tokens]
        mentions = []
        for index, word in enumerate(normalized):
            terms = lookup.get(word, {})
            if expanded:
                terms = dict(terms)
                for length in range(1, min(3, len(tokens) - index) + 1):
                    term = tuple(normalized[index:index + length])
                    options = self._model_codes.get(_model_code(term))
                    if options:
                        terms.setdefault(term, options)
            for term, options in terms.items():
                end_index = index + len(term)
                if term != tuple(normalized[index:end_index]):
                    continue
                if len(term) > 1 and not all(_GLUE.fullmatch(query[tokens[j].end() : tokens[j + 1].start()]) for j in range(index, end_index - 1)):
                    continue
                hint = None
                if index and re.fullmatch(r"\s*[:=]?\s*", query[tokens[index - 1].end() : tokens[index].start()]):
                    hint = _CUES.get(normalized[index - 1])
                if hint:
                    options = tuple(value for value in options if value.attr_id == hint)
                source = tokens[index].group()
                if len(term) == 1 and not hint and (word in _FILLERS or word in _CUES or len(word) == 1 or (
                    len(word) == 2 and source.isalpha() and not source.isupper() and word != "vw"
                    and not (expanded and word in _CONTEXTUAL_TERMS)
                )):
                    continue
                if len(term) == 1 and word.isdecimal() and (len(word) < 3 or hint != "110001" and word.startswith("0")):
                    continue
                if options:
                    mentions.append((tokens[index].start(), tokens[end_index - 1].end(), options, hint))
        return mentions

    def _numbers(self, query, closed):
        amounts = list(_AMOUNT.finditer(query))
        index = 0
        while index < len(amounts):
            first = amounts[index]
            index += 1
            prefix = _PREFIX.search(query[: first.start()])
            operator = " ".join((prefix["operator"] or prefix["bound"] or "").lower().split()) if prefix else ""
            field = _field_id(prefix["field"] if prefix else None)
            second = amounts[index] if index < len(amounts) else None
            join = query[first.end() : second.start()] if second else ""
            connected = _BETWEEN.fullmatch(join) or operator == "между" and re.fullmatch(r"\s*и\s*", join, re.I)
            if second and join.strip().lower() in {"до", "по"} and operator not in {"от", "с"}:
                connected = _can_share_range(first, second)
            if second and connected:
                index += 1
            else:
                second = None
            start = prefix.start() if prefix else first.start()
            end = (second or first).end()
            if not second and not operator and query[end : end + 1] == "+":
                operator = "от"
                end += 1
            if any(left <= first.start() and right >= end and any(value.attr_id in {"110005", "110001"} and not query[left:right].isdecimal() for value in options)
                   for left, right, options, _ in closed):
                continue
            if not operator and not field and not second and any(
                left == first.start() and right == end and hint == "110001" for left, right, _, hint in closed
            ):
                continue
            attr_id, lower, upper, strict_lower, strict_upper, reason = _numeric_filter(first, second, field, operator)
            if attr_id is None and not reason:
                continue
            if not reason and (attr_id not in self._names or re.search(r"[-−]\s*$", query[: first.start()])):
                reason = "unsupported"
            if not reason and attr_id == _YEAR and start == first.start() and not second and not first["year"] and any(
                left == first.start() and right == first.end() and any(value.attr_id == "110001" for value in options)
                for left, right, options, _ in closed
            ):
                reason = "ambiguous"
            if reason:
                yield _unresolved(query, start, end, reason)
                continue
            row = {**_span(query, start, end), "attr_id": attr_id, "attr_name": self._names[attr_id], "min": lower, "max": upper}
            if strict_lower:
                row["min_inclusive"] = False
            if strict_upper:
                row["max_inclusive"] = False
            yield row

    def parse(self, query: str) -> dict:
        """Return accepted filters and unresolved evidence, in source order."""
        clauses = list(_clauses(query))
        closed = self._closed_mentions(query)
        blocked = list(_blocked_evidence(query, clauses))
        unsupported = [
            row for row in (_unresolved(query, *match.span(), "unsupported") for match in _UNSUPPORTED.finditer(query))
            if not _intersects(row, blocked)
        ]
        unsupported.extend(blocked)
        numbers = [row for row in self._numbers(query, closed) if not _intersects(row, unsupported)]
        filters, unresolved = [], list(unsupported)
        for row in numbers:
            if "reason" in row:
                unresolved.append(row)
            else:
                filters.append(row)

        used = []
        for start, end, options, hint in sorted(closed, key=lambda item: (item[0], -item[1])):
            if _intersects(_span(query, start, end), numbers + unsupported):
                continue
            if any((start >= left and end <= right) and not (
                start == left and end < right
                and len(options) == len(selected) == 1
                and selected[0].attr_id == "110001" and options[0].attr_id == "110000"
            ) for left, right, selected in used):
                continue
            if any(start < right and end > right for _, right, _ in used):
                continue
            used.append((start, end, options))
            reason = None
            if len(options) > 1:
                reason = "ambiguous"
            elif options[0].attr_id == "110010" and not hint:
                reason = "unqualified"
            elif options[0].attr_id == "110007" and _normalize(query[start:end]) in {"полный", "передний", "задний"} and not hint:
                reason = "unqualified"
            if reason:
                row = _unresolved(query, start, end, reason)
                row["candidates"] = [value.grounded(query, start, end) for value in options]
                unresolved.append(row)
            else:
                filters.append(options[0].grounded(query, start, end))

        covered = filters + unresolved
        filters = _reconcile(query, filters, unresolved)
        for clause in clauses:
            start = clause.start
            for row in [*sorted(covered, key=lambda row: row["start"]), {"start": clause.end, "end": clause.end}]:
                end = min(row["start"], clause.end)
                if end > start:
                    unresolved.extend(_remaining_evidence(query, start, end))
                start = max(start, row["end"])
                if start >= clause.end:
                    break
        return {"filters": sorted(filters, key=_sort_span), "unresolved": sorted(unresolved, key=_sort_span)}

    def candidates(self, query: str) -> list[dict]:
        """Return additional grounded proposals a model may select by index."""
        return self._additional_candidates(query, self.parse(query))

    def _additional_candidates(self, query, parsed):
        clauses = list(_clauses(query))
        occupied = {row["attr_id"] for row in parsed["filters"]}
        occupied.update(value["attr_id"] for row in parsed["unresolved"] if row["reason"] in {"ambiguous", "conflict"} for value in row.get("candidates", []))
        proposals = {}
        for row in parsed["unresolved"]:
            if row["reason"] == "unqualified" and not _model_blocked(clauses, row["start"]):
                for value in row.get("candidates", ()):
                    if value["attr_id"] not in occupied:
                        proposals[_proposal_key(value)] = value
            if row["reason"] != "unrecognized":
                continue
            for match in _TOKEN.finditer(query, row["start"], row["end"]):
                word = _normalize(match.group())
                if len(word) < 5 or not word.isalpha() or _model_blocked(clauses, match.start()):
                    continue
                nearby = {term for deletion in _edits(word) for term in self._nearby.get(deletion, ())}
                for term in nearby:
                    options = self._terms[term][(term,)]
                    if not _one_edit(word, term) or options[0].attr_id in occupied:
                        continue
                    value = options[0].grounded(query, *match.span())
                    proposals[_proposal_key(value)] = value
        return [proposals[key] for key in sorted(proposals)]

    def model_candidates(self, query: str, baseline: dict) -> list[dict]:
        """Return additional closed-value options for the expanded student linker."""
        blocked = [
            row for row in baseline["unresolved"]
            if row["reason"] in {"alternative", "negated", "unsupported", "invalid_range", "conflict"}
        ]
        conflicted_attributes = {
            item["attr_id"]
            for row in blocked if row["reason"] == "conflict"
            for item in row.get("candidates", ())
        }
        blocked.extend(row for row in baseline["filters"] if "value_id" not in row)
        parts = [clause for clause in _clauses(query) if _PART_REQUEST.search(query, clause.start, clause.end)]
        by_span = defaultdict(lambda: defaultdict(set))
        for start, end, values, hint in self._closed_mentions(query, expanded=True):
            if _intersects(_span(query, start, end), blocked):
                continue
            if query[start:end].isdecimal() and hint != _MODEL:
                continue
            for value in values:
                by_span[start, end][value.attr_id].add(value)

        def in_part_request(start):
            return any(clause.start <= start < clause.end for clause in parts)

        mentions = [
            (start, end, next(iter(options)))
            for (start, end), attrs in by_span.items()
            for attr, options in attrs.items()
            if len(options) == 1 and (attr in _VEHICLE_IDENTITY or not in_part_request(start))
        ]
        explicit_makes = [(start, end, value) for start, end, value in mentions if value.attr_id == _MAKE]
        vehicle_context = any(value.attr_id in _VEHICLE_IDENTITY for _, _, value in mentions)
        has_engine_volume = any(row["attr_id"] == _LITERS for row in baseline["filters"])
        catalog_makes = {
            (start, end): self._model_makes.get(_key(value.value_name))
            for start, end, value in mentions if value.attr_id == _MODEL
        }

        proposals = {}
        existing = {(row["attr_id"], row.get("value_id")) for row in baseline["filters"]}

        def add(row):
            key = row["attr_id"], row["value_id"]
            if row["attr_id"] not in conflicted_attributes and key not in existing:
                proposals.setdefault(key, row)

        longer = defaultdict(list)
        for start, end, value in sorted(mentions, key=lambda item: (item[0], -item[1], item[2])):
            term = _normalize(query[start:end])
            attr = value.attr_id
            if term in {"at", "mt"} and not (vehicle_context or has_engine_volume):
                continue
            if term in {"li", "волга"} and value not in catalog_makes.values():
                continue
            if attr == _MAKE and any(
                start <= left <= end and right > end and owner != value
                for (left, right), owner in catalog_makes.items() if owner
            ):
                continue
            if any(left <= start and end <= right for left, right in longer[attr]):
                continue
            if _MAKE in by_span[start, end] and _MODEL in by_span[start, end]:
                owner = catalog_makes.get((start, end))
                known_model = owner is not None and any(
                    right <= start and make == owner for _, right, make in explicit_makes
                )
                if attr == _MAKE and known_model:
                    continue
                if attr == _MODEL and not known_model:
                    continue
            add(value.grounded(query, start, end))
            longer[attr].append((start, end))
            if attr == _MODEL:
                owner = catalog_makes[start, end]
                if owner and not any(not (start <= left and right <= end) or make == owner
                                     for left, right, make in explicit_makes):
                    add(owner.grounded(query, start, end))

        previous = self._additional_candidates(query, baseline)
        for row in previous:
            if not in_part_request(row["start"]) and not _intersects(row, blocked):
                add(row)
        return sorted(list(proposals.values())[:_MODEL_OPTION_LIMIT], key=_proposal_key)


def _sort_span(row):
    return row["start"], row["end"]


def _proposal_key(row):
    return row["start"], row["end"], row["attr_id"], row["value_id"]


def _field_id(field):
    if field:
        word = _normalize(field)
        if word.startswith("пробег"):
            return _MILEAGE
        if word.startswith("год"):
            return _YEAR
        if word.startswith("об"):
            return _LITERS
        if word.startswith("мощность"):
            return _POWER
        return _MONEY
    return None


def _amount(match, shared_scale=None):
    literal = re.sub(r"\s", "", match["number"])
    number = Decimal(literal.replace(match["group"], "") if match["group"] else literal.replace(",", "."))
    scale = _normalize((match["scale"] or shared_scale or "").strip("."))
    if scale.startswith(("млрд", "миллиард")):
        return number * 1_000_000_000
    if scale.startswith(("млн", "миллион")):
        return number * 1_000_000
    if scale:
        return number * 1000
    return number


def _can_share_range(first, second):
    first_units = {unit for unit in _UNITS if first[unit]}
    second_units = {unit for unit in _UNITS if second[unit]}
    if first_units and second_units and first_units != second_units:
        return False
    shared = second["scale"] if not first_units else None
    return _amount(first, shared) <= _amount(second)


def _numeric_filter(first, second, field, operator):
    amounts = (first, second) if second else (first,)
    units = {attr_id for unit, attr_id in _UNITS.items() if any(amount[unit] for amount in amounts)}
    # A trailing scale applies to a bare endpoint in "от 1 до 2 млн".
    shared_scale = second["scale"] if second and not first["scale"] and not any(first[unit] for unit in _UNITS) else None
    lower_value = _amount(first, shared_scale)
    upper_value = _amount(second) if second else lower_value
    year_range = all(1886 <= number <= 2100 and number == number.to_integral_value() for number in (lower_value, upper_value))
    if any(amount["presymbol"] == "₽" for amount in amounts):
        units.add(_MONEY)
    if any(amount["unsupported"] or amount["presymbol"] in {"$", "€", "£"} for amount in amounts) or len(units) > 1 or field and units and field not in units:
        return None, None, None, False, False, "unsupported"
    attr_id = field or next(iter(units), None)
    if not attr_id and any(amount["scale"] for amount in amounts):
        attr_id = _MONEY
    if not attr_id and year_range:
        attr_id = _YEAR
    if not attr_id:
        return None, None, None, False, False, "unqualified" if operator or second else None
    if attr_id == _YEAR and not year_range or min(lower_value, upper_value) < 0 or attr_id == _LITERS and min(lower_value, upper_value) == 0:
        return attr_id, None, None, False, False, "invalid_range"
    if second and operator not in {"", "от", "с", "между"} or not second and operator == "между":
        return attr_id, None, None, False, False, "invalid_range"
    lower, upper = lower_value, upper_value
    strict_lower = strict_upper = False
    if not second:
        if operator in _UPPER or operator in _STRICT_UPPER:
            lower, strict_upper = None, operator in _STRICT_UPPER
        elif operator in _LOWER or operator in _STRICT_LOWER:
            upper, strict_lower = None, operator in _STRICT_LOWER
    if attr_id != _LITERS:
        if any(value is not None and value != value.to_integral_value() for value in (lower, upper)):
            return attr_id, None, None, False, False, "invalid_range"
        lower = int(lower) + strict_lower if lower is not None else None
        upper = int(upper) - strict_upper if upper is not None else None
        strict_lower = strict_upper = False
    else:
        lower = float(lower) if lower is not None else None
        upper = float(upper) if upper is not None else None
    if upper is not None and (upper < 0 or lower is not None and lower > upper):
        return attr_id, None, None, False, False, "invalid_range"
    return attr_id, lower, upper, strict_lower, strict_upper, None


def _reconcile(query, filters, unresolved):
    by_attr = defaultdict(list)
    for row in sorted(filters, key=_sort_span):
        by_attr[row["attr_id"]].append(row)
    accepted = []
    for rows in by_attr.values():
        first = rows[0]
        if len(rows) == 1 or "value_id" in first and len({row["value_id"] for row in rows}) == 1:
            accepted.append(first)
            continue
        if "value_id" not in first:
            lower, lower_open = max((row["min"] if row["min"] is not None else -math.inf, row.get("min_inclusive") is False) for row in rows)
            upper, upper_closed = min((row["max"] if row["max"] is not None else math.inf, row.get("max_inclusive") is not False) for row in rows)
            if lower < upper or lower == upper and not lower_open and upper_closed:
                combined = {**first, **_span(query, first["start"], max(row["end"] for row in rows))}
                combined.update(min=lower if math.isfinite(lower) else None, max=upper if math.isfinite(upper) else None)
                combined["evidence_parts"] = [_span(query, row["start"], row["end"]) for row in rows]
                combined.pop("min_inclusive", None)
                combined.pop("max_inclusive", None)
                if lower_open:
                    combined["min_inclusive"] = False
                if not upper_closed:
                    combined["max_inclusive"] = False
                accepted.append(combined)
                continue
        for row in rows:
            conflict = _unresolved(query, row["start"], row["end"], "conflict")
            if "value_id" in row:
                conflict["candidates"] = [row]
            unresolved.append(conflict)
    return accepted
