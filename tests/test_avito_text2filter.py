import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_text2filter import Catalog


class Text2FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog_values = [
            ("110000", "Марка", "1", "Toyota"),
            ("110000", "Марка", "2", "Honda"),
            ("110000", "Марка", "3", "ВАЗ (LADA)"),
            ("110000", "Марка", "4", "Hyundai"),
            ("110000", "Марка", "5", "Solaris"),
            ("110000", "Марка", "6", "Kia"),
            ("110000", "Марка", "7", "Volvo"),
            ("110000", "Марка", "8", "OMODA"),
            ("110000", "Марка", "9", "Mercedes-Benz"),
            ("110000", "Марка", "101", "BMW"),
            ("110000", "Марка", "102", "Ferrari"),
            ("110000", "Марка", "103", "ГАЗ"),
            ("110000", "Марка", "104", "Peugeot"),
            ("110001", "Модель", "10", "Camry"),
            ("110001", "Модель", "11", "Honda Accord"),
            ("110001", "Модель", "12", "2114"),
            ("110001", "Модель", "13", "2008"),
            ("110001", "Модель", "14", "Solaris"),
            ("110001", "Модель", "15", "Sorento"),
            ("110001", "Модель", "16", "XC60"),
            ("110001", "Модель", "17", "C5"),
            ("110001", "Модель", "18", "C5"),
            ("110001", "Модель", "19", "GL-класс"),
            ("110001", "Модель", "111", "5 серия"),
            ("110001", "Модель", "112", "Venza"),
            ("110001", "Модель", "113", "F12berlinetta"),
            ("110001", "Модель", "114", "C-класс"),
            ("110001", "Модель", "115", "ГАЗель Next"),
            ("110001", "Модель", "116", "Land Cruiser"),
            ("110001", "Модель", "117", "3008"),
            ("110010", "Комплектация", "101", "Sorento"),
            ("110002", "Год выпуска", "50001", "2018"),
            ("110003", "Тип кузова", "20", "Седан"),
            ("110005", "Поколение", "30", "VII (2011—2014)"),
            ("110006", "Тип двигателя", "40", "Бензин"),
            ("110006", "Тип двигателя", "41", "Дизель"),
            ("110007", "Привод", "50", "Полный"),
            ("110007", "Привод", "51", "Передний"),
            ("110008", "Коробка передач", "60", "Автомат"),
            ("110008", "Коробка передач", "61", "Механика"),
            ("110011", "Мощность двигателя, л. с.", "", ""),
            ("110012", "Объём двигателя, л", "", ""),
            ("2687", "Пробег, км", "", ""),
            ("633", "Цвет", "70", "Чёрный"),
            ("633", "Цвет", "71", "Белый"),
            ("price", "Цена", "", ""),
        ]
        catalog_values.extend(
            ("110010", "Комплектация", str(1000 + index), "Comfort")
            for index in range(25)
        )
        csv_data = io.StringIO()
        writer = csv.writer(csv_data)
        writer.writerow(["attribute_id", "attribute_name", "value_id", "value_name"])
        writer.writerows(catalog_values)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        path = Path(directory.name) / "catalog.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("attribute_values.csv", csv_data.getvalue())
            archive.writestr("alternative_attribute_values.csv", "ignored")
        cls.catalog = Catalog.from_zip(path)

    def assert_evidence(self, query, rows):
        for row in rows:
            self.assertEqual(query[row["start"] : row["end"]], row["evidence"])

    def test_converts_listed_gearbox_and_rubles_preserving_unsupported_intent(self):
        query = "  автомат до 1,5 млн без ДТП"
        result = self.catalog.parse(query)
        self.assertEqual(json.loads(json.dumps(result, ensure_ascii=False)), result)
        self.assertEqual(
            [(row["attr_id"], row.get("value_id"), row.get("max")) for row in result["filters"]],
            [("110008", "60", None), ("price", None, 1500000)],
        )
        self.assertIn("без ДТП", [row["evidence"] for row in result["unresolved"]])
        self.assert_evidence(query, result["filters"] + result["unresolved"])

    def test_typed_ranges_exact_values_and_unicode_offsets(self):
        query = "🚘 Тойота Camry, 2018–2021 г., пробег до 120\u202f000 км, 2,5 л, 150 л. с., ЧЕРНЫЙ"
        result = self.catalog.parse(query)
        filters = {row["attr_id"]: row for row in result["filters"]}
        self.assertEqual(set(filters), {"110000", "110001", "164669", "2687", "110012", "110011", "633"})
        self.assertEqual(filters["110000"]["value_id"], "1")
        self.assertEqual(filters["633"]["value_id"], "70")
        for attr_id, lower, upper in [("164669", 2018, 2021), ("2687", None, 120000), ("110012", 2.5, 2.5), ("110011", 150, 150)]:
            self.assertEqual((filters[attr_id]["min"], filters[attr_id]["max"]), (lower, upper))
        self.assert_evidence(query, result["filters"] + result["unresolved"])

    def test_shared_range_suffix_and_strict_integer_bounds(self):
        query = "цена от 1,2 до 1,8 млн, пробег меньше 100 тысяч км, не старше 2019 года"
        rows = self.catalog.parse(query)["filters"]
        self.assertEqual(
            [(row["attr_id"], row["min"], row["max"]) for row in rows],
            [("price", 1200000, 1800000), ("2687", None, 99999), ("164669", 2019, None)],
        )
        self.assert_evidence(query, rows)

    def test_all_duplicate_ids_are_reported_and_none_can_be_selected_by_a_model(self):
        query = "Toyota комплектация Comfort"
        result = self.catalog.parse(query)
        self.assertEqual([row["attr_id"] for row in result["filters"]], ["110000"])
        ambiguous = next(row for row in result["unresolved"] if row["reason"] == "ambiguous")
        self.assertEqual(len(ambiguous["candidates"]), 25)
        self.assert_evidence(query, [ambiguous, *ambiguous["candidates"]])
        self.assertFalse(self.catalog.candidates(query))

    def test_negations_disjunctions_and_conflicts_do_not_become_positive_filters(self):
        for fragment in (
            "не автомат", "без механики", "кроме Toyota", "белый либо чёрный", "белый/чёрный",
            "белый, или совершенно другой цвет", "Honda Accord или Toyota Camry", "белый и черный",
            "до 1 млн или до 2 млн", "Toyota или неизвестная марка", "не Toyota и Honda",
        ):
            with self.subTest(fragment=fragment):
                result = self.catalog.parse(fragment + ", дизель")
                self.assertEqual([row["value_id"] for row in result["filters"]], ["41"])
                self.assertTrue(result["unresolved"])

    def test_unmatched_numeric_and_unknown_conditions_are_not_invented(self):
        query = "Aerocar, 100, кредит, без ДТП и один владелец"
        result = self.catalog.parse(query)
        self.assertFalse(result["filters"])
        self.assertTrue(result["unresolved"])
        self.assert_evidence(query, result["unresolved"])

    def test_merged_bounds_keep_intervening_conditions_and_separate_conjunctive_hints(self):
        query = "цена от 1 млн, панорамная крыша, цена до 2 млн, фаркоп и подогрев сидений"
        result = self.catalog.parse(query)
        row, = result["filters"]
        self.assertEqual((row["min"], row["max"]), (1000000, 2000000))
        self.assertEqual([part["evidence"] for part in row["evidence_parts"]], ["цена от 1 млн", "цена до 2 млн"])
        self.assertEqual([row["evidence"] for row in result["unresolved"]], ["панорамная крыша", "фаркоп", "подогрев сидений"])
        self.assert_evidence(query, [row, *row["evidence_parts"], *result["unresolved"]])

    def test_negation_and_disjunction_are_each_preserved_as_a_single_fragment(self):
        query = "не дизель, от 2020 года; белый, или чёрный"
        result = self.catalog.parse(query)
        self.assertEqual([(row["reason"], row["evidence"]) for row in result["unresolved"]], [
            ("negated", "не дизель"), ("alternative", "белый, или чёрный"),
        ])
        self.assert_evidence(query, result["unresolved"])

    def test_conflicting_ranges_and_units_are_not_silently_reinterpreted(self):
        for query in ("пробег от 150 до 100 тыс. км", "цена до 1 млн, цена от 2 млн", "до 5000 евро", "до -100 км", "€2020", "мощность 150 кВт", "до 120 км/ч"):
            with self.subTest(query=query):
                result = self.catalog.parse(query)
                self.assertFalse(result["filters"])
                self.assertTrue(result["unresolved"])

    def test_generation_and_numeric_model_stay_grounded_in_catalog(self):
        result = self.catalog.parse("ВАЗ 2114, VII (2011—2014)")
        self.assertEqual([row["value_id"] for row in result["filters"]], ["3", "12", "30"])
        self.assertEqual(self.catalog.parse("модель 2008")["filters"][0]["value_id"], "13")
        self.assertEqual(self.catalog.parse("год выпуска 2008")["filters"][0]["min"], 2008)
        both = self.catalog.parse("2008")
        self.assertFalse(both["filters"])
        self.assertEqual(both["unresolved"][0]["reason"], "ambiguous")

    def test_short_range_and_unit_notation(self):
        query = "10-15 тыс. км, от 2018-2021 г.в., Toyota не меньше 180 л/с, до ₽1.500.000"
        rows = self.catalog.parse(query)["filters"]
        self.assertEqual(
            [(row["attr_id"], row.get("min"), row.get("max")) for row in rows],
            [("2687", 10000, 15000), ("164669", 2018, 2021), ("110000", None, None), ("110011", 180, None), ("price", None, 1500000)],
        )
        self.assert_evidence(query, rows)
        self.assertEqual(self.catalog.parse("2018+")["filters"][0]["min"], 2018)
        self.assertIsNone(self.catalog.parse("2018+")["filters"][0]["max"])

    def test_model_gets_only_bounded_unique_catalog_proposals_for_a_misspelling(self):
        query = "Tooyota, автомат"
        candidates = self.catalog.candidates(query)
        self.assertEqual([(row["attr_id"], row["value_id"], row["evidence"]) for row in candidates], [("110000", "1", "Tooyota")])
        self.assert_evidence(query, candidates)
        for unsafe in ("кроме Tooyota", "не Tooyota", "Tooyota или Honda", "Honda или Tooyota", "Honda, Tooyota"):
            with self.subTest(unsafe=unsafe):
                self.assertFalse(self.catalog.candidates(unsafe))
        self.assertFalse(self.catalog.candidates("до 150"))

    def model_options(self, query):
        rows = self.catalog.model_candidates(query, self.catalog.parse(query))
        self.assert_evidence(query, rows)
        return {(row["attr_id"], row["value_id"], row["evidence"]) for row in rows}

    def test_expanded_model_linking_is_opt_in_and_can_infer_a_known_make(self):
        query = "🚘 Соренто без ДТП"
        self.assertFalse(self.catalog.parse(query)["filters"])
        self.assertFalse(self.catalog.candidates(query))
        self.assertEqual(self.model_options(query), {
            ("110000", "6", "Соренто"), ("110001", "15", "Соренто"),
        })

    def test_model_linking_uses_known_make_context_without_breaking_catalog_id_ties(self):
        self.assertEqual(self.model_options("Hyundai Solaris"), {("110001", "14", "Solaris")})
        self.assertEqual(self.model_options("Киа Sorento"), {("110001", "15", "Sorento")})
        self.assertFalse(self.model_options("OMODA C5"))

    def test_model_linking_handles_code_notation_without_reinterpreting_numbers(self):
        cases = {
            "Volvo хс 60, 2020": {("110001", "16", "хс 60")},
            "Mercedes GL-350": {("110001", "19", "GL-350")},
            "Volvo 60 км": set(),
            "2008": set(),
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.model_options(query), expected)

    def test_model_linking_scopes_part_descriptors_and_negative_or_alternative_mentions(self):
        self.assertEqual(self.model_options("передний бампер на Киа Соренто"), {("110001", "15", "Соренто")})
        self.assertEqual(self.model_options("без ДТП, Киа Соренто, только не автомат"), {("110001", "15", "Соренто")})
        for query in ("не Соренто", "Соренто или Camry", "кроме GL350", "передний бампер"):
            with self.subTest(query=query):
                self.assertFalse(self.model_options(query))


    def test_model_linking_never_turns_conflicts_into_positive_options(self):
        for query in ("Toyota Honda", "Toyota Honda омода", "коробка автомат, механика"):
            with self.subTest(query=query):
                baseline = self.catalog.parse(query)
                self.assertTrue(any(row["reason"] == "conflict" for row in baseline["unresolved"]))
                self.assertFalse(self.model_options(query))

    def v3_options(self, query):
        rows = self.catalog.model_candidates_v3(query, self.catalog.parse(query))
        self.assert_evidence(query, rows)
        return {
            (row["attr_id"], row.get("value_id", (row.get("min"), row.get("max"))), row["evidence"])
            for row in rows
        }

    def test_v3_links_glued_make_and_model_with_independent_offsets(self):
        query = "🚙 бмве39 3.0 миханика"
        self.assertFalse(self.model_options(query))
        self.assertEqual(self.v3_options(query), {
            ("110000", "101", "бмв"), ("110001", "111", "е39"),
            ("110012", (3.0, 3.0), "3.0"), ("110008", "61", "миханика"),
        })
        self.assertEqual(self.v3_options("тойтавенза"), {
            ("110000", "1", "тойта"), ("110001", "112", "венза"),
        })
        self.assertFalse(self.v3_options("киавенза"))

    def test_v3_conditions_short_aliases_on_catalog_make(self):
        self.assertEqual(self.v3_options("Ferrari F12"), {("110001", "113", "F12")})
        self.assertEqual(self.v3_options("Mercedes с 180"), {("110001", "114", "с 180")})
        self.assertEqual(self.v3_options("Газель Next"), {("110000", "103", "Газель")})
        for query in ("F12", "Honda F12", "с 180", "OMODA C5", "Ferrari C5 F12", "Mercedes с 180 км"):
            with self.subTest(query=query):
                self.assertFalse(self.v3_options(query))

    def test_v3_requires_an_identified_model_for_unmarked_numeric_context(self):
        self.assertIn(("110012", (2.5, 2.5), "2,5"), self.v3_options("Toyota Camry 2,5 автомат"))
        self.assertIn(("110012", (1.6, 1.6), "1'6"), self.v3_options("Toyota Camry 2013 1'6"))
        self.assertIn(("164669", (2008, 2008), "2008"), self.v3_options("Toyota Land Cruiser 2008"))
        for query in (
            "Toyota 1.6", "Camry до 1.6", "Camry 1.6 млн", "Camry 1.6.2013",
            "Camry 1.6/2.0", "Camry 1.6 до 2.0", "Camry 1.6 кВт", "Camry 1.6 м", "Camry\n1.6",
            "2008", "Peugeot 3008 2008",
        ):
            with self.subTest(query=query):
                self.assertFalse(any(attr in {"110012", "164669"} for attr, _, _ in self.v3_options(query)))

    def test_v3_preserves_negation_alternatives_and_conflicts(self):
        for query in (
            "не бмве39", "кроме Газель", "Ferrari F12 или Toyota", "BMW, Toyota, бмве39",
            "Toyota, Ferrari F12", "Camry без двигателя 1.6", "запчасти на Camry 1.6",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.v3_options(query), {
                    (attr, value_id, evidence) for attr, value_id, evidence in self.model_options(query)
                })


if __name__ == "__main__":
    unittest.main()
