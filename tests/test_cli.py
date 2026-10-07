# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 JoshKirk800
"""Offline tests: scan id extraction, format choice, settings sheet. Run: python -m unittest discover -s tests"""
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import acquire_scan  # noqa: E402
import exporters  # noqa: E402
import insole  # noqa: E402

UID = "00000000-1111-4222-8333-444444444444"   # dummy id, not a real scan


class ScanId(unittest.TestCase):
    def test_fleet_feet_link_with_scan_param(self):
        link = f"https://www.fleetfeet.com/fit-id/scan?store=1&scanned=2&scan={UID}&utm_source=fleetfeet"
        self.assertEqual(acquire_scan.find_scan_id(link), UID)

    def test_scanned_param_is_not_a_scan_id(self):
        link = "https://www.fleetfeet.com/fit-id/scan?store=1&scanned=2&utm_source=fleetfeet&utm_campaign=shoe-scan-basic"
        self.assertIsNone(acquire_scan.find_scan_id(link))

    def test_volumental_link_and_bare_id(self):
        self.assertEqual(acquire_scan.find_scan_id(f"https://my.volumental.com/{UID}/"), UID)
        self.assertEqual(acquire_scan.find_scan_id(UID.upper()), UID)

    def test_url_encoded_and_non_uuid_hex_id(self):
        self.assertEqual(acquire_scan.find_scan_id("https://x.example/?a=1%26scan%3D" + UID), UID)
        self.assertEqual(acquire_scan.find_scan_id("https://x.example/p?scan=0123456789abcdef0123456789abcdef"), "0123456789abcdef0123456789abcdef")

    def test_garbage(self):
        self.assertIsNone(acquire_scan.find_scan_id("hello world"))
        with self.assertRaises(SystemExit):
            acquire_scan.parse_scan_id("hello world")


class Redirects(unittest.TestCase):
    """Fleet Feet emails link through a SendGrid click-tracking redirect to the fit id page with scan=<id>."""

    @staticmethod
    def fake_opener(responses):
        """opener whose open() raises the next canned HTTPError (a redirect) or returns normally for a page."""
        calls = []

        def open_(req, timeout=None):
            calls.append(req.full_url)
            code, loc = responses[len(calls) - 1]
            if code == 200:
                return object()
            raise urllib.error.HTTPError(req.full_url, code, "x", {"Location": loc}, None)
        return mock.Mock(open=open_), calls

    def resolve(self, link, responses):
        opener, calls = self.fake_opener(responses)
        with mock.patch("urllib.request.build_opener", return_value=opener):
            return acquire_scan.resolve_scan_id(link), calls

    def test_follows_sendgrid_redirect_to_scan_param(self):
        target = f"https://www.fleetfeet.com/fit-id/scan?scan={UID}&store=1&scanned=2&utm_source=fleetfeet"
        sid, calls = self.resolve("https://u1.ct.sendgrid.net/ls/click?upn=u001.abc", [(302, target)])
        self.assertEqual(sid, UID)
        self.assertEqual(len(calls), 1)               # stops at the Location: the final page is never loaded

    def test_follows_several_hops_and_relative_locations(self):
        sid, calls = self.resolve("https://a.example/x", [(301, "/y"), (302, f"https://b.example/?scan={UID}")])
        self.assertEqual((sid, calls[1]), (UID, "https://a.example/y"))

    def test_page_without_an_id_gives_none(self):
        self.assertIsNone(self.resolve("https://www.fleetfeet.com/fit-id/scan?store=1&scanned=2", [(200, None)])[0])

    def test_non_redirect_error_and_non_http_text_are_not_followed(self):
        self.assertIsNone(self.resolve("https://a.example/x", [(404, None)])[0])
        with mock.patch("urllib.request.build_opener") as build:
            self.assertIsNone(acquire_scan.resolve_scan_id("hello"))
            build.assert_not_called()

    def test_id_in_the_text_needs_no_network(self):
        with mock.patch("urllib.request.build_opener") as build:
            self.assertEqual(acquire_scan.resolve_scan_id(f"https://x.example/?scan={UID}"), UID)
            build.assert_not_called()


class Choices(unittest.TestCase):
    def test_format_argument(self):
        self.assertEqual(insole.choose_formats("all"), list(exporters.FORMATS))
        self.assertEqual(insole.choose_formats("stl, prusa"), ["stl", "prusa"])
        with self.assertRaises(SystemExit):
            insole.choose_formats("cura")

    def test_menu_default_and_numbers(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(insole.choose_formats(None), ["bambu"])
        with mock.patch("builtins.input", side_effect=["9", "x", "3"]):
            self.assertEqual(insole.choose_formats(None), [list(exporters.FORMATS)[2]])
        n = len(exporters.FORMATS)
        with mock.patch("builtins.input", return_value=str(n + 1)):
            self.assertEqual(insole.choose_formats(None), list(exporters.FORMATS))

    def test_no_input_falls_back_to_default(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(insole.choose_formats(None), ["bambu"])

    def test_link_prompt_retries_then_gives_up(self):
        with mock.patch("builtins.input", side_effect=["nope", "still nope", "no"]), self.assertRaises(SystemExit):
            insole.get_scan_id(None)
        with mock.patch("builtins.input", side_effect=["nope", f"scan={UID}"]):
            self.assertEqual(insole.get_scan_id(None), UID)


class Sheet(unittest.TestCase):
    def test_settings_sheet_lists_every_zone_with_its_density(self):
        design = {"process": {"top_shell_layers": "3", "bottom_shell_layers": "2", "sparse_infill_density": "12%"},
                  "plans": {"left": {"label": "Corrective"}, "right": {"label": "Neutral"}}}
        zones = {"left": [("/x/left_zone_01_30pct.stl", "Heel Rim Medial", 30)], "right": []}
        text = exporters.settings_sheet(design, zones)
        self.assertIn("`left_zone_01_30pct.stl` | Heel Rim Medial | 30% gyroid", text)
        self.assertIn("No infill zones", text)
        self.assertIn("3 top / 2 bottom", text)

    def test_prusa_config_translates_keys_and_supports(self):
        cfg = exporters.prusa_config({"sparse_infill_density": "12%", "top_shell_layers": "3", "bottom_shell_layers": "2",
                                      "enable_support": "1", "support_top_z_distance": "0.3"})
        for line in ("; fill_density = 12%", "; top_solid_layers = 3", "; bottom_solid_layers = 2", "; support_material = 1", "; fill_pattern = gyroid"):
            self.assertIn(line, cfg)
        self.assertNotIn("support_material = 1", exporters.prusa_config({"sparse_infill_density": "12%"}))


if __name__ == "__main__":
    unittest.main()
