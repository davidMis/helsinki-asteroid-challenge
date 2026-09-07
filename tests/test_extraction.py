"""Verify the preserved numerical source against its recorded original ASTs."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExtractionTests(unittest.TestCase):
    def test_preserved_symbols(self):
        records = json.loads((ROOT / "provenance/source_extraction.json").read_text())["files"]
        self.assertGreater(len(records), 0)
        for record in records:
            path = ROOT / record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["extracted_sha256"], str(path)
            )
            tree = ast.parse(path.read_text())
            actual = {
                hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                for node in tree.body
            }
            for symbol in record["unchanged_symbols"]:
                self.assertIn(symbol["ast_sha256"], actual, str(symbol["symbols"]))


if __name__ == "__main__":
    unittest.main()
