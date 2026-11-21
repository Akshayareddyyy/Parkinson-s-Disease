"""
Unit tests for scientific paper generation utilities, LaTeX conversions,
citation ordering, and document rendering in code/paperlib.py.
"""

import sys
import os
import unittest
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import paperlib as PL


class TestPaperGeneration(unittest.TestCase):

    def test_tex_text_formatting(self):
        """Verifies markdown bold, italic, and citation conversion to LaTeX."""
        cites = {"ref1": 1, "ref2": 2}
        src = "This is **bold** and *italic* with [@ref1; @ref2]."
        out = PL.tex_text(src, cites)

        self.assertIn(r"\textbf{bold}", out)
        self.assertIn(r"\textit{italic}", out)
        self.assertIn(r"\cite{ref1,ref2}", out)

    def test_order_cites_first_appearance(self):
        """Verifies order_cites extracts citation keys in order of appearance without duplicates."""
        blocks = [
            {"t": "p", "text": "Prior work [@little2009; @tsanas2010] established baseline."},
            {"t": "p", "text": "Federated learning was introduced in [@mcmahan2017; @little2009]."},
            {"t": "p", "text": "Quantum circuits were proposed by [@havlicek2019]."}
        ]
        order = PL.order_cites(blocks)

        expected = ["little2009", "tsanas2010", "mcmahan2017", "havlicek2019"]
        self.assertEqual(order, expected)

    def test_cite_numbers_substitution(self):
        """Verifies cite_numbers maps citation keys to sorted 1-based bracketed indices."""
        order = ["little2009", "tsanas2010", "mcmahan2017"]
        text = "Methods follow [@tsanas2010; @little2009]."
        out = PL.cite_numbers(text, order)

        # little2009 is 1, tsanas2010 is 2 -> sorted [1, 2]
        self.assertIn("[1, 2]", out)

    def test_render_tex_document_structure(self):
        """Verifies render_tex produces valid IEEEtran LaTeX document structure."""
        meta = {
            "title": "Parkinson Telemonitoring Study",
            "authors": "A. Researcher",
            "affil": "Dept of CS",
            "abstract": "This study evaluates telemonitoring.",
            "keywords": "Parkinson Disease, Telemonitoring, Machine Learning"
        }
        blocks = [
            {"t": "h1", "text": "Introduction"},
            {"t": "p", "text": "Parkinson disease progression telemonitoring [@little2009]."}
        ]
        refs = {
            "little2009": "M. A. Little et al., IEEE TBME, 2009."
        }
        order = ["little2009"]

        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tf:
            temp_path = tf.name

        try:
            PL.render_tex(meta, blocks, refs, order, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(r"\documentclass[conference,a4paper]{IEEEtran}", content)
            self.assertIn(meta["title"], content)
            self.assertIn(r"\begin{document}", content)
            self.assertIn(r"\maketitle", content)
            self.assertIn(r"\begin{abstract}", content)
            self.assertIn(r"\section{Introduction}", content)
            self.assertIn(r"\end{document}", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
