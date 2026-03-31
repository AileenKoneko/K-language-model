from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from k_language_model.data import ByteTokenizer, load_dataset


class ByteTokenizerTests(unittest.TestCase):
    def test_encode_decode_round_trip_for_utf8_text(self) -> None:
        tokenizer = ByteTokenizer()
        text = "A\u0105\U0001F642\n"

        encoded = tokenizer.encode(text)

        self.assertEqual(encoded, list(text.encode("utf-8")))
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_decode_invalid_utf8_bytes_uses_visible_escape_sequences(self) -> None:
        tokenizer = ByteTokenizer()

        decoded = tokenizer.decode([65, 255, 10])

        self.assertEqual(decoded, "A\\xff\n")

    def test_load_dataset_byte_tokenizer_preserves_raw_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.bin"
            val_path = root / "val.bin"
            train_bytes = b"A\r\nB\xff"
            val_bytes = b"\x00C\r\n"
            train_path.write_bytes(train_bytes)
            val_path.write_bytes(val_bytes)

            train_data, val_data, tokenizer = load_dataset(
                dataset="shakespeare",
                data_path=str(train_path),
                val_path=str(val_path),
                tokenizer_type="byte",
            )

        self.assertEqual(tokenizer.tokenizer_type, "byte")
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(train_data.tolist(), list(train_bytes))
        self.assertEqual(val_data.tolist(), list(val_bytes))

    def test_load_dataset_full_shakespeare_merges_raw_directory_and_splits_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "b.txt").write_bytes(b"BBB")
            (raw_dir / "a.txt").write_bytes(b"AA\n")
            (raw_dir / "c.txt").write_bytes(b"C")

            train_data, val_data, tokenizer = load_dataset(
                dataset="full-shakespeare",
                data_path=str(raw_dir),
                tokenizer_type="byte",
                val_frac=0.25,
            )

            merged_path = root / "raw.merged.txt"
            merged_bytes = merged_path.read_bytes()

        self.assertEqual(tokenizer.tokenizer_type, "byte")
        self.assertEqual(merged_bytes, b"AA\nBBB\nC")
        self.assertEqual(train_data.tolist(), list(b"AA\nBBB"))
        self.assertEqual(val_data.tolist(), list(b"\nC"))

    def test_load_dataset_full_shakespeare_clean_strips_folger_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "play.txt").write_text(
                "\n".join(
                    [
                        "Sample Play",
                        "by William Shakespeare",
                        "Edited by Somebody",
                        "Folger Shakespeare Library",
                        "https://example.test",
                        "Created on Jul 31, 2015, from FDT version 0.9.2",
                        "",
                        "Characters in the Play",
                        "======================",
                        "PAGE",
                        "KING",
                        "",
                        "ACT 1",
                        "=====",
                        "Scene 1",
                        "=======",
                        "KING",
                        "Speak the speech.",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "sonnets.txt").write_text(
                "\n".join(
                    [
                        "Sonnets",
                        "by William Shakespeare",
                        "Edited by Somebody",
                        "Folger Shakespeare Library",
                        "https://example.test",
                        "Created on Jul 31, 2015, from FDT version 0.9.0.1",
                        "",
                        "1",
                        "",
                        "From fairest creatures we desire increase,",
                    ]
                ),
                encoding="utf-8",
            )

            train_data, val_data, tokenizer = load_dataset(
                dataset="full-shakespeare-clean",
                data_path=str(raw_dir),
                tokenizer_type="char",
                val_frac=0.25,
            )

            cleaned_path = root / "raw.cleaned.v1.txt"
            cleaned_text = cleaned_path.read_text(encoding="utf-8")

        self.assertEqual(tokenizer.tokenizer_type, "char")
        self.assertTrue(len(train_data) > 0)
        self.assertTrue(len(val_data) > 0)
        self.assertNotIn("by William Shakespeare", cleaned_text)
        self.assertNotIn("Characters in the Play", cleaned_text)
        self.assertNotIn("PAGE", cleaned_text)
        self.assertIn("ACT 1", cleaned_text)
        self.assertIn("Speak the speech.", cleaned_text)
        self.assertIn("From fairest creatures we desire increase,", cleaned_text)


if __name__ == "__main__":
    unittest.main()
