"""Offline checks. Run: python -m unittest test_pipeline -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from features import lexical_features, LEXICAL_NAMES, domain_of, feature_matrix
from common import load_data, split_data, fit_vocab, encode_urls, evaluate


class PipelineTests(unittest.TestCase):
    def test_url_features(self):
        values = dict(zip(LEXICAL_NAMES, lexical_features('https://a.example.co.uk/LoGin?id=123')))
        self.assertEqual(values['dot_count'], 3)
        self.assertEqual(values['query_param_count'], 1)
        self.assertEqual(values['subdomain_count'], 1)
        self.assertEqual(values['keyword_login'], 1)
        self.assertEqual(domain_of('https://a.example.co.uk/x'), 'example.co.uk')
        self.assertEqual(domain_of('https://b.example.co.uk/y'), 'example.co.uk')
        self.assertTrue(np.isfinite(lexical_features('http://127.0.0.1/a')).all())

    def test_url_only_features(self):
        x, names = feature_matrix(['https://example.com', 'http://127.0.0.1/login'])
        self.assertEqual(x.shape, (2, 41))
        self.assertEqual(names, LEXICAL_NAMES)
        self.assertTrue(np.isfinite(x).all())

    def test_vocab_oov_and_case(self):
        vocab = fit_vocab(['aA'])
        encoded = encode_urls(['aAz'], vocab, 5)
        self.assertNotEqual(encoded[0, 0], encoded[0, 1])
        self.assertEqual(encoded[0, 2], 1)
        self.assertTrue((encoded[0, 3:] == 0).all())

    def test_cleaning_and_group_split(self):
        records = [{'url': f'https://h{j}.domain{i}.com/path', 'label': i % 2}
                   for i in range(60) for j in range(2)]
        records += [records[0], {'url': 'http://conflict.com', 'label': 0},
                    {'url': 'http://conflict.com', 'label': 1}, {'url': '', 'label': 0}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.csv'
            pd.DataFrame(records).to_csv(path, index=False)
            df, report = load_data(path)
        self.assertEqual(len(df), 120)
        self.assertEqual(report['conflicting_rows_removed'], 2)
        splits = split_data(df, 'domain', 42)
        groups = [set(df.iloc[ix].domain) for ix in splits.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, splits.values())), len(df))

    def test_positive_label(self):
        m = evaluate([0, 0, 1, 1], [.1, .8, .9, .2], .5)
        self.assertEqual((m['tn'], m['fp'], m['fn'], m['tp']), (1, 1, 1, 1))

    def test_network_roundtrip(self):
        import tensorflow as tf
        from model import build_model, get_extractor
        tf.keras.utils.set_random_seed(42)
        model = build_model(12, max_len=12, embedding_dim=4, filters=4, lstm_units=4, deep_dim=3)
        x = np.array([[2, 3, 4] + [0] * 9, [5, 6] + [0] * 10], dtype=np.int32)
        model.train_on_batch(x, np.array([0, 1], dtype=np.float32))
        expected = get_extractor(model).predict(x, verbose=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'network.keras'
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            actual = get_extractor(loaded).predict(x, verbose=0)
        np.testing.assert_allclose(expected, actual, atol=1e-6)
        self.assertEqual(expected.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
