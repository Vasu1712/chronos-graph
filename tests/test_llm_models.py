import numpy as np
import pytest

from chronos.llm.embed import BGE_QUERY_INSTRUCTION, Embedder, from_blob, to_blob, top_k
from chronos.llm.nli import NLI


class FakeST:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, **kw):
        self.calls.append((list(texts), batch_size))
        v = np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True) if normalize_embeddings else v


class FakeCrossEncoder:
    def predict(self, pairs, batch_size, **kw):
        # label order: contradiction, entailment, neutral
        return np.array([[0.0, 5.0, 0.0] if h in p else [5.0, 0.0, 0.0] for p, h in pairs])


def test_embedder_normalizes_and_prefixes_queries():
    st = FakeST()
    emb = Embedder(model=st)
    vecs = emb.encode_passages(["a", "bbb"])
    assert vecs.dtype == np.float32 and vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    emb.encode_queries(["where?"])
    assert st.calls[-1][0] == [BGE_QUERY_INSTRUCTION + "where?"]
    assert emb.encode([]).shape[0] == 0


def test_blob_roundtrip_and_topk():
    m = np.eye(4, dtype=np.float32)
    assert np.array_equal(from_blob(to_blob(m[2])), m[2])
    q = np.array([0.1, 0.9, 0.0, 0.2], dtype=np.float32)
    assert [i for i, _ in top_k(q, m, 2)] == [1, 3]
    assert top_k(q, m[:0], 3) == []


def test_nli_scores_softmax_and_label_mapping():
    nli = NLI(model=FakeCrossEncoder(), labels=("contradiction", "entailment", "neutral"))
    s_yes, s_no = nli.score([("Caroline lives in Mumbai.", "lives in Mumbai"), ("x", "y")])
    assert s_yes.label == "ENTAILMENT" and s_yes.entailment > 0.9
    assert s_no.label == "CONTRADICTION"
    assert s_yes.entailment + s_yes.neutral + s_yes.contradiction == pytest.approx(1.0)
    assert nli.entails("Caroline lives in Mumbai.", "lives in Mumbai")
    assert nli.score([]) == []
