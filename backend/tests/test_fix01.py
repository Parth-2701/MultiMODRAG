import pytest
from backend.rag.hybrid_retrieval import rrf, tokenize, hybrid_search, K_DENSE

def test_rrf_convexity():
    # rrf([[1,2,3],[3,2,1]]) must give docs 1 and 3 identical scores, and doc 2 slightly lower
    scores = dict(rrf([[1, 2, 3], [3, 2, 1]], k=60))
    assert abs(scores[1] - scores[3]) < 1e-9
    expected_1_3 = (1.0 / 61.0) + (1.0 / 63.0)
    expected_2 = 2.0 / 62.0
    assert abs(scores[1] - expected_1_3) < 1e-6
    assert abs(scores[2] - expected_2) < 1e-6
    assert scores[1] > scores[2]

def test_rrf_ties():
    # rrf([[1,2],[3,4]]) must rank 1 and 3 first (tied)
    result = rrf([[1, 2], [3, 4]], k=60)
    top_docs = {result[0][0], result[1][0]}
    assert top_docs == {1, 3}
    assert abs(result[0][1] - result[1][1]) < 1e-9

def test_tokenize():
    # tokenize("What does the cat do in the car with a GPU?") returns ["cat","car","gpu"]
    # (stop words removed, nothing dropped by length)
    tokens = tokenize("What does the cat do in the car with a GPU?")
    assert tokens == ["cat", "car", "gpu"]

def test_sparse_surfaces_rare_keyword_over_dense():
    class MockFrames:
        def __init__(self, data):
            self.data = data
            self.combined_caption = self
            self.title = "demo.mp4"
            self.pos = "pos"
            self.video = "video"
            self.limit_val = None

        def __ne__(self, other):
            return True

        def __eq__(self, other):
            return True

        def __and__(self, other):
            return self

        def similarity(self, query):
            return "sim"

        def where(self, flt):
            return self

        def order_by(self, sim, asc=False):
            return self

        def limit(self, n):
            self.limit_val = n
            return self

        def select(self, *cols):
            return self

        def collect(self):
            return self.data

    mock_data = [
        {"pos": i, "combined_caption": f"Generic video frame {i} about nature", "video": "v.mp4", "title": "demo.mp4"}
        for i in range(50)
    ]
    mock_data[45]["combined_caption"] = "A special experiment on quantum computing in a lab"

    mf = MockFrames(mock_data)
    results = hybrid_search(mf, "demo.mp4", "quantum", fps=1.0)
    
    # 3. Rare keyword present in exactly one caption returns in top 3
    top_3_pos = [r["pos"] for r in results[:3]]
    assert 45 in top_3_pos

    # 4. Dense query limit is bounded to K_DENSE, not all 50 frames
    assert mf.limit_val == K_DENSE
    assert mf.limit_val == 40
