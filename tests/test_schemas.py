import unittest

from app.schemas import QueryCase, RetrievalParams


class SchemaTests(unittest.TestCase):
    def test_retrieval_params_are_normalized(self) -> None:
        params = RetrievalParams(video_weight=8.0, audio_weight=2.0, topk=3)
        self.assertAlmostEqual(params.video_weight, 0.8)
        self.assertAlmostEqual(params.audio_weight, 0.2)
        self.assertEqual(params.topk, 3)

    def test_query_case_from_dict(self) -> None:
        payload = {
            "query_id": "q1",
            "source_video_id": "s1",
            "edit_instruction": "make it louder",
            "target_video_id": "t1",
            "preserve_tags": ["music"],
            "required_audio_tags": ["cheering"],
        }
        query = QueryCase.from_dict(payload)
        self.assertEqual(query.query_id, "q1")
        self.assertEqual(query.required_audio_tags, ["cheering"])


if __name__ == "__main__":
    unittest.main()

