[1mdiff --git a/app/backends/base.py b/app/backends/base.py[m
[1mindex af3bd95..2360c42 100644[m
[1m--- a/app/backends/base.py[m
[1m+++ b/app/backends/base.py[m
[36m@@ -9,10 +9,22 @@[m [mfrom typing import Any[m
 from app.schemas import CandidateMetadata, CompareResult, InspectionRecord, QueryCase, RetrievalCandidate, RetrievalParams[m
 [m
 TOKEN_RE = re.compile(r"[a-z0-9]+")[m
[32m+[m[32mAPP_ROOT = Path(__file__).resolve().parent.parent[m
[32m+[m[32mPROJECT_ROOT = APP_ROOT.parent[m
[32m+[m
[32m+[m
[32m+[m[32mdef resolve_data_path(path: str | Path) -> Path:[m
[32m+[m[32m    candidate = Path(path)[m
[32m+[m[32m    if candidate.is_absolute():[m
[32m+[m[32m        return candidate[m
[32m+[m[32m    direct = (PROJECT_ROOT / candidate).resolve()[m
[32m+[m[32m    if direct.exists():[m
[32m+[m[32m        return direct[m
[32m+[m[32m    return (APP_ROOT / candidate).resolve()[m
 [m
 [m
 def load_json(path: str | Path) -> Any:[m
[31m-    with Path(path).open("r", encoding="utf-8") as handle:[m
[32m+[m[32m    with resolve_data_path(path).open("r", encoding="utf-8") as handle:[m
         return json.load(handle)[m
 [m
 [m
[36m@@ -129,4 +141,3 @@[m [mclass RetrievalBackend(ABC):[m
         source = self.get_candidate(query.source_video_id)[m
         candidate = self.get_candidate(candidate_id)[m
         return heuristic_compare(query, source, candidate)[m
[31m-[m
[1mdiff --git a/app/controller/__init__.py b/app/controller/__init__.py[m
[1mindex 70d4d53..6803b1f 100644[m
[1m--- a/app/controller/__init__.py[m
[1m+++ b/app/controller/__init__.py[m
[36m@@ -1,5 +1,11 @@[m
[31m-from .openai_responses import OpenAIResponsesController[m
 from .scripted import ScriptedController[m
 [m
 __all__ = ["OpenAIResponsesController", "ScriptedController"][m
 [m
[32m+[m
[32m+[m[32mdef __getattr__(name: str):[m
[32m+[m[32m    if name == "OpenAIResponsesController":[m
[32m+[m[32m        from .openai_responses import OpenAIResponsesController[m
[32m+[m
[32m+[m[32m        return OpenAIResponsesController[m
[32m+[m[32m    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")[m
[1mdiff --git a/app/demo.py b/app/demo.py[m
[1mindex 39b104e..b0a4505 100644[m
[1m--- a/app/demo.py[m
[1m+++ b/app/demo.py[m
[36m@@ -1,12 +1,30 @@[m
 from __future__ import annotations[m
 [m
 import argparse[m
[32m+[m[32mimport importlib.util[m
 import os[m
[32m+[m[32mfrom pathlib import Path[m
 [m
 from app.artifacts import append_jsonl, write_run_artifacts[m
 from app.backends import FileRetrievalBackend, MockRetrievalBackend[m
 from app.config import load_yaml[m
[31m-from app.controller import OpenAIResponsesController, ScriptedController[m
[32m+[m[32mfrom app.controller import ScriptedController[m
[32m+[m
[32m+[m
[32m+[m[32mdef resolve_config_path(config_path: str, value: str | None) -> str | None:[m
[32m+[m[32m    if value is None:[m
[32m+[m[32m        return None[m
[32m+[m[32m    path = Path(value)[m
[32m+[m[32m    if path.is_absolute():[m
[32m+[m[32m        return str(path)[m
[32m+[m
[32m+[m[32m    config_dir = Path(config_path).resolve().parent[m
[32m+[m[32m    direct_candidate = (config_dir / path).resolve()[m
[32m+[m[32m    if direct_candidate.exists():[m
[32m+[m[32m        return str(direct_candidate)[m
[32m+[m
[32m+[m[32m    project_candidate = (config_dir.parent / path).resolve()[m
[32m+[m[32m    return str(project_candidate)[m
 [m
 [m
 def build_backend(mode: str, config_path: str | None):[m
[36m@@ -17,15 +35,23 @@[m [mdef build_backend(mode: str, config_path: str | None):[m
             raise ValueError("--config is required in real mode")[m
         config = load_yaml(config_path)[m
         return FileRetrievalBackend([m
[31m-            candidates_path=config["candidates_path"],[m
[31m-            queries_path=config["queries_path"],[m
[31m-            retrieval_scores_path=config.get("retrieval_scores_path"),[m
[32m+[m[32m            candidates_path=resolve_config_path(config_path, config["candidates_path"]),[m
[32m+[m[32m            queries_path=resolve_config_path(config_path, config["queries_path"]),[m
[32m+[m[32m            retrieval_scores_path=resolve_config_path(config_path, config.get("retrieval_scores_path")),[m
         )[m
     raise ValueError(f"unsupported mode: {mode}")[m
 [m
 [m
[32m+[m[32mdef openai_available() -> bool:[m
[32m+[m[32m    return importlib.util.find_spec("openai") is not None[m
[32m+[m
[32m+[m
 def build_controller(planner: str, backend, model: str):[m
     if planner == "openai":[m
[32m+[m[32m        if not openai_available():[m
[32m+[m[32m            raise RuntimeError("planner=openai requires the openai package to be installed")[m
[32m+[m[32m        from app.controller import OpenAIResponsesController[m
[32m+[m
         return OpenAIResponsesController(backend=backend, model=model)[m
     return ScriptedController(backend=backend)[m
 [m
[36m@@ -33,7 +59,9 @@[m [mdef build_controller(planner: str, backend, model: str):[m
 def resolve_default_planner(explicit: str | None) -> str:[m
     if explicit:[m
         return explicit[m
[31m-    return "openai" if os.environ.get("OPENAI_API_KEY") else "scripted"[m
[32m+[m[32m    if os.environ.get("OPENAI_API_KEY") and openai_available():[m
[32m+[m[32m        return "openai"[m
[32m+[m[32m    return "scripted"[m
 [m
 [m
 def parse_args() -> argparse.Namespace:[m
[36m@@ -52,11 +80,7 @@[m [mdef main() -> None:[m
     planner_name = resolve_default_planner(args.planner)[m
     backend = build_backend(args.mode, args.config)[m
     controller = build_controller(planner_name, backend, args.model)[m
[31m-    queries = ([m
[31m-        [backend.get_query(args.query_id)][m
[31m-        if args.query_id[m
[31m-        else backend.list_queries()[m
[31m-    )[m
[32m+[m[32m    queries = [backend.get_query(args.query_id)] if args.query_id else backend.list_queries()[m
 [m
     traces = [][m
     for query in queries:[m
[1mdiff --git a/app/eval.py b/app/eval.py[m
[1mindex 1a1bc19..1e2aa75 100644[m
[1m--- a/app/eval.py[m
[1m+++ b/app/eval.py[m
[36m@@ -2,7 +2,9 @@[m [mfrom __future__ import annotations[m
 [m
 import argparse[m
 import json[m
[32m+[m[32mfrom collections import Counter, defaultdict[m
 from pathlib import Path[m
[32m+[m[32mfrom typing import Any[m
 [m
 [m
 def parse_args() -> argparse.Namespace:[m
[36m@@ -11,14 +13,132 @@[m [mdef parse_args() -> argparse.Namespace:[m
     return parser.parse_args()[m
 [m
 [m
[31m-def main() -> None:[m
[31m-    args = parse_args()[m
[31m-    path = Path(args.input)[m
[31m-    rows = [][m
[32m+[m[32mdef load_rows(path: Path) -> list[dict[str, Any]]:[m
[32m+[m[32m    rows: list[dict[str, Any]] = [][m
     with path.open("r", encoding="utf-8") as handle:[m
         for line in handle:[m
             if line.strip():[m
                 rows.append(json.loads(line))[m
[32m+[m[32m    return rows[m
[32m+[m
[32m+[m
[32m+[m[32mdef query_type(row: dict[str, Any]) -> str:[m
[32m+[m[32m    query = row.get("query", {})[m
[32m+[m[32m    has_audio = bool(query.get("required_audio_tags"))[m
[32m+[m[32m    has_object = bool(query.get("required_objects"))[m
[32m+[m[32m    has_temporal = bool(query.get("required_temporal"))[m
[32m+[m
[32m+[m[32m    active = [[m
[32m+[m[32m        name[m
[32m+[m[32m        for name, flag in ([m
[32m+[m[32m            ("audio", has_audio),[m
[32m+[m[32m            ("object", has_object),[m
[32m+[m[32m            ("temporal", has_temporal),[m
[32m+[m[32m        )[m
[32m+[m[32m        if flag[m
[32m+[m[32m    ][m
[32m+[m[32m    if not active:[m
[32m+[m[32m        return "other"[m
[32m+[m[32m    if len(active) == 1:[m
[32m+[m[32m        return active[0][m
[32m+[m[32m    return "+".join(active)[m
[32m+[m
[32m+[m
[32m+[m[32mdef final_comparison(row: dict[str, Any]) -> dict[str, Any]:[m
[32m+[m[32m    final_candidate_id = row.get("final_candidate_id")[m
[32m+[m[32m    if not final_candidate_id:[m
[32m+[m[32m        return {}[m
[32m+[m
[32m+[m[32m    for round_row in reversed(row.get("rounds", [])):[m
[32m+[m[32m        comparisons = round_row.get("comparisons", {}) or {}[m
[32m+[m[32m        if final_candidate_id in comparisons:[m
[32m+[m[32m            return comparisons[final_candidate_id] or {}[m
[32m+[m[32m    return {}[m
[32m+[m
[32m+[m
[32m+[m[32mdef error_labels(row: dict[str, Any]) -> list[str]:[m
[32m+[m[32m    if row.get("success") is True:[m
[32m+[m[32m        return [][m
[32m+[m
[32m+[m[32m    labels: list[str] = [][m
[32m+[m[32m    comparison = final_comparison(row)[m
[32m+[m[32m    missing = set(comparison.get("missing", []))[m
[32m+[m[32m    conflicts = set(comparison.get("conflicts", []))[m
[32m+[m
[32m+[m[32m    mapping = {[m
[32m+[m[32m        "required-audio": "audio",[m
[32m+[m[32m        "required-object": "object",[m
[32m+[m[32m        "required-temporal": "temporal",[m
[32m+[m[32m        "preserve-scene": "preserve",[m
[32m+[m[32m    }[m
[32m+[m[32m    for key, label in mapping.items():[m
[32m+[m[32m        if key in missing:[m
[32m+[m[32m            labels.append(f"missing_{label}")[m
[32m+[m
[32m+[m[32m    for conflict in sorted(conflicts):[m
[32m+[m[32m        labels.append(f"conflict_{conflict}")[m
[32m+[m
[32m+[m[32m    if not labels:[m
[32m+[m[32m        predicted = row.get("final_candidate_id")[m
[32m+[m[32m        target = row.get("query", {}).get("target_video_id")[m
[32m+[m[32m        if predicted and target and predicted != target:[m
[32m+[m[32m            labels.append("wrong_candidate")[m
[32m+[m[32m        else:[m
[32m+[m[32m            labels.append("unknown")[m
[32m+[m
[32m+[m[32m    return labels[m
[32m+[m
[32m+[m
[32m+[m[32mdef print_type_breakdown(rows: list[dict[str, Any]]) -> None:[m
[32m+[m[32m    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)[m
[32m+[m[32m    for row in rows:[m
[32m+[m[32m        buckets[query_type(row)].append(row)[m
[32m+[m
[32m+[m[32m    print("type_breakdown:")[m
[32m+[m[32m    for bucket in sorted(buckets):[m
[32m+[m[32m        items = buckets[bucket][m
[32m+[m[32m        success = sum(1 for row in items if row.get("success") is True)[m
[32m+[m[32m        print(f"  {bucket}: count={len(items)} success_rate={success / max(1, len(items)):.3f}")[m
[32m+[m
[32m+[m
[32m+[m[32mdef print_error_breakdown(rows: list[dict[str, Any]]) -> None:[m
[32m+[m[32m    failed_rows = [row for row in rows if row.get("success") is not True][m
[32m+[m[32m    counter: Counter[str] = Counter()[m
[32m+[m[32m    for row in failed_rows:[m
[32m+[m[32m        counter.update(error_labels(row))[m
[32m+[m
[32m+[m[32m    print("error_breakdown:")[m
[32m+[m[32m    if not counter:[m
[32m+[m[32m        print("  none")[m
[32m+[m[32m        return[m
[32m+[m
[32m+[m[32m    for label, count in counter.most_common():[m
[32m+[m[32m        print(f"  {label}: {count}")[m
[32m+[m
[32m+[m
[32m+[m[32mdef print_failed_cases(rows: list[dict[str, Any]]) -> None:[m
[32m+[m[32m    failed_rows = [row for row in rows if row.get("success") is not True][m
[32m+[m[32m    print("failed_cases:")[m
[32m+[m[32m    if not failed_rows:[m
[32m+[m[32m        print("  none")[m
[32m+[m[32m        return[m
[32m+[m
[32m+[m[32m    for row in failed_rows:[m
[32m+[m[32m        labels = ",".join(error_labels(row))[m
[32m+[m[32m        print([m
[32m+[m[32m            "  "[m
[32m+[m[32m            f"query={row.get('query', {}).get('query_id')} "[m
[32m+[m[32m            f"type={query_type(row)} "[m
[32m+[m[32m            f"target={row.get('query', {}).get('target_video_id')} "[m
[32m+[m[32m            f"final={row.get('final_candidate_id')} "[m
[32m+[m[32m            f"errors={labels}"[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m
[32m+[m[32mdef main() -> None:[m
[32m+[m[32m    args = parse_args()[m
[32m+[m[32m    path = Path(args.input)[m
[32m+[m[32m    rows = load_rows(path)[m
 [m
     total = len(rows)[m
     successes = sum(1 for row in rows if row.get("success") is True)[m
[36m@@ -37,8 +157,10 @@[m [mdef main() -> None:[m
     print(f"avg_rounds={avg_rounds:.2f}")[m
     print(f"avg_tool_calls={avg_tools:.2f}")[m
     print(f"first_round_top1={first_round_hits / max(1, total):.3f}")[m
[32m+[m[32m    print_type_breakdown(rows)[m
[32m+[m[32m    print_error_breakdown(rows)[m
[32m+[m[32m    print_failed_cases(rows)[m
 [m
 [m
 if __name__ == "__main__":[m
     main()[m
[31m-[m
[1mdiff --git a/tests/test_real_backend.py b/tests/test_real_backend.py[m
[1mindex 85dbd16..63050f8 100644[m
[1m--- a/tests/test_real_backend.py[m
[1m+++ b/tests/test_real_backend.py[m
[36m@@ -12,12 +12,12 @@[m [mclass RealBackendReplayTests(unittest.TestCase):[m
             retrieval_scores_path="data/real_sample/retrieval_scores.json",[m
         )[m
         controller = ScriptedController(backend)[m
[31m-        trace = controller.run("real_q_audio")[m
[31m-        self.assertEqual(trace.final_candidate_id, "real_band_cheer")[m
[31m-        self.assertTrue(trace.success)[m
[31m-        self.assertLessEqual(len(trace.rounds), 3)[m
[32m+[m[32m        for query in backend.list_queries():[m
[32m+[m[32m            trace = controller.run(query.query_id)[m
[32m+[m[32m            self.assertEqual(trace.final_candidate_id, query.target_video_id, query.query_id)[m
[32m+[m[32m            self.assertTrue(trace.success, query.query_id)[m
[32m+[m[32m            self.assertLessEqual(len(trace.rounds), 3, query.query_id)[m
 [m
 [m
 if __name__ == "__main__":[m
     unittest.main()[m
[31m-[m
