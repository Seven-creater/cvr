from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.audio_cvr_weak_accept import (
    AUDIT_GATES,
    EXTRA_AUDIT_QUOTAS,
    _safe_embedding_name,
    assemble_imagebind_variant_cache,
    assemble_e5_variant_cache,
    generate_reference_variants,
    prepare_human_audit,
    summarize_human_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AudioCVRWeakAcceptTests(unittest.TestCase):
    def _full_fixture(self, root: Path) -> tuple[Path, Path, list[dict]]:
        datasets = (
            ["avatar"] * 500
            + ["vggsound"] * 250
            + ["AVE-Dataset"] * 180
            + ["worldsense"] * 40
            + ["VGG-MonoAudio"] * 30
        )
        rows = []
        for index, dataset in enumerate(datasets):
            reference = root / f"reference_{index}.mp4"
            target = root / f"target_{index}.mp4"
            reference.write_bytes(f"ref-{index}".encode())
            target.write_bytes(f"target-{index}".encode())
            rows.append(
                {
                    "sample_id": f"sample_{index:04d}",
                    "proposal_id": f"proposal_{index:04d}",
                    "reference_video": reference.name,
                    "target_video": target.name,
                    "edit_text": f"change sound {index}",
                    "dataset": dataset,
                    "decision": "pass",
                }
            )
        full = root / "full.jsonl"
        core = root / "core.jsonl"
        _write_jsonl(full, rows)
        _write_jsonl(core, rows[:150])
        return full, core, rows

    def test_human_audit_is_blind_stratified_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full, core, _ = self._full_fixture(root)
            output = root / "audit"
            summary = prepare_human_audit(
                full_path=full,
                core_path=core,
                output_dir=output,
                media_roots=[root],
                expected_full_sha256="",
            )
            self.assertEqual(200, summary["unique_sample_count"])
            self.assertEqual(20, summary["hidden_repeat_count"])
            private = [
                json.loads(line)
                for line in (output / "private_manifest.jsonl").read_text().splitlines()
            ]
            public = [
                json.loads(line)
                for line in (output / "public_manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(220, len(private))
            self.assertEqual(220, len(public))
            self.assertNotIn("sample_id", public[0])
            self.assertNotIn("dataset", public[0])
            primaries = [row for row in private if not row["is_hidden_repeat"]]
            supplements = [
                row for row in primaries if row["audit_partition"] == "supplement50"
            ]
            counts = {
                dataset: sum(row["dataset"] == dataset for row in supplements)
                for dataset in EXTRA_AUDIT_QUOTAS
            }
            self.assertEqual(EXTRA_AUDIT_QUOTAS, counts)
            summary = json.loads(
                (output / "audit_manifest_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                EXTRA_AUDIT_QUOTAS, summary["supplement_requested_quotas"]
            )
            self.assertEqual(
                EXTRA_AUDIT_QUOTAS, summary["supplement_realized_quotas"]
            )
            self.assertEqual(30, sum(row["requires_variant_check"] for row in primaries))

    def test_human_summary_reports_intrarater_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full, core, _ = self._full_fixture(root)
            audit = root / "audit"
            prepare_human_audit(
                full_path=full,
                core_path=core,
                output_dir=audit,
                media_roots=[root],
                expected_full_sha256="",
            )
            manifest = [
                json.loads(line)
                for line in (audit / "private_manifest.jsonl").read_text().splitlines()
            ]
            responses = []
            for row in manifest:
                response = {
                    "review_id": row["review_id"],
                    "confidence": 4,
                    "note": "",
                    **{gate: True for gate in AUDIT_GATES},
                }
                if row["requires_variant_check"]:
                    response["temporal_preserves_pre_edit"] = True
                    response["spatial_preserves_pre_edit"] = True
                responses.append(response)
            _write_jsonl(audit / "responses.jsonl", responses)
            report = summarize_human_audit(audit, root / "summary")
            self.assertEqual(1.0, report["core150"]["valid_rate"])
            self.assertEqual(1.0, report["hidden_repeat"]["exact_all_gate_agreement"])
            self.assertEqual(30, report["variant_semantics"]["count"])

    def test_partial_human_summary_reports_actual_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full, core, _ = self._full_fixture(root)
            audit = root / "audit"
            prepare_human_audit(
                full_path=full,
                core_path=core,
                output_dir=audit,
                media_roots=[root],
                expected_full_sha256="",
            )
            manifest = [
                json.loads(line)
                for line in (audit / "private_manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            responses = [
                {
                    "review_id": row["review_id"],
                    "confidence": 4,
                    "note": "",
                    **{gate: True for gate in AUDIT_GATES},
                }
                for row in manifest[:10]
            ]
            _write_jsonl(audit / "responses.jsonl", responses)
            with self.assertRaisesRegex(ValueError, "human audit incomplete"):
                summarize_human_audit(audit, root / "strict_summary")
            report = summarize_human_audit(
                audit, root / "partial_summary", allow_partial=True
            )
            self.assertTrue(report["partial_audit"])
            self.assertEqual(10, report["completed_display_item_count"])
            self.assertEqual(len(manifest), report["planned_display_item_count"])

    def test_reference_variant_generation_is_item_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "reference.mp4"
            source.write_bytes(b"source")
            output = root / "variant.mp4"
            item = root / "item.json"
            plan = root / "plan.jsonl"
            _write_jsonl(
                plan,
                [
                    {
                        "sample_id": "sample",
                        "condition": "transcoded",
                        "source_path": str(source),
                        "output_path": str(output),
                        "item_path": str(item),
                    }
                ],
            )

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"variant")
                return mock.Mock(returncode=0, stdout="", stderr="")

            probe = {"duration": 6.0, "width": 320, "height": 240, "has_audio": True}
            with mock.patch(
                "app.audio_cvr_weak_accept._probe_media", return_value=probe
            ), mock.patch(
                "app.audio_cvr_weak_accept.subprocess.run", side_effect=fake_run
            ):
                first = generate_reference_variants(
                    plan_path=plan, shard_index=0, shard_count=1, retries=2
                )
                second = generate_reference_variants(
                    plan_path=plan, shard_index=0, shard_count=1, retries=2
                )
            self.assertEqual(1, first["generated_count"])
            self.assertEqual(1, second["reused_count"])
            self.assertTrue(item.is_file())

    def test_e5_variant_cache_replaces_only_reference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            count = 1000
            dimension = 4
            exact = root / "exact"
            exact.mkdir()
            sample_ids = np.asarray([f"sample_{index:04d}" for index in range(count)])
            target = np.zeros((count, dimension), dtype=np.float32)
            reference = np.ones((count, dimension), dtype=np.float32)
            gallery = np.concatenate([target, reference], axis=0)
            np.savez_compressed(
                exact / "eval_embeddings.npz",
                sample_ids=sample_ids,
                query=np.full((count, dimension), 2, dtype=np.float32),
                target=target,
                reference=reference,
                gallery=gallery,
                reference_gallery_index=np.arange(count, count * 2),
            )
            manifest_rows = []
            embedding_root = root / "embeddings"
            for sample_id in sample_ids:
                manifest_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "condition": "spatial",
                        "output_path": "unused.mp4",
                    }
                )
                path = (
                    embedding_root
                    / "off"
                    / "items"
                    / "spatial"
                    / f"{_safe_embedding_name(str(sample_id))}.npy"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.full(dimension, 3, dtype=np.float32))
            manifest = root / "manifest.jsonl"
            _write_jsonl(manifest, manifest_rows)

            output = root / "variant_cache"
            audit = assemble_e5_variant_cache(
                exact_cache_dir=exact,
                variant_manifest=manifest,
                variant_embedding_root=embedding_root,
                video_audio_mode="off",
                condition="spatial",
                output_dir=output,
            )
            self.assertTrue(audit["non_reference_embeddings_bitwise_identical"])
            with np.load(output / "eval_embeddings.npz") as data:
                np.testing.assert_array_equal(data["gallery"][:count], target)
                np.testing.assert_array_equal(
                    data["gallery"][count:],
                    np.full((count, dimension), 3, dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    data["query"], np.full((count, dimension), 2, dtype=np.float32)
                )

    def test_imagebind_variant_cache_replaces_only_reference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            count = 1000
            dimension = 8
            exact = root / "exact"
            exact.mkdir()
            sample_ids = np.asarray([f"sample_{index:04d}" for index in range(count)])
            reference_indices = np.arange(count, count * 2)
            vision = np.zeros((count * 2, dimension), dtype=np.float32)
            audio = np.zeros((count * 2, dimension), dtype=np.float32)
            vision[count:] = 1
            audio[count:] = 2
            np.savez_compressed(
                exact / "imagebind_embeddings.npz",
                sample_ids=sample_ids,
                reference_indices=reference_indices,
                gallery_vision=vision,
                gallery_audio=audio,
                query_vision=np.zeros((count, dimension), dtype=np.float32),
                query_audio=np.zeros((count, dimension), dtype=np.float32),
                query_text=np.zeros((count, dimension), dtype=np.float32),
            )
            inventory_rows = []
            cache = root / "cache"
            for index, sample_id in enumerate(sample_ids):
                media_id = f"reference_variant::temporal::{sample_id}"
                inventory_rows.append(
                    {
                        "media_id": media_id,
                        "sample_id": str(sample_id),
                        "condition": "temporal",
                    }
                )
            inventory = root / "inventory.jsonl"
            _write_jsonl(inventory, inventory_rows)
            output = root / "variant"
            with mock.patch(
                "app.audio_cvr_external_baseline._load_media_embedding",
                return_value=(
                    np.full(dimension, 3, dtype=np.float32),
                    np.full(dimension, 4, dtype=np.float32),
                ),
            ):
                audit = assemble_imagebind_variant_cache(
                    exact_assembly_dir=exact,
                    variant_inventory=inventory,
                    cache_root=cache,
                    condition="temporal",
                    output_dir=output,
                )
            self.assertTrue(audit["non_reference_vision_bitwise_identical"])
            with np.load(output / "imagebind_embeddings.npz") as data:
                np.testing.assert_array_equal(data["gallery_vision"][:count], 0)
                np.testing.assert_array_equal(data["gallery_audio"][:count], 0)


if __name__ == "__main__":
    unittest.main()
