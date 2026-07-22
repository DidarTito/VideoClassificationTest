from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.checkpoint_artifacts import (
    FIELDS,
    dataset_hint,
    download,
    download_http,
    read_manifest,
    refresh,
    safe_destination,
    write_manifest,
)


class CheckpointArtifactTests(unittest.TestCase):
    def test_dataset_hints_do_not_mix_k400_and_ssv2(self):
        self.assertEqual(
            dataset_hint(Path("timesformer/model_SSv2.pyth")), "ssv2"
        )
        self.assertEqual(
            dataset_hint(Path("videoswin/model_kinetics400.pth")), "kinetics400"
        )
        self.assertEqual(
            dataset_hint(Path("videomae/VideoMAE-ViT-B_1600epoch.pth")),
            "kinetics400-pretraining",
        )

    def test_manifest_destination_cannot_escape_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            with self.assertRaises(ValueError):
                safe_destination("../outside.pth", root, checkpoint_root)
            with self.assertRaises(ValueError):
                safe_destination("checkpoints/../../outside.pth", root, checkpoint_root)

    def test_refresh_hashes_files_and_preserves_reviewed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            artifact = checkpoint_root / "family" / "model_k400.pth"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"checkpoint")
            manifest = checkpoint_root / "manifest.csv"
            old = {field: "" for field in FIELDS}
            old.update(
                {
                    "path": "checkpoints/family/model_k400.pth",
                    "bytes": "1",
                    "sha256": "old",
                    "model_family": "old",
                    "dataset_hint": "old",
                    "artifact_role": "old",
                    "source_url": "https://example.invalid/model.pth",
                    "license": "reviewed-license",
                    "redistribution_status": "upstream-only",
                }
            )
            write_manifest([old], manifest)

            self.assertEqual(refresh(root, checkpoint_root, manifest), 0)
            row = read_manifest(manifest)[0]
            self.assertEqual(row["bytes"], str(len(b"checkpoint")))
            self.assertNotEqual(row["sha256"], "old")
            self.assertEqual(row["dataset_hint"], "kinetics400")
            self.assertEqual(row["source_url"], old["source_url"])
            self.assertEqual(row["license"], old["license"])
            self.assertEqual(
                row["redistribution_status"], old["redistribution_status"]
            )

    def test_dropbox_share_link_is_converted_to_direct_download(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [b"weights", b""]
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "scripts.checkpoint_artifacts.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            destination = Path(temporary) / "model.pyth.part"
            download_http(
                "https://www.dropbox.com/s/token/model.pyth?dl=0", destination
            )
            self.assertEqual(destination.read_bytes(), b"weights")
        requested_url = urlopen.call_args.args[0].full_url
        self.assertIn("dropbox.com/s/token/model.pyth", requested_url)
        self.assertIn("dl=1", requested_url)

    def test_manifest_http_source_can_be_reproduced_and_verified(self):
        payload = b"official checkpoint"
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            manifest = checkpoint_root / "manifest.csv"
            row = {field: "" for field in FIELDS}
            row.update(
                {
                    "path": "checkpoints/family/model.pth",
                    "bytes": str(len(payload)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_url": "https://official.invalid/model.pth",
                    "redistribution_status": "unverified-do-not-redistribute",
                }
            )
            write_manifest([row], manifest)

            def fake_http(_url, output):
                output.write_bytes(payload)

            with mock.patch(
                "scripts.checkpoint_artifacts.download_http", side_effect=fake_http
            ):
                status = download(
                    root=root,
                    checkpoint_root=checkpoint_root,
                    manifest_path=manifest,
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                (checkpoint_root / "family" / "model.pth").read_bytes(), payload
            )

    def test_manifest_drive_id_uses_gdown_path(self):
        payload = b"drive checkpoint"
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            manifest = checkpoint_root / "manifest.csv"
            row = {field: "" for field in FIELDS}
            row.update(
                {
                    "path": "checkpoints/family/model.pth",
                    "bytes": str(len(payload)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "google_drive_id": "official-drive-id",
                    "redistribution_status": "unverified-do-not-redistribute",
                }
            )
            write_manifest([row], manifest)

            def fake_drive(file_id, output):
                self.assertEqual(file_id, "official-drive-id")
                output.write_bytes(payload)

            with mock.patch(
                "scripts.checkpoint_artifacts.download_gdrive",
                side_effect=fake_drive,
            ):
                status = download(
                    root=root,
                    checkpoint_root=checkpoint_root,
                    manifest_path=manifest,
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                (checkpoint_root / "family" / "model.pth").read_bytes(), payload
            )


if __name__ == "__main__":
    unittest.main()
