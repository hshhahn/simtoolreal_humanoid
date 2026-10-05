"""Checkpoint aliases must work on the first save and subsequent saves."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rl_games.algos_torch import torch_ext


class CheckpointSymlinkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "nn" / "checkpoint.pth"
        self.target.parent.mkdir()
        self.target.write_bytes(b"checkpoint contents")
        self.link = self.root / "best" / "model.pth"
        self.link.parent.mkdir()
        self.relative_target = os.path.relpath(self.target, self.link.parent)

    def replace_without_retry(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(
            torch_ext.time, "sleep"
        ) as sleep:
            torch_ext.safe_symlink(self.relative_target, self.link)
        sleep.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(self.link.is_symlink())
        self.assertEqual(os.readlink(self.link), self.relative_target)
        self.assertEqual(self.link.read_bytes(), b"checkpoint contents")

    def test_first_checkpoint_creates_alias_without_retry(self):
        self.replace_without_retry()

    def test_existing_alias_is_replaced_without_deleting_old_checkpoint(self):
        old_target = self.target.with_name("previous.pth")
        old_target.write_bytes(b"previous checkpoint")
        self.link.symlink_to(os.path.relpath(old_target, self.link.parent))
        self.replace_without_retry()
        self.assertEqual(old_target.read_bytes(), b"previous checkpoint")

    def test_dangling_alias_is_replaced_without_retry(self):
        self.link.symlink_to("missing.pth")
        self.assertFalse(self.link.exists())
        self.replace_without_retry()

    def test_existing_regular_file_is_replaced_with_alias(self):
        self.link.write_bytes(b"previous alias contents")
        self.replace_without_retry()

    def test_real_unlink_error_retries_and_prevents_creating_alias(self):
        self.link.symlink_to(self.relative_target)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            torch_ext.os, "remove", side_effect=PermissionError("unlink denied")
        ) as remove, mock.patch.object(torch_ext.time, "sleep") as sleep, mock.patch.object(
            torch_ext.os, "symlink"
        ) as symlink:
            with self.assertRaises(RuntimeError):
                torch_ext.safe_symlink(self.relative_target, self.link)
        self.assertEqual(remove.call_count, 5)
        self.assertEqual(sleep.call_args_list, [mock.call(n) for n in (1, 2, 4, 8, 16)])
        symlink.assert_not_called()
        self.assertEqual(self.link.read_bytes(), b"checkpoint contents")


if __name__ == "__main__":
    unittest.main()
