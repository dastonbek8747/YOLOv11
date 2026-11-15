"""Repo gigienasi: .env git'da kuzatilmasligi va sir yo'qligini tekshiradi.

SEC-008 regressiya testi. Muammo tuzatilgunga qadar YIQILADI.
"""
import os
import re
import subprocess
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_PATTERNS = [
    r"""\bAKIA[0-9A-Z]{16}\b""",
    r"""\bsk-[A-Za-z0-9]{20,}\b""",
    r"""\bgsk_[A-Za-z0-9]{20,}\b""",
    r"""\bAIza[0-9A-Za-z_\-]{30,}\b""",
    r"""\b\d{8,10}:[A-Za-z0-9_-]{30,}\b""",
]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "tests"}
SCAN_EXT = {".py", ".js", ".ts", ".env", ".yml", ".yaml", ".toml", ".json", ".cfg", ".ini"}


def tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return out.splitlines()


def iter_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SCAN_EXT or name == ".env":
                yield os.path.join(root, name)


class RepoHygieneTest(unittest.TestCase):
    def test_env_not_tracked(self):
        self.assertNotIn(".env", tracked_files(), ".env git tomonidan kuzatilmoqda")

    def test_no_hardcoded_secrets_in_source(self):
        findings = []
        for path in iter_files():
            if os.path.basename(path).endswith(".example"):
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                for m in re.finditer(pattern, text):
                    line_no = text[: m.start()].count("\n") + 1
                    rel = os.path.relpath(path, REPO_ROOT)
                    findings.append(f"{rel}:{line_no} [{m.group(0)[:10]}...]")
        self.assertEqual(findings, [], "Hardcoded sirlar topildi:\n" + "\n".join(findings))


if __name__ == "__main__":
    unittest.main()
