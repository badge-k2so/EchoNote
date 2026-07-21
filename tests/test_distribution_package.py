"""distribution/ 配下の配布用スクリプトの体裁・内容の検証。

- .ps1 は UTF-8 BOM 付き（PowerShell 5.1 の文字化け対策）
- .bat は ASCII のみ（CP932 前提端末での文字化け対策）
- setup.bat が ExecutionPolicy Bypass でセットアップを起動する
- Python インストーラーが既定で同梱される（-SkipPythonInstaller で除外）
- セットアップ冒頭に空き容量チェックがある
- 端末スペック（メモリ11.5GB超）でAI要約用4Bモデルの保持/削除を自動判定する
- verify_setup.py が結果を setup_report.txt に保存する
"""
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DISTRIBUTION = REPO_ROOT / "distribution"


class PowerShellScriptEncodingTests(unittest.TestCase):
    def test_ps1_files_have_utf8_bom(self) -> None:
        ps1_files = sorted(DISTRIBUTION.glob("*.ps1"))
        self.assertTrue(ps1_files, "distribution/ に .ps1 が見つかりません")
        for path in ps1_files:
            with self.subTest(file=path.name):
                head = path.read_bytes()[:3]
                self.assertEqual(
                    head, b"\xef\xbb\xbf", f"{path.name} に UTF-8 BOM がありません"
                )


class BatchFileTests(unittest.TestCase):
    def test_bat_files_are_ascii_only(self) -> None:
        bat_files = sorted(DISTRIBUTION.glob("*.bat"))
        self.assertTrue(bat_files, "distribution/ に .bat が見つかりません")
        for path in bat_files:
            with self.subTest(file=path.name):
                data = path.read_bytes()
                non_ascii = [b for b in data if b > 0x7F]
                self.assertFalse(
                    non_ascii,
                    f"{path.name} に非ASCII文字が含まれています（文字化けの原因）",
                )

    def test_setup_bat_bypasses_execution_policy(self) -> None:
        text = (DISTRIBUTION / "setup.bat").read_text(encoding="ascii")
        self.assertIn("-ExecutionPolicy Bypass", text)
        self.assertIn("setup_test_pc.ps1", text)
        self.assertIn("-NoProfile", text)

    def test_launcher_pauses_when_precondition_fails(self) -> None:
        text = (DISTRIBUTION / "OtoWeaveを起動.bat").read_text(encoding="ascii")
        self.assertIn("pythonw.exe", text)
        self.assertIn("pause", text)
        self.assertIn("setup.bat", text)


class BuildScriptTests(unittest.TestCase):
    def test_python_installer_bundled_by_default(self) -> None:
        text = (DISTRIBUTION / "build_distribution.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("$SkipPythonInstaller", text)
        self.assertIn("-not $SkipPythonInstaller", text)
        self.assertNotIn("IncludePythonInstaller", text)
        # 入手できない場合の案内（入手元URLと置き場所）がメッセージに含まれる
        self.assertIn("python-3.12", text)
        self.assertIn("www.python.org", text)

    def test_build_script_ships_setup_bat(self) -> None:
        text = (DISTRIBUTION / "build_distribution.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("setup.bat", text)

    def test_build_script_ships_verify_offline(self) -> None:
        # 外部送信なしの証跡を作るスクリプトはパッケージ同梱が前提
        # （テスト機はオフラインなので後から配れない）。
        text = (DISTRIBUTION / "build_distribution.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("verify_offline.ps1", text)


class SetupScriptTests(unittest.TestCase):
    def test_disk_space_check_runs_before_python_check(self) -> None:
        text = (DISTRIBUTION / "setup_test_pc.ps1").read_text(encoding="utf-8-sig")
        disk_pos = text.find("空き容量を確認しています")
        python_pos = text.find("Python 3.12 を確認しています")
        self.assertGreater(disk_pos, -1, "空き容量チェックが見つかりません")
        self.assertGreater(python_pos, -1, "Python チェックが見つかりません")
        self.assertLess(disk_pos, python_pos, "空き容量チェックが Python 確認より後です")

    def test_lite_edition_detection_clues(self) -> None:
        text = (DISTRIBUTION / "setup_test_pc.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("_Lite", text)
        self.assertIn("Qwen3.5-4B-Q4_K_M.gguf", text)


class SetupScriptSpecDetectionTests(unittest.TestCase):
    """setup_test_pc.ps1 の端末スペック自動判定（AI要約4Bの保持/削除）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (DISTRIBUTION / "setup_test_pc.ps1").read_text(encoding="utf-8-sig")

    def test_reads_physical_memory(self) -> None:
        self.assertIn("Get-CimInstance Win32_ComputerSystem", self.text)
        self.assertIn("TotalPhysicalMemory", self.text)

    def test_uses_same_threshold_as_app(self) -> None:
        # アプリ側 otoweave_app/llm_chat.py の
        # _LOW_MEMORY_THRESHOLD_BYTES = int(11.5 * 1024**3) と同じ閾値
        self.assertIn("11.5", self.text)
        self.assertIn("_LOW_MEMORY_THRESHOLD_BYTES", self.text)
        self.assertIn("llm_chat.py", self.text)

    def test_targets_summary_model_file(self) -> None:
        self.assertIn("Qwen3.5-4B-Q4_K_M.gguf", self.text)
        # 削除前に存在確認する（Lite版でも壊れない）
        self.assertIn("Test-Path $Model4BPath", self.text)

    def test_force_options_exist(self) -> None:
        self.assertIn("$KeepSummaryModel", self.text)
        self.assertIn("$RemoveSummaryModel", self.text)

    def test_detection_appends_to_setup_report(self) -> None:
        self.assertIn("setup_report.txt", self.text)
        self.assertIn("Add-Content", self.text)


class PrivacyDocTests(unittest.TestCase):
    """docs/データの取り扱いと確認方法.md（学校・保護者向け説明文書）。"""

    DOC = DISTRIBUTION / "docs" / "データの取り扱いと確認方法.md"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = cls.DOC.read_text(encoding="utf-8")

    def test_doc_exists(self) -> None:
        self.assertTrue(self.DOC.is_file(), "データの取り扱いと確認方法.md がありません")

    def test_main_sections_present(self) -> None:
        for heading in [
            "扱うデータと保存場所",
            "送信する仕組みがそもそも無い",
            "自分で確かめる3つの方法",
            "機内モード",
            "通信監視",
            "ソースコード開示",
            "注意点",
            "お問い合わせ",
        ]:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_mentions_verification_assets(self) -> None:
        # 3つの確認方法が具体的な手段（同梱物）とひも付いている
        self.assertIn("verify_offline.ps1", self.text)
        self.assertIn("offline_report.txt", self.text)
        self.assertIn("127.0.0.1", self.text)
        self.assertIn("-BlockNetwork", self.text)

    def test_honest_caveats_present(self) -> None:
        # 注意点を正直に書く（OneDrive同期・共有端末・30日ゴミ箱）
        self.assertIn("OneDrive", self.text)
        self.assertIn("_trash", self.text)
        self.assertIn("30日", self.text)
        self.assertIn("アカウント", self.text)


class VerifyOfflineScriptTests(unittest.TestCase):
    """verify_offline.ps1（通信監視・記録スクリプト）。"""

    SCRIPT = DISTRIBUTION / "verify_offline.ps1"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = cls.SCRIPT.read_text(encoding="utf-8-sig")

    def test_script_exists_with_utf8_bom(self) -> None:
        # BOM は PowerShellScriptEncodingTests でも網羅されるが、
        # このスクリプト単体でも明示的に確認する
        self.assertTrue(self.SCRIPT.is_file(), "verify_offline.ps1 がありません")
        self.assertEqual(self.SCRIPT.read_bytes()[:3], b"\xef\xbb\xbf")

    def test_monitors_tcp_connections(self) -> None:
        self.assertIn("Get-NetTCPConnection", self.text)
        self.assertIn("Get-Process", self.text)
        # 監視対象はパス（配布フォルダ配下）で判定する
        self.assertIn("$_.Path.StartsWith($Root", self.text)

    def test_loopback_exclusion_logic(self) -> None:
        # ループバック（127.0.0.1 / ::1 など）宛ては「外部」に数えない
        self.assertIn("Test-LoopbackAddress", self.text)
        self.assertIn("'::1'", self.text)
        self.assertIn("'127.*'", self.text)

    def test_duration_parameter_and_report(self) -> None:
        self.assertIn("$DurationMinutes = 10", self.text)
        self.assertIn("offline_report.txt", self.text)
        self.assertIn("期待値: 0 件", self.text)

    def test_parses_under_powershell(self) -> None:
        command = (
            "$tokens = $null; $errors = $null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{self.SCRIPT}', [ref]$tokens, [ref]$errors) | Out-Null;"
            "exit $errors.Count"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, "verify_offline.ps1 に構文エラー")


class BlockNetworkOptionTests(unittest.TestCase):
    """setup_test_pc.ps1 の -BlockNetwork（外部送信ブロック・管理者向け）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (DISTRIBUTION / "setup_test_pc.ps1").read_text(encoding="utf-8-sig")

    def test_block_network_parameter_exists(self) -> None:
        self.assertIn("[switch]$BlockNetwork", self.text)

    def test_creates_outbound_block_rules(self) -> None:
        self.assertIn("New-NetFirewallRule", self.text)
        self.assertIn("-Direction Outbound", self.text)
        self.assertIn("-Action Block", self.text)
        self.assertIn("OtoWeave-NoNetwork-", self.text)

    def test_replaces_existing_rules_on_rerun(self) -> None:
        self.assertIn("Remove-NetFirewallRule", self.text)
        self.assertIn("'OtoWeave-NoNetwork-*'", self.text)

    def test_checks_admin_and_skips_gracefully(self) -> None:
        self.assertIn("WindowsBuiltInRole]::Administrator", self.text)
        self.assertIn("管理者権限が無いため", self.text)

    def test_existing_spec_detection_untouched(self) -> None:
        # 直前に実装されたスペック自動判定が残っていること（退行防止）
        self.assertIn("$KeepSummaryModel", self.text)
        self.assertIn("$RemoveSummaryModel", self.text)
        self.assertIn("11.5", self.text)


class LauncherOfflineEnvTests(unittest.TestCase):
    """起動batのオフライン環境変数（外部問い合わせの遮断）。"""

    def test_offline_env_vars_set(self) -> None:
        # ASCII のみで読めること自体も検証になる（文字化け対策の維持）
        text = (DISTRIBUTION / "OtoWeaveを起動.bat").read_text(encoding="ascii")
        self.assertIn('set "HF_HUB_OFFLINE=1"', text)
        self.assertIn('set "TRANSFORMERS_OFFLINE=1"', text)
        self.assertIn('set "HF_HOME=', text)


class ReadmeAndTestPlanLinkTests(unittest.TestCase):
    """既存ドキュメントから新文書・スクリプトへの導線。"""

    def test_readme_mentions_privacy_doc(self) -> None:
        text = (DISTRIBUTION / "はじめにお読みください.txt").read_text(encoding="utf-8")
        self.assertIn("データの取り扱いと確認方法.md", text)
        self.assertIn("verify_offline.ps1", text)

    def test_test_plan_has_offline_scenario(self) -> None:
        text = (DISTRIBUTION / "docs" / "テスト手順書.md").read_text(encoding="utf-8")
        self.assertIn("機内モード", text)
        self.assertIn("verify_offline.ps1", text)
        self.assertIn("offline_report.txt", text)


class PowerShellSyntaxTests(unittest.TestCase):
    def test_ps1_files_parse_without_errors(self) -> None:
        """PowerShell 5.1 のパーサーで構文エラーが無いことを確認する。"""
        ps1_files = sorted(DISTRIBUTION.glob("*.ps1"))
        self.assertTrue(ps1_files, "distribution/ に .ps1 が見つかりません")
        for path in ps1_files:
            with self.subTest(file=path.name):
                command = (
                    "$tokens = $null; $errors = $null;"
                    "[System.Management.Automation.Language.Parser]::ParseFile("
                    f"'{path}', [ref]$tokens, [ref]$errors) | Out-Null;"
                    "$errors | ForEach-Object { $_.Message };"
                    "exit $errors.Count"
                )
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{path.name} に構文エラー: {result.stdout} {result.stderr}",
                )


class VerifySetupReportTests(unittest.TestCase):
    def test_write_report_creates_setup_report_txt(self) -> None:
        sys.path.insert(0, str(DISTRIBUTION))
        try:
            verify_setup = importlib.import_module("verify_setup")
        finally:
            sys.path.remove(str(DISTRIBUTION))
        original_root = verify_setup.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                verify_setup.ROOT = Path(tmp)
                verify_setup.write_report(["[OK] テスト行", "すべてOKです。"])
                report = Path(tmp) / "setup_report.txt"
                self.assertTrue(report.is_file())
                content = report.read_text(encoding="utf-8")
                self.assertIn("[OK] テスト行", content)
                self.assertIn("すべてOKです。", content)
        finally:
            verify_setup.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
