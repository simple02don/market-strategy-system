import subprocess
from pathlib import Path


def test_setup_cron_removes_legacy_tail_jobs_but_keeps_legacy_report_service(tmp_path):
    project_dir = tmp_path / "market-strategy-system"
    project_dir.mkdir()
    source = Path(__file__).resolve().parents[1] / "setup_cron.sh"
    script = project_dir / "setup_cron.sh"
    script.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cron_file = tmp_path / "crontab"
    cron_file.write_text(
        "35 14 * * 1-5 cd /home/ubuntu/jckx-tail-overnight && ./run.sh run --strict-asof\n"
        "@reboot cd /home/ubuntu/jckx-tail-overnight && ./start_http.sh\n",
        encoding="utf-8",
    )
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-l\" ]; then cat \"$FAKE_CRON_FILE\"; else cp \"$1\" \"$FAKE_CRON_FILE\"; fi\n",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    subprocess.run(
        [str(script)],
        check=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "FAKE_CRON_FILE": str(cron_file),
            "JCKX_APP_USER": "ubuntu",
        },
        capture_output=True,
        text=True,
    )

    installed = cron_file.read_text(encoding="utf-8")
    assert "jckx-tail-overnight && ./run.sh" not in installed
    assert "jckx-tail-overnight && ./start_http.sh" in installed
    assert "50-56/2 14 * * 1-5" in installed
    assert f"cd {project_dir} && ./run.sh tail-review" in installed
