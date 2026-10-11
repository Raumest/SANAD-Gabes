"""M6: every dashboard page renders without exceptions on the committed demo outputs."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + sorted(p.relative_to(APP).as_posix() for p in (APP / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(APP / page), default_timeout=120)
    at.session_state["run_id"] = "demo"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
