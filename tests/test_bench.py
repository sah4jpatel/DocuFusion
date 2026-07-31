"""Benchmark-harness tests.

These run against a miniature bench tree rather than the real 1403-PDF dataset
so they stay fast and hermetic. What they pin is the part that silently
invalidates a whole benchmark run: the candidate filename convention the
scorer matches with a regex, and the rule that a failed conversion must still
leave a file behind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docfusion.bench.harness import (
    BenchPaths,
    candidate_relpath,
    parse_bench_output,
    prepare_candidate,
    score_candidate,
    write_report_json,
)


@pytest.fixture()
def mini_bench(tmp_path: Path, simple_pdf: Path) -> BenchPaths:
    data = tmp_path / "bench_data"
    (data / "pdfs" / "arxiv_math").mkdir(parents=True)
    (data / "pdfs" / "tables").mkdir(parents=True)
    for name in ("paper_pg1.pdf", "paper_pg2.pdf"):
        (data / "pdfs" / "arxiv_math" / name).write_bytes(simple_pdf.read_bytes())
    (data / "pdfs" / "tables" / "sheet_pg1.pdf").write_bytes(simple_pdf.read_bytes())
    (data / "arxiv_math.jsonl").write_text(
        json.dumps({"pdf": "arxiv_math/paper_pg1.pdf", "page": 1, "type": "present"}) + "\n",
        encoding="utf-8",
    )
    return BenchPaths(tmp_path)


class TestLayout:
    def test_accepts_snapshot_root_or_data_dir(self, mini_bench):
        assert mini_bench.data_dir.name == "bench_data"
        assert BenchPaths(mini_bench.data_dir).data_dir == mini_bench.data_dir

    def test_relpath_keeps_the_category_directory(self):
        """The scorer keys on ``<category>/<stem>_pg{page}_repeat{n}.md``.

        Flattening these makes every test report "missing MD repeats" and the
        entire run scores zero — a silent, total-looking failure.
        """
        assert candidate_relpath("arxiv_math/2503.03754_pg10.pdf") == Path(
            "arxiv_math/2503.03754_pg10_pg1_repeat1.md")
        assert candidate_relpath(Path("tables/a.pdf"), page=2, repeat=3) == Path(
            "tables/a_pg2_repeat3.md")

    def test_missing_bench_gives_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="hf download"):
            BenchPaths(tmp_path / "nope").validate()


class TestPrepare:
    def test_writes_one_file_per_pdf(self, mini_bench):
        candidate = prepare_candidate(mini_bench, convert=lambda p: f"# {p.stem}", name="test")
        files = sorted(f.relative_to(candidate.directory).as_posix()
                       for f in candidate.directory.rglob("*.md"))
        assert files == ["arxiv_math/paper_pg1_pg1_repeat1.md",
                         "arxiv_math/paper_pg2_pg1_repeat1.md",
                         "tables/sheet_pg1_pg1_repeat1.md"]
        assert candidate.pdfs_converted == 3

    def test_category_filter(self, mini_bench):
        candidate = prepare_candidate(mini_bench, convert=lambda p: "x",
                                      name="t2", categories=["tables"])
        assert [f.name for f in candidate.directory.rglob("*.md")] == ["sheet_pg1_pg1_repeat1.md"]

    def test_failed_conversion_still_writes_a_file(self, mini_bench):
        """A missing file is a *candidate error* that zeroes the entire run.

        Writing an empty file instead keeps one broken PDF from looking like
        total pipeline failure.
        """
        def explode(path: Path) -> str:
            if "paper_pg1" in path.name:
                raise RuntimeError("boom")
            return "# fine"

        candidate = prepare_candidate(mini_bench, convert=explode, name="t3")
        assert candidate.pdfs_failed == 1
        assert candidate.failures[0][0] == "paper_pg1.pdf"
        assert (candidate.directory / "arxiv_math" / "paper_pg1_pg1_repeat1.md").read_text(encoding="utf-8") == ""
        assert len(list(candidate.directory.rglob("*.md"))) == 3

    def test_accepts_document_result_objects(self, mini_bench):
        class FakeResult:
            markdown = "# from result"
            decisions = [object(), object()]
            tier2_pages = [1]

        candidate = prepare_candidate(mini_bench, convert=lambda p: FakeResult(), name="t4")
        assert candidate.pages_total == 6          # 3 pdfs x 2 pages
        assert candidate.pages_escalated == 3
        assert candidate.escalation_rate == pytest.approx(0.5)
        assert "from result" in (candidate.directory / "tables" / "sheet_pg1_pg1_repeat1.md").read_text(
            encoding="utf-8"
        )

    def test_skip_existing_avoids_rework(self, mini_bench):
        prepare_candidate(mini_bench, convert=lambda p: "first", name="t5")
        prepare_candidate(mini_bench, convert=lambda p: "second", name="t5")
        assert (mini_bench.candidate_dir("t5") / "tables" / "sheet_pg1_pg1_repeat1.md").read_text(
            encoding="utf-8"
        ) == "first"

    def test_limit_bounds_the_run(self, mini_bench):
        candidate = prepare_candidate(mini_bench, convert=lambda p: "x", name="t6", limit=1)
        assert len(list(candidate.directory.rglob("*.md"))) == 1


class TestScoreParsing:
    # Verbatim shape of upstream's report (olmocr.bench.benchmark).
    SAMPLE = """
  Average Score: 78.9% (95% CI: [77.7%, 80.1%]) over 2323 tests.

============================================================
Final Summary with 95% Confidence Intervals:
docfusion            : Average Score: 78.9% ± 1.2% (average of per-JSONL scores)
    baseline: 94.3% average pass rate over 419 tests
    order   : 65.6% average pass rate over 884 tests

    Results by JSONL file:
        baseline                      : 94.5% (394/417 tests)
        multi_column.jsonl            : 65.6% (580/884 tests)
        table_tests.jsonl             : 79.2% (809/1022 tests)
"""

    def test_parses_overall_and_categories(self):
        report = parse_bench_output(self.SAMPLE, "docfusion")
        assert report.overall == pytest.approx(78.9)
        assert report.ci_low == pytest.approx(77.7)
        assert report.ci_high == pytest.approx(80.1)
        assert {c.name for c in report.categories} == {"multi_column", "table_tests"}
        by_name = {c.name: c for c in report.categories}
        assert by_name["multi_column"].score == pytest.approx(65.6)
        assert by_name["multi_column"].tests == 884

    def test_unparseable_output_never_invents_a_score(self):
        report = parse_bench_output("scorer exploded", "docfusion")
        assert report.overall is None
        assert not report.ok

    def test_report_json_roundtrip(self, tmp_path, mini_bench):
        candidate = prepare_candidate(mini_bench, convert=lambda p: "x", name="t7")
        report = parse_bench_output(self.SAMPLE, "t7")
        out = tmp_path / "report.json"
        write_report_json(report, candidate, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["overall"] == pytest.approx(78.9)
        assert data["conversion"]["pdfs_converted"] == 3


class TestScoreGuards:
    def test_scoring_without_prepare_is_an_error(self, mini_bench):
        with pytest.raises(FileNotFoundError, match="run prepare first"):
            score_candidate(mini_bench, name="never-prepared")
