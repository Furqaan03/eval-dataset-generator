from src.labeling.dataset import EvalCase
from src.runner.eval_runner import coverage_report, detect_regressions, run_eval, score_response


def _case(cid, must_contain=None, must_not=None):
    return EvalCase(id=cid, user_prompt="p", reference_answer="a", expected_behavior="answer",
                    must_contain=must_contain or [], must_not_contain=must_not or [], category="cat", difficulty="simple")


def test_score_passes_with_required_tokens():
    case = _case("1", must_contain=["8080"], must_not=["error"])
    assert score_response(case, "The port is 8080") is True
    assert score_response(case, "The port is 9090") is False   # missing required
    assert score_response(case, "8080 but error") is False     # contains trap


def test_run_and_detect_regressions():
    cases = [_case("1", must_contain=["yes"]), _case("2", must_contain=["ok"])]
    prev = run_eval(cases, {"1": "yes", "2": "ok"})           # both pass
    curr = run_eval(cases, {"1": "no", "2": "ok"})            # case 1 regressed
    diff = detect_regressions(curr, prev)
    assert diff["new_failures"] == ["1"]
    assert diff["new_passes"] == []


def test_coverage_report():
    cases = [_case("1"), _case("2")]
    report = coverage_report(cases)
    assert report["total"] == 2
    assert report["by_category"]["cat"] == 2
    assert report["auto_labeled"] == 2  # EvalCase defaults source="auto"
