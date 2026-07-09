from src.ingestion.sampling import signal_boosted_sample, stratified_sample, uniform_sample
from src.ingestion.schema import LogEntry, redact_pii


def _entry(i, feature="chat", model="gpt-4o", feedback=None, retried=False, latency=100.0):
    return LogEntry(id=str(i), user_prompt=f"prompt {i}", model=model, response=f"resp {i}",
                    user_feedback=feedback, retried=retried, latency_ms=latency, feature=feature)


def test_redact_email_and_phone():
    text = "Contact me at john@example.com or 555-123-4567"
    red = redact_pii(text)
    assert "john@example.com" not in red
    assert "555-123-4567" not in red
    assert "REDACTED_EMAIL" in red


def test_redacted_logentry():
    e = LogEntry(id="1", user_prompt="my email is a@b.com", model="m", response="ok")
    assert "a@b.com" not in e.redacted().user_prompt


def test_uniform_sample_size():
    entries = [_entry(i) for i in range(100)]
    assert len(uniform_sample(entries, 10)) == 10


def test_stratified_covers_segments():
    entries = [_entry(i, feature="A") for i in range(20)] + [_entry(i, feature="B") for i in range(20, 40)]
    sample = stratified_sample(entries, 10)
    features = {e.feature for e in sample}
    assert features == {"A", "B"}


def test_signal_boosted_prefers_interesting():
    # Downvoted entries are the interesting tail; the rest are genuinely boring
    # (low latency, no feedback, no retry) so p90 latency doesn't flag them.
    interesting = [_entry(i, feedback="thumbs_down", latency=100.0) for i in range(10)]
    boring = [_entry(i, latency=50.0) for i in range(10, 40)]
    sample = signal_boosted_sample(interesting + boring, 10)
    downvoted = sum(1 for e in sample if e.user_feedback == "thumbs_down")
    assert downvoted >= 5  # interesting entries oversampled (70% target)
