from __future__ import annotations

import json
from pathlib import Path

from factory.config import Settings
from factory.feeds import fetch_diverse_recent, fetch_recent
from factory.trend_ranking import align_primary_sources_to_trends
from factory.trend_sources import fetch_trend_snapshot


def main() -> None:
    settings = Settings.from_env()
    selection = fetch_diverse_recent(
        max_age_hours=settings.max_source_age_hours,
        min_publishers=settings.min_primary_sources,
        fetcher=fetch_recent,
    )
    snapshot = fetch_trend_snapshot(
        max_age_hours=min(settings.max_source_age_hours, 72),
    )
    alignment = align_primary_sources_to_trends(selection.items, snapshot)
    status = dict(snapshot.provider_status)
    provider_successes = sum(value.startswith("ok:") for value in status.values())
    payload = snapshot.as_dict()
    payload.update(
        {
            "primary_source_count": len(selection.items),
            "primary_publisher_count": selection.publisher_count,
            "primary_source_max_age_hours": selection.max_age_hours,
            "provider_successes": provider_successes,
            "alignment": alignment.as_dict(),
        }
    )
    Path("trend-preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    failures: list[str] = []
    if selection.publisher_count < settings.min_primary_sources:
        failures.append(
            f"primary publishers {selection.publisher_count} < {settings.min_primary_sources}"
        )
    if len(snapshot.items) < 3:
        failures.append(f"trend signals {len(snapshot.items)} < 3")
    if provider_successes < 2:
        failures.append(f"successful trend providers {provider_successes} < 2")
    if not alignment.matches:
        failures.append("no trend signal matched an official primary-source article")
    if failures:
        raise SystemExit("Live trend preflight failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
