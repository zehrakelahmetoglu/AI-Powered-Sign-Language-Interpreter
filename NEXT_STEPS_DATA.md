# Data Improvement Next Steps

This file is a practical handoff guide for the next teammate.

## 1) One Last Video Download Round

Run these in order to maximize variety and avoid spending all time on one bucket:

1. python youtube_dataset_builder.py --cats alfabe sayilar --no-skip
2. python youtube_dataset_builder.py --cats aile zamirler --no-skip
3. python youtube_dataset_builder.py --cats icecekler aylar fiiller --no-skip
4. python verify_data.py
5. python dataset_health_score.py

Notes:
- Use --limit 50 if internet or runtime is limited.
- Keep --no-skip in the final round to force additional sequences.

## 2) Assign Teammate Tasks

Main goal for next teammate:
- Raise all classes to at least 20 samples first.

Daily workflow:
1. Run python collection_backlog_planner.py --target 20 --top 120 --people 2 --daily-target-per-person 80
2. Open handoff_tasks.md and pick assigned class list.
3. Collect missing samples (manual camera or source videos).
4. Run python verify_data.py.
5. Run python dataset_health_score.py and record score in commit message.

Definition of done for Phase 1:
- 100 percent of active classes reach 20+ samples.
- Dataset Health Score reaches at least 35/100.

## 3) Quality Rules

- Do not add near-duplicate sequences in a single burst.
- Use variation: person, distance, angle, lighting, background.
- Keep frame quality clean and full hand visible.
- Remove broken files immediately if verify_data reports errors.

## 4) Phase 2 Target

After Phase 1 is done:
- Raise classes to 50+ samples.
- Aim Dataset Health Score >= 60/100.
