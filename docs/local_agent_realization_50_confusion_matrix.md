# Local Agent Realization Confusion Matrix

- Source: `results/local_agent_realization_50.csv`
- Metric: gold action vs predicted action (`recommended_action`)
- Percentages are row-normalized within each gold action.

## Counts

| gold_action | repair | clarify | support | handoff | listen | total |
|---|---|---|---|---|---|---|
| repair | 13 | 0 | 0 | 0 | 13 |
| clarify | 2 | 8 | 1 | 0 | 13 |
| support | 3 | 0 | 6 | 0 | 12 |
| handoff | 1 | 0 | 0 | 11 | 12 |

## Row Percentages

| gold_action | repair | clarify | support | handoff | listen | total |
|---|---|---|---|---|---|---|
| repair | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| clarify | 15.4% | 61.5% | 7.7% | 0.0% | 15.4% | 100.0% |
| support | 25.0% | 0.0% | 50.0% | 0.0% | 25.0% | 100.0% |
| handoff | 8.3% | 0.0% | 0.0% | 91.7% | 0.0% | 100.0% |

## Cell View

| gold_action | repair | clarify | support | handoff | listen | total |
|---|---|---|---|---|---|---|
| repair | 13 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 13 |
| clarify | 2 (15.4%) | 8 (61.5%) | 1 (7.7%) | 0 (0.0%) | 2 (15.4%) | 13 |
| support | 3 (25.0%) | 0 (0.0%) | 6 (50.0%) | 0 (0.0%) | 3 (25.0%) | 12 |
| handoff | 1 (8.3%) | 0 (0.0%) | 0 (0.0%) | 11 (91.7%) | 0 (0.0%) | 12 |