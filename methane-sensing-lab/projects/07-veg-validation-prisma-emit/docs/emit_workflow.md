
---

## docs/workflow.md

```markdown
# Workflow

## EMIT Workflow

1. Reformat EMIT L1B observation data.
2. Reformat EMIT L1B radiance data.
3. Run MAG1C on the reformatted radiance product.
4. Visualize methane enhancement results in `Contrast_exp.ipynb`.

## Sentinel-2 Workflow

1. Prepare Sentinel-2 input data.
2. Run `mag1c_s2.py`.
3. Compare Sentinel-2 methane enhancement results with EMIT MAG1C output.

## Main Commands

### EMIT

```bash
bash scripts/run_emit_case1.sh