# First submission handoff

The Git repository is local and has no GitHub remote. Uploading it and sending
the submission remain manual steps.

1. Create a **private** GitHub repository from this directory and grant the
   organizers access. Preserve this independent Git history when uploading.
2. Publish a GitHub release from the local first-submission tag. The organizers
   consider the latest release before the deadline. Attach the seven STL files
   from `results/` and the method PDF; retain the manifest and checksums.
3. Send the seven reconstructions and the private repository/release link to
   **hac2026@helsinki.fi**, identifying the registered team. The method note can
   accompany the submission; its title page deliberately does not assume team
   names or affiliations.
4. Keep the repository available for post-deadline troubleshooting through
   GitHub Issues. Make it permanently public by **October 31, 2026** to remain
   eligible to win.

The current deadline is **September 13, 2026, 23:59 EEST** (**15:59 Chicago**).
Recheck the [official instructions](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/)
immediately before uploading in case access or release details have changed.
They were checked September 7; the latest materials notice was August 27.

The submitted orientation is fixed: rotation about z, vertical contacts at
z=-1 and z=1, light toward (-1,0,0), and the starting phase of the lightcurves.
Do not rotate, recenter, rescale, smooth, or repair the final files in another
application after certification. If outputs are changed, reconstruct and
package them again so their certificates and checksums describe the files
actually delivered.

To reproduce without replacing the committed results, use new directories:

```sh
python -m hac_forward.reconstruct --data-dir data --models 4,5,6,7,8,9,10 \
  --config configs/submission.json --output-dir runs/reproduction
python -m hac_forward.package --run-dir runs/reproduction \
  --config configs/submission.json --output-dir results-reproduction
```

Original challenge data and evaluator source remain external. The run
directories and Python environment are ignored by Git and do not need to be
uploaded; the source, explicit configuration, public regression fixtures,
calibration evidence, results, and document are the release contents.
