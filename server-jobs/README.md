# Remote evaluation jobs

This directory holds job scripts for the project's GPU notebook server. Its runner executes each
script committed under `cpu/` and `gpu/` in filename order, once per script and one at a time
within each folder. Output is recorded on the `server-logs` branch as `job_<lane>_<name>.out`.

Commits to these two folders can trigger work on that server. Contributors should check the
runner's behavior before changing these scripts.
