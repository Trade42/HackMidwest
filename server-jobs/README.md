# server-jobs

Job scripts for the team's GPU notebook server. A small runner there executes every script committed
under `cpu/` and `gpu/`, in name order, once each, one at a time per folder, and writes each job's
printout to the log branch `server-logs` as `job_<lane>_<name>.out`.

**Do not add, change or delete anything here** unless you run the project's orchestration: whatever
is committed in these two folders is executed on that server.
