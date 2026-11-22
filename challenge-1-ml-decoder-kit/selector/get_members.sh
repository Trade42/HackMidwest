#!/bin/sh
# Copy the members' per-shot files of the exploration shots (train and validation, never test) out of a
# checkout of this repository, into a folder of your choice, with the layout the README's commands use.
#
#   sh selector/get_members.sh <a checkout with origin/laptop-artifacts and origin/server-logs fetched> <folder>
#
# It names every file explicitly; nothing under files/clean/ (the run on the test shots) is listed or read.
# A file that is not on the branch yet is reported and skipped.
set -u
repo=$1
dest=$2
copy() {   # <branch> <path>
    mkdir -p "$dest/$1/$(dirname "$2")"
    if git -C "$repo" show "origin/$1:$2" > "$dest/$1/$2" 2>/dev/null; then
        echo "copied $1:$2  $(sha256sum "$dest/$1/$2" | cut -c1-64)"
    else
        rm -f "$dest/$1/$2"
        echo "not on origin/$1 (yet): $2"
    fi
}
for s in train validation; do
    for b in X Z; do
        copy laptop-artifacts "priortilt/results_8b/selector_${s}_$b.npz"
        copy laptop-artifacts "priortilt/results_p005main/selector_${s}_$b.npz"
        for p in 0.003 0.004 0.005; do
            copy server-logs "files/tcbp_p$p/${s}_$b.tcbp.npz"
            copy server-logs "files/joint_p$p/${s}_$b.joint.npz"
            copy server-logs "files/hier/cand_p$p/${s}_$b.chunks.npz"
        done
    done
done
