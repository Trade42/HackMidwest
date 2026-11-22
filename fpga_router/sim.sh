#!/bin/sh
# Simulate the whole design on one set of weights and vectors. Prints PASS or FAIL, and the latency in clocks.
#   ./sim.sh standin              the made-up weights, 300 shots
#   ./sim.sh trained_X_p0.004     the trained router, 5,000 validation shots
#   ./sim.sh standin trained_X_p0.004    weights of the first folder, vectors of the second: must print FAIL
# Needs verilator (5.x, with --timing) and g++ on the PATH. BUILD (default: /tmp/fpga_router_build) receives the binary.
set -e
cd "$(dirname "$0")"
weights=${1:?which folder of weights: standin or trained_X_p0.004}
vectors=${2:-$weights}
build=${BUILD:-/tmp/fpga_router_build}/sim_${weights}_${vectors}
echo "\$ verilator --binary --timing -Wno-fatal --top-module tb_router -I$weights (router_params.vh) -I$vectors (tb_vectors.vh) tb_router.v router_top.v router_core.v router_frontend.v uart_rx.v uart_tx.v"
verilator --version
mkdir -p "$build/params" "$build/nvec"
cp "$weights/router_params.vh" "$build/params/"       # one include folder per file, so no folder can shadow another
cp "$vectors/tb_vectors.vh" "$build/nvec/"
verilator --binary --timing -Wno-fatal --top-module tb_router -I"$build/params" -I"$build/nvec" --Mdir "$build/obj" \
    tb_router.v router_top.v router_core.v router_frontend.v uart_rx.v uart_tx.v > "$build/build.log" 2>&1 || { cat "$build/build.log"; exit 1; }
echo "sha256 of the weights and the vectors:"
sha256sum "$weights/router_params.vh" "$vectors/vectors.mem"
echo "\$ (cd $vectors && Vtb_router)"
cd "$vectors" && "$build/obj/Vtb_router"
