`timescale 1ns / 1ps
module tb_core;
    `include "router_params.svh"
    `include "tb_meta.svh"
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1;
    logic in_valid = 0;
    logic [271:0] x_flat = 0;
    wire out_valid, escalate;
    wire signed [55:0] score;
    logic [328:0] vectors [0:NCORE-1];
    router_core #(.W_FLAT(W_FLAT), .BIAS(BIAS), .THRESH(THRESH)) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .x_flat(x_flat),
        .out_valid(out_valid), .score(score), .escalate(escalate));
    integer errors = 0;
    integer waited;
    integer handle;
    initial begin
        handle = $fopen("generated/core_vectors.mem", "r");
        if (!handle) $fatal(1, "generated/core_vectors.mem missing");
        $fclose(handle);
        $readmemh("generated/core_vectors.mem", vectors);
        repeat (5) @(negedge clk);
        rst = 0;
        for (integer k = 0; k < NCORE; k++) begin
            @(negedge clk);
            x_flat = vectors[k][271:0];
            in_valid = 1;
            @(negedge clk);
            in_valid = 0;
            waited = 0;
            while (!out_valid && waited < 10) begin
                @(negedge clk);
                waited++;
            end
            if (!out_valid || waited != 3 || score !== vectors[k][327:272] || escalate !== vectors[k][328]) begin
                errors++;
                if (errors < 10) $display("MISMATCH core shot %0d after %0d waits got score %0d escalate %b", k, waited, score, escalate);
            end
        end
        if (errors == 0) $display("PASS core: %0d saved validation shots, score and verdict", NCORE);
        else $fatal(1, "FAIL core: %0d mismatches", errors);
        $finish;
    end
endmodule
