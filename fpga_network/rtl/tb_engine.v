`timescale 1ns / 1ps
// Testbench of nn_engine: the memories, filled from the hex files that run_engine.py writes into the current folder;
// runs the layer table, then writes the activation memory's output region to a_out.hex and prints the clock count.
module tb_engine;
    `include "tb_params.vh"                 // N_LAYERS, OUT_LO, OUT_HI, from run_engine.py
    localparam integer N = 32, VW = 16, WW = 16, AW = 48;
    localparam integer AA = 20, WA = 13, BA = 8, CA = 10, TA = 19, GA = 11, DA = 6, DW = 640;

    reg clk = 1'b0, rst = 1'b1, start = 1'b0;
    always #5 clk = ~clk;

    reg [DW-1:0]        D [0:(1<<DA)-1];
    reg signed [VW-1:0] A [0:(1<<AA)-1];
    reg [N*WW-1:0]      W [0:(1<<WA)-1];
    reg [N*AW-1:0]      B [0:(1<<BA)-1];
    reg [N*8-1:0]       S [0:(1<<BA)-1];
    reg [WW+AW+7:0]     C [0:(1<<CA)-1];
    reg signed [VW-1:0] T [0:(1<<TA)-1];
    reg [7:0]           G [0:(1<<GA)-1];

    wire done; wire [47:0] cycles;
    wire [DA-1:0] d_addr; wire [AA-1:0] a_addr, b_addr; wire signed [VW-1:0] b_wdata; wire b_we;
    wire [WA-1:0] w_addr; wire [BA-1:0] bias_addr, s_addr; wire [CA-1:0] c_addr; wire [TA-1:0] t_addr; wire [GA-1:0] g_addr;

    nn_engine #(.N(N), .VW(VW), .WW(WW), .AW(AW), .AA(AA), .WA(WA), .BA(BA), .CA(CA), .TA(TA), .GA(GA), .DA(DA), .DW(DW)) dut (
        .clk(clk), .rst(rst), .start(start), .n_layers(N_LAYERS[DA-1:0]), .done(done), .cycles(cycles),
        .d_addr(d_addr), .d_data(D[d_addr]),
        .a_addr(a_addr), .a_data(A[a_addr]),
        .b_addr(b_addr), .b_data(A[b_addr]), .b_wdata(b_wdata), .b_we(b_we),
        .w_addr(w_addr), .w_data(W[w_addr]),
        .bias_addr(bias_addr), .bias_data(B[bias_addr]),
        .s_addr(s_addr), .s_data(S[s_addr]),
        .c_addr(c_addr), .c_data(C[c_addr]),
        .t_addr(t_addr), .t_data(T[t_addr]),
        .g_addr(g_addr), .g_data(G[g_addr]));

    always @(posedge clk) if (b_we) A[b_addr] <= b_wdata;

    initial begin
        $readmemh("desc.hex", D);
        $readmemh("a_in.hex", A);
        $readmemh("w.hex", W);
        $readmemh("bias.hex", B);
        $readmemh("shift.hex", S);
        $readmemh("chan.hex", C);
        $readmemh("table.hex", T);
        $readmemh("graph.hex", G);
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk); start = 1'b1;
        @(posedge clk); start = 1'b0;
        wait (done);
        @(posedge clk);
        $writememh("a_out.hex", A, OUT_LO, OUT_HI);
        $display("DONE: %0d layers in %0d clocks", N_LAYERS, cycles);
        $finish;
    end
endmodule
