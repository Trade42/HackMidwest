`timescale 1ns / 1ps
// Latency of fpga_board's router_top, in clocks, measured on its own UART vectors (a check from fpga_router/).
// It sends every packet of generated/uart_vectors.mem over the UART pin, as fpga_board/tb/tb_uart.sv does, and counts:
//   last byte handed over by uart_rx (its valid pulse) -> the core's out_valid -> the first start bit of the reply on txd.
// Run from a folder holding generated/ (fpga_board/, or an export of its exporter).
module tb_latency;
    `include "tb_meta.svh"
    localparam integer CPB = 8;
    localparam integer PKT = 124, RPL = 8, LINE = PKT + RPL;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rxd = 1;
    wire txd;
    wire [15:0] led;
    router_top #(.CLKS_PER_BIT(CPB), .IDLE_CLKS(2000)) dut (.clk_100mhz(clk), .uart_rxd(rxd), .uart_txd(txd), .led(led));
    logic [7:0] vec [0:NUART*LINE-1];

    task automatic send_byte(input logic [7:0] b);
        rxd = 0;
        repeat (CPB) @(negedge clk);
        for (integer i = 0; i < 8; i++) begin rxd = b[i]; repeat (CPB) @(negedge clk); end
        rxd = 1;
        repeat (CPB) @(negedge clk);
    endtask

    integer cyc = 0, t_byte = 0, to_verdict = -1, to_start = -1, differ = 0, seen = 0;
    logic txd_before = 1, replying = 0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (dut.rx_valid) begin t_byte = cyc; replying = 0; end
        if (dut.out_valid) begin
            if (to_verdict != -1 && to_verdict != cyc - t_byte) differ++;
            to_verdict = cyc - t_byte; replying = 1; seen++;
        end
        if (replying && txd_before && !txd) begin
            if (to_start != -1 && to_start != cyc - t_byte) differ++;
            to_start = cyc - t_byte; replying = 0;
        end
        txd_before = txd;
    end

    initial begin
        $readmemh("generated/uart_vectors.mem", vec);
        repeat (40) @(negedge clk);
        for (integer v = 0; v < NUART; v++) begin
            for (integer k = 0; k < PKT; k++) send_byte(vec[v*LINE+k]);
            repeat (RPL * 10 * CPB + 100) @(negedge clk);
        end
        $display("fpga_board latency, clocks, on %0d packets: last byte handed over by uart_rx -> core out_valid = %0d; -> start bit of the reply on txd = %0d; packets on which a count differed = %0d",
                 seen, to_verdict, to_start, differ);
        $finish;
    end
endmodule
