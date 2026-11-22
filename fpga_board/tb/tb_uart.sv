`timescale 1ns / 1ps
module tb_uart;
    `include "tb_meta.svh"
    localparam integer CPB = 8;
    localparam integer PKT = 124, RPL = 8, LINE = PKT + RPL;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rxd = 1;
    wire txd;
    wire [15:0] led;
    logic [7:0] vec [0:NUART*LINE-1];
    logic [7:0] got [0:RPL-1];
    integer got_n = 0;
    router_top #(.CLKS_PER_BIT(CPB), .IDLE_CLKS(2000)) dut (
        .clk_100mhz(clk), .uart_rxd(rxd), .uart_txd(txd), .led(led));

    task automatic send_byte(input logic [7:0] b);
        rxd = 0;
        repeat (CPB) @(negedge clk);
        for (integer i = 0; i < 8; i++) begin
            rxd = b[i];
            repeat (CPB) @(negedge clk);
        end
        rxd = 1;
        repeat (CPB) @(negedge clk);
    endtask

    logic [7:0] rb;
    always begin
        @(negedge txd);
        repeat (CPB + CPB/2) @(posedge clk);
        for (integer j = 0; j < 8; j++) begin
            rb[j] = txd;
            repeat (CPB) @(posedge clk);
        end
        if (got_n < RPL) got[got_n] = rb;
        got_n++;
    end

    integer errors = 0;
    integer waited;
    integer handle;
    initial begin
        handle = $fopen("generated/uart_vectors.mem", "r");
        if (!handle) $fatal(1, "generated/uart_vectors.mem missing");
        $fclose(handle);
        $readmemh("generated/uart_vectors.mem", vec);
        repeat (40) @(negedge clk);
        for (integer v = 0; v < NUART; v++) begin
            if (v == NUART/2) begin
                // Deliberately abandon a partial packet; silence must resync it.
                for (integer k = 0; k < 5; k++) send_byte(8'ha5);
                repeat (3000) @(negedge clk);
            end
            got_n = 0;
            for (integer k = 0; k < PKT; k++) send_byte(vec[v*LINE+k]);
            waited = 0;
            while (got_n < RPL && waited < 2000) begin
                @(negedge clk);
                waited++;
            end
            for (integer k = 0; k < RPL; k++)
                if (got_n != RPL || got[k] !== vec[v*LINE+PKT+k]) begin
                    errors++;
                    if (errors < 10) $display("MISMATCH UART shot %0d byte %0d got %02x want %02x", v, k, got[k], vec[v*LINE+PKT+k]);
                end
        end
        if (errors == 0) $display("PASS UART: %0d original saved shots, byte-exact replies", NUART);
        else $fatal(1, "FAIL UART: %0d mismatches", errors);
        $finish;
    end
endmodule
