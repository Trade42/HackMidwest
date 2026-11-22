`timescale 1ns / 1ps
// End-to-end test. Every vector goes in over the UART pin and the 6 reply bytes must match the Python model.
module tb_router;
    `include "tb_vectors.vh"                         // NVEC, from router_fpga.py export
    localparam integer CPB = 8;                      // fast UART for simulation
    localparam integer PKT = 123, RPL = 6, LINE = PKT + RPL;

    reg clk = 1'b0;
    always #5 clk = ~clk;                            // 100 MHz

    reg         rxd = 1'b1;
    wire        txd;
    wire [15:0] led;
    router_top #(.CLKS_PER_BIT(CPB), .IDLE_CLKS(2000)) dut (.clk_100mhz(clk), .uart_rxd(rxd), .uart_txd(txd), .led(led));

    reg [7:0] vec [0:NVEC*LINE-1];
    reg [7:0] got [0:RPL-1];
    integer   got_n = 0;

    task send_byte(input [7:0] b);
        integer i;
        begin
            rxd = 1'b0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin rxd = b[i]; repeat (CPB) @(posedge clk); end
            rxd = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask

    reg [7:0] rb;                                    // UART monitor on the reply pin
    integer   j;
    always begin
        @(negedge txd);
        repeat (CPB + CPB / 2) @(posedge clk);
        for (j = 0; j < 8; j = j + 1) begin rb[j] = txd; repeat (CPB) @(posedge clk); end
        if (got_n < RPL) got[got_n] = rb;
        got_n = got_n + 1;
    end

    integer v, k, errors = 0, waited;
    initial begin
        $readmemh("vectors.mem", vec);
        repeat (40) @(posedge clk);
        for (v = 0; v < NVEC; v = v + 1) begin
            if (v == NVEC / 2) begin                 // a broken packet mid-run must not shift the later ones
                for (k = 0; k < 5; k = k + 1) send_byte(8'hA5);
                repeat (3000) @(posedge clk);
            end
            got_n = 0;
            for (k = 0; k < PKT; k = k + 1) send_byte(vec[v*LINE + k]);
            waited = 0;
            while (got_n < RPL && waited < 2000) begin @(posedge clk); waited = waited + 1; end
            for (k = 0; k < RPL; k = k + 1)
                if (got_n != RPL || got[k] !== vec[v*LINE + PKT + k]) begin
                    errors = errors + 1;
                    if (errors <= 10) $display("MISMATCH shot %0d byte %0d: got %02x, want %02x (%0d reply bytes)", v, k, got[k], vec[v*LINE + PKT + k], got_n);
                end
        end
        if (errors == 0) $display("PASS: %0d shots match the Python model bit for bit", NVEC);
        else             $display("FAIL: %0d byte mismatches", errors);
        $finish;
    end
endmodule
