`timescale 1ns / 1ps
// End-to-end test. Every vector goes in over the UART pin and the 6 reply bytes must match the Python model.
// It also counts, on every shot, the clocks from the last packet byte to the verdict and to the reply's first bit.
// Run it from the folder that holds the vectors (standin/ or trained_X_p0.004/): it reads vectors.mem from there.
module tb_router;
    `include "tb_vectors.vh"                         // NVEC, from router_fpga.py export; found through the include directory
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

    // Latency. Everything is sampled on the clock edge, so a difference of two counts is a number of clock periods.
    //   t_byte:  uart_rx hands over a byte (its valid pulse; for the last byte this is the middle of its stop bit)
    //   verdict: the core's out_valid pulse, with score and escalate
    //   start:   the falling edge of the reply's first start bit on the txd pin
    integer cyc = 0, t_byte = 0, lat_verdict = -1, lat_start = -1, lat_changes = 0, lat_seen = 0;
    reg     txd_before = 1'b1, replying = 1'b0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (dut.rx_valid) begin t_byte = cyc; replying = 1'b0; end
        if (dut.out_valid) begin
            if (lat_verdict != -1 && lat_verdict != cyc - t_byte) lat_changes = lat_changes + 1;
            lat_verdict = cyc - t_byte; replying = 1'b1; lat_seen = lat_seen + 1;
        end
        if (replying && txd_before && !txd) begin
            if (lat_start != -1 && lat_start != cyc - t_byte) lat_changes = lat_changes + 1;
            lat_start = cyc - t_byte; replying = 1'b0;
        end
        txd_before = txd;
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
        $display("latency, clocks, counted on %0d verdicts: last byte handed over by uart_rx -> verdict valid = %0d; -> start bit of the reply on txd = %0d; shots on which a count differed = %0d",
                 lat_seen, lat_verdict, lat_start, lat_changes);
        $finish;
    end
endmodule
