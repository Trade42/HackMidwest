// 8N1 UART receiver. CLKS_PER_BIT = clock frequency / baud rate.
module uart_rx #(parameter integer CLKS_PER_BIT = 868) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rxd,
    output reg        valid,                         // one clock per received byte
    output reg  [7:0] data
);
    reg [1:0]  sync;                                 // two flip-flops against metastability
    reg [15:0] count;
    reg [3:0]  bit_index;                            // 0 = start, 1..8 = data, 9 = stop
    reg        busy;
    wire       rx = sync[1];

    initial begin sync = 2'b11; busy = 1'b0; valid = 1'b0; end

    always @(posedge clk) begin
        sync  <= {sync[0], rxd};
        valid <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
        end else if (!busy) begin
            if (!rx) begin busy <= 1'b1; count <= CLKS_PER_BIT / 2; bit_index <= 0; end
        end else if (count != 0) begin
            count <= count - 1'b1;
        end else begin
            count <= CLKS_PER_BIT - 1;
            if (bit_index == 0) begin
                if (rx) busy <= 1'b0;                // false start
                else bit_index <= 1;
            end else if (bit_index <= 8) begin
                data <= {rx, data[7:1]};             // LSB first
                bit_index <= bit_index + 1'b1;
            end else begin
                busy  <= 1'b0;
                valid <= rx;                         // the stop bit must be high
            end
        end
    end
endmodule
