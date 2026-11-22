// 8N1 UART transmitter. Pulse start with data while busy is low.
module uart_tx #(parameter integer CLKS_PER_BIT = 868) (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] data,
    output reg        txd,
    output wire       busy
);
    reg [15:0] count;
    reg [3:0]  bits_left;                            // start + 8 data + stop
    reg [8:0]  shift;
    assign busy = (bits_left != 0);

    initial begin txd = 1'b1; bits_left = 0; end

    always @(posedge clk) begin
        if (rst) begin
            txd <= 1'b1; bits_left <= 0;
        end else if (!busy) begin
            if (start) begin txd <= 1'b0; shift <= {1'b1, data}; bits_left <= 10; count <= CLKS_PER_BIT - 1; end
        end else if (count != 0) begin
            count <= count - 1'b1;
        end else begin                               // one bit period is over
            bits_left <= bits_left - 1'b1;
            count     <= CLKS_PER_BIT - 1;
            if (bits_left != 1) begin txd <= shift[0]; shift <= {1'b1, shift[8:1]}; end
        end
    end
endmodule
