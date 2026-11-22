// Turns one 123-byte packet into the 16 router features.
//   bytes   0..116  syndrome: 13 rounds x 72 bits, round r in bytes 9r..9r+8
//   bytes 117..118  BP iterations to the first solution, little endian
//   bytes 119..120  weight of that solution
//   bytes 121..122  16 * log2(1 + mean |posterior|), rounded
// x[0..12] = detectors fired per round, x[13] = iterations, x[14] = weight, x[15] = posterior.
module router_frontend #(
    parameter integer ROUNDS = 13,
    parameter integer BYTES_PER_ROUND = 9,
    parameter integer XW = 16
) (
    input  wire                     clk,
    input  wire                     rst,             // also resyncs after a broken packet
    input  wire                     byte_valid,
    input  wire [7:0]               byte_data,
    output reg                      x_valid,         // one clock, with x_flat complete
    output reg  [(ROUNDS+3)*XW-1:0] x_flat
);
    localparam integer TAIL = 6;
    reg [4:0] round;                                 // ROUNDS means "in the tail"
    reg [3:0] pos;                                   // byte inside the round or the tail
    wire      in_tail = (round == ROUNDS);

    function [3:0] ones8(input [7:0] b);
        ones8 = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + b[7];
    endfunction

    initial begin round = 0; pos = 0; x_valid = 1'b0; end

    always @(posedge clk) begin
        x_valid <= 1'b0;
        if (rst) begin
            round <= 0; pos <= 0;
        end else if (byte_valid) begin
            if (!in_tail) begin
                x_flat[round*XW +: XW] <= (pos == 0 ? {XW{1'b0}} : x_flat[round*XW +: XW]) + ones8(byte_data);
                if (pos == BYTES_PER_ROUND - 1) begin pos <= 0; round <= round + 1'b1; end
                else pos <= pos + 1'b1;
            end else begin
                x_flat[ROUNDS*XW + pos*8 +: 8] <= byte_data;
                if (pos == TAIL - 1) begin pos <= 0; round <= 0; x_valid <= 1'b1; end
                else pos <= pos + 1'b1;
            end
        end
    end
endmodule
