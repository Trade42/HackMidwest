// Router on the Urbana board: one 123-byte packet in over UART, one 6-byte verdict out.
//   reply bytes 0..4: the 40-bit score, little endian. byte 5: 1 = escalate, 0 = accept the first solution.
module router_top #(
    parameter integer CLKS_PER_BIT = 868,            // 100 MHz / 115200 baud
    parameter integer IDLE_CLKS    = 2_000_000       // 20 ms of silence drops a half-received packet
) (
    input  wire        clk_100mhz,
    input  wire        uart_rxd,                     // host to FPGA
    output wire        uart_txd,                     // FPGA to host
    output wire [15:0] led
);
    `include "router_params.vh"                      // W_FLAT, BIAS, THRESH, FRAC from router_fpga.py export
    localparam integer N = 16, XW = 16, WW = 16, AW = 40;

    wire clk = clk_100mhz;
    reg [3:0] por = 4'hF;                            // power-on reset
    always @(posedge clk) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0);

    wire       rx_valid;
    wire [7:0] rx_data;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst(rst), .rxd(uart_rxd), .valid(rx_valid), .data(rx_data));

    // One lost byte must not shift every later packet: after IDLE_CLKS of silence the packet counter restarts.
    reg [31:0] idle = 0;
    always @(posedge clk)
        if (rst || rx_valid)        idle <= 0;
        else if (idle != IDLE_CLKS) idle <= idle + 1'b1;
    wire resync = (idle == IDLE_CLKS - 1) && !rx_valid;

    wire            x_valid;
    wire [N*XW-1:0] x_flat;
    router_frontend #(.XW(XW)) u_front (.clk(clk), .rst(rst | resync), .byte_valid(rx_valid), .byte_data(rx_data),
                                        .x_valid(x_valid), .x_flat(x_flat));

    wire                 out_valid, escalate;
    wire signed [AW-1:0] score;
    router_core #(.N(N), .XW(XW), .WW(WW), .AW(AW), .W_FLAT(W_FLAT), .BIAS(BIAS)) u_core (
        .clk(clk), .rst(rst), .in_valid(x_valid), .x_flat(x_flat), .thresh(THRESH),
        .out_valid(out_valid), .score(score), .escalate(escalate));

    reg  [47:0] reply;
    reg  [2:0]  left = 0;                            // reply bytes still to send
    reg         tx_start = 1'b0;
    wire        tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst(rst), .start(tx_start), .data(reply[7:0]),
                                                 .txd(uart_txd), .busy(tx_busy));

    reg [13:0] shots = 0;
    reg        last  = 1'b0;
    always @(posedge clk) begin
        tx_start <= 1'b0;
        if (rst) begin
            left <= 0; shots <= 0; last <= 1'b0;
        end else if (out_valid) begin
            reply <= {7'b0, escalate, score};
            left  <= 6; shots <= shots + 1'b1; last <= escalate;
        end else if (tx_start) begin
            reply <= {8'b0, reply[47:8]};
            left  <= left - 1'b1;
        end else if (left != 0 && !tx_busy) begin
            tx_start <= 1'b1;
        end
    end

    assign led = {shots, 1'b0, last};                // led[0] = last verdict, led[15:2] = shots seen
endmodule
