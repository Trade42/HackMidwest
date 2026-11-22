// One packet in, eight reply bytes out: signed Q(FRAC) score (7 bytes LE), verdict.
module router_top #(
    parameter integer CLKS_PER_BIT = 868,
    parameter integer IDLE_CLKS = 2_000_000
) (
    input  logic clk_100mhz,
    input  logic uart_rxd,
    output logic uart_txd,
    output logic [15:0] led
);
    `include "router_params.svh"
    logic [3:0] por = 4'hf;
    always_ff @(posedge clk_100mhz) if (por != 0) por <= por - 1'b1;
    logic rst;
    assign rst = por != 0;

    logic rx_valid;
    logic [7:0] rx_data;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk_100mhz), .rst(rst), .rxd(uart_rxd), .valid(rx_valid), .data(rx_data));

    logic [31:0] idle = 0;
    always_ff @(posedge clk_100mhz)
        if (rst || rx_valid) idle <= 0;
        else if (idle < IDLE_CLKS) idle <= idle + 1'b1;
    logic resync;
    assign resync = idle == IDLE_CLKS-1 && !rx_valid;

    logic x_valid;
    logic [17*16-1:0] x_flat;
    router_frontend u_front (
        .clk(clk_100mhz), .rst(rst || resync), .byte_valid(rx_valid),
        .byte_data(rx_data), .x_valid(x_valid), .x_flat(x_flat));

    logic out_valid, verdict;
    logic signed [55:0] score;
    router_core #(.W_FLAT(W_FLAT), .BIAS(BIAS), .THRESH(THRESH)) u_core (
        .clk(clk_100mhz), .rst(rst), .in_valid(x_valid), .x_flat(x_flat),
        .out_valid(out_valid), .score(score), .escalate(verdict));

    logic [63:0] reply;
    logic [3:0] remaining = 0;
    logic tx_start = 0;
    logic tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk_100mhz), .rst(rst), .start(tx_start), .data(reply[7:0]),
        .txd(uart_txd), .busy(tx_busy));

    logic [13:0] shots = 0;
    logic last_verdict = 0;
    always_ff @(posedge clk_100mhz) begin
        tx_start <= 0;
        if (rst) begin
            remaining <= 0;
            shots <= 0;
            last_verdict <= 0;
        end else if (out_valid) begin
            reply <= {7'b0, verdict, score};
            remaining <= 8;
            shots <= shots + 1'b1;
            last_verdict <= verdict;
        end else if (tx_start) begin
            reply <= {8'b0,reply[63:8]};
            remaining <= remaining - 1'b1;
        end else if (remaining != 0 && !tx_busy) begin
            tx_start <= 1'b1;
        end
    end
    assign led = {shots, 1'b0, last_verdict};
endmodule
