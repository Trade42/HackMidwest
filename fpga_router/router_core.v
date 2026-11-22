// Learned router: score = BIAS + sum over k of W[k] * x[k]. The shot escalates when score > thresh.
// x[k] are unsigned XW-bit features, W[k] signed WW-bit weights. Latency: 4 clocks from in_valid to out_valid.
// in_gate travels with in_valid: with in_gate = 0 the shot is never escalated, whatever its score.
module router_core #(
    parameter integer N  = 16,                       // number of features, a multiple of 4
    parameter integer XW = 16,
    parameter integer WW = 16,
    parameter integer AW = 40,
    parameter [N*WW-1:0] W_FLAT = {(N*WW){1'b0}},    // weight k at bits [k*WW +: WW]
    parameter signed [AW-1:0] BIAS = {AW{1'b0}}
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire                 in_gate,             // 0 = accept this shot without asking the model
    input  wire [N*XW-1:0]      x_flat,              // feature k at bits [k*XW +: XW]
    input  wire signed [AW-1:0] thresh,
    output reg                  out_valid,
    output reg  signed [AW-1:0] score,
    output reg                  escalate             // 1 = do not trust the first solution
);
    localparam integer PW = XW + WW + 1;             // one product
    localparam integer SW = PW + 2;                  // sum of four products
    localparam integer G  = N / 4;

    reg [N*PW-1:0]      prod;
    reg [G*SW-1:0]      part;
    reg signed [AW-1:0] acc, total;
    reg [2:0]           valid, gate;

    always @(posedge clk) begin : multiply           // stage 1: N multipliers
        integer k;
        for (k = 0; k < N; k = k + 1)
            prod[k*PW +: PW] <= $signed({1'b0, x_flat[k*XW +: XW]}) * $signed(W_FLAT[k*WW +: WW]);
    end

    always @(posedge clk) begin : add_groups         // stage 2: sums of four
        integer g;
        for (g = 0; g < G; g = g + 1)
            part[g*SW +: SW] <= $signed(prod[(4*g  )*PW +: PW]) + $signed(prod[(4*g+1)*PW +: PW])
                              + $signed(prod[(4*g+2)*PW +: PW]) + $signed(prod[(4*g+3)*PW +: PW]);
    end

    always @(*) begin : add_all                      // stage 3: bias plus the partial sums
        integer g;
        acc = BIAS;
        for (g = 0; g < G; g = g + 1)
            acc = acc + $signed(part[g*SW +: SW]);
    end

    always @(posedge clk) begin
        total    <= acc;
        score    <= total;                           // stage 4: compare
        escalate <= (total > thresh) && gate[2];
        gate     <= {gate[1:0], in_gate};
        if (rst) begin
            valid <= 3'b000; out_valid <= 1'b0;
        end else begin
            valid <= {valid[1:0], in_valid}; out_valid <= valid[2];
        end
    end
endmodule
