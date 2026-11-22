// Frozen logistic router, with the software first-leg bypass.
// x[0]=iterations, x[1]=converged, x[2]=mean |posterior| in Q8.8,
// x[3]=correction weight, x[4..16]=13 syndrome cycle weights.
module router_core #(
    parameter integer N = 17,
    parameter integer XW = 16,
    parameter integer WW = 24,
    parameter integer AW = 56,
    parameter logic [N*WW-1:0] W_FLAT = '0,
    parameter logic signed [AW-1:0] BIAS = '0,
    parameter logic signed [AW-1:0] THRESH = '0
) (
    input  logic clk,
    input  logic rst,
    input  logic in_valid,
    input  logic [N*XW-1:0] x_flat,
    output logic out_valid,
    output logic signed [AW-1:0] score,
    output logic escalate
);
    localparam integer PW = XW + WW + 1;
    localparam integer SW = PW + 2;
    localparam integer G = (N + 3) / 4;
    logic signed [PW-1:0] product [0:N-1];
    logic signed [SW-1:0] partial [0:G-1];
    logic signed [AW-1:0] total;
    logic [2:0] valid_pipe, easy_pipe;

    always_ff @(posedge clk) begin
        for (integer k = 0; k < N; k++)
            product[k] <= $signed({1'b0, x_flat[k*XW +: XW]}) * $signed(W_FLAT[k*WW +: WW]);
    end

    // Explicit combinational group sum: a nonblocking sum loop would retain only its last term.
    logic signed [SW-1:0] group_sum [0:G-1];
    logic signed [AW-1:0] all_sum;
    always_comb begin
        for (integer g = 0; g < G; g++) begin
            group_sum[g] = '0;
            for (integer j = 0; j < 4; j++)
                if (4*g+j < N)
                    group_sum[g] = group_sum[g] + product[4*g+j];
        end
        all_sum = BIAS;
        for (integer g = 0; g < G; g++) all_sum = all_sum + partial[g];
    end

    always_ff @(posedge clk) begin
        for (integer g = 0; g < G; g++) partial[g] <= group_sum[g];
        total <= all_sum;
        score <= total;
        // Easy means a first solution converged in the initial 80 iterations.
        escalate <= !easy_pipe[2] && (total > THRESH);
        if (rst) begin
            valid_pipe <= '0;
            easy_pipe <= '0;
            out_valid <= 1'b0;
        end else begin
            valid_pipe <= {valid_pipe[1:0], in_valid};
            easy_pipe <= {easy_pipe[1:0],
                          (x_flat[XW +: XW] != 0 && x_flat[0 +: XW] != 0 && x_flat[0 +: XW] <= 80)};
            out_valid <= valid_pipe[2];
        end
    end
endmodule
