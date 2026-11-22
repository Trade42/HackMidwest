// A small engine that walks the network layer by layer (fpga_network/README.md, step 4).
//
// N multiply-add units ("lanes") work on N output channels of one position at once. On every clock one input value
// is read from the activation memory and given to all lanes, each with its own weight (the weight memory's word holds
// N weights). After K inputs, each lane's sum gets its bias (preloaded), then the lanes are written one by one:
// round to the output's scale (add half, arithmetic shift right; or shift left), saturate, optionally look the
// result up in a SiLU table, optionally add the old residual stream (rescaled, saturated), and write.
//
// A table of layer descriptions (one DW-bit word per layer) tells the controller what to do. Four kinds:
//   LINEAR  out[p, g*N + l] = f(sum_k W[g][k][l] * in[p*K + k] + bias[g][l])           (down, up)
//   HOP_CD  checks -> data qubits: target (cycle c, data node t) reads, for 6 neighbour slots s, the c_in values of
//           check node nbr[t][s] in round c ("same") and round c + 1 ("after"): K = 2 * 6 * c_in
//   HOP_DC  data qubits -> checks: target (round r, check t) reads cycle r ("same", if r < T_src) and cycle r - 1
//           ("after", if r >= 1); a missing input counts as zero; the first, middle and last rounds have own biases
//   AFFINE  per channel o, for every value: f(a[o] * in[e] + c[o]): the normalisation before a SiLU
// The two target blocks of a collecting step (X/Z checks, left/right data qubits) have their own weights and biases.
//
// Every memory is outside this module (ports), so that synthesis counts the engine. A memory's address is a register
// of this module and its data is valid in the next clock, as a block RAM's registered address gives.
// All values are VW-bit two's complement; a layer saturates to its own width (sat_bits <= VW).
module nn_engine #(
    parameter integer N  = 32,      // lanes (multiply-add units)
    parameter integer VW = 16,      // value width in memory
    parameter integer WW = 16,      // weight width
    parameter integer AW = 48,      // accumulator width (40 bits suffice at 16 bits: reference.accumulator_bits)
    parameter integer AA = 20,      // activation address bits
    parameter integer WA = 13,      // weight address bits
    parameter integer BA = 8,       // bias / shift address bits
    parameter integer CA = 10,      // per-channel (affine) memory address bits
    parameter integer TA = 19,      // SiLU table address bits
    parameter integer GA = 11,      // neighbour table address bits
    parameter integer DA = 6,       // layer table address bits
    parameter integer DW = 640      // one layer description: 20 fields of 32 bits
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire [DA-1:0]        n_layers,
    output reg                  done,
    output reg  [47:0]          cycles,

    output reg  [DA-1:0]        d_addr,  input wire [DW-1:0]         d_data,
    output reg  [AA-1:0]        a_addr,  input wire signed [VW-1:0]  a_data,      // read port: inputs
    output reg  [AA-1:0]        b_addr,  input wire signed [VW-1:0]  b_data,      // read/write port: outputs
    output reg  signed [VW-1:0] b_wdata, output reg                  b_we,
    output reg  [WA-1:0]        w_addr,  input wire [N*WW-1:0]       w_data,
    output reg  [BA-1:0]        bias_addr, input wire [N*AW-1:0]     bias_data,
    output reg  [BA-1:0]        s_addr,  input wire [N*8-1:0]        s_data,
    output reg  [CA-1:0]        c_addr,  input wire [WW+AW+8-1:0]    c_data,      // {shift, c, a}
    output reg  [TA-1:0]        t_addr,  input wire signed [VW-1:0]  t_data,
    output reg  [GA-1:0]        g_addr,  input wire [7:0]            g_data
);
    localparam [3:0] LINEAR = 0, HOP_CD = 1, HOP_DC = 2, AFFINE = 3;

    // ---------------------------------------------------------------- the rounding rule, as in reference.requant
    function signed [AW-1:0] rescale(input signed [AW-1:0] v, input signed [7:0] s);
        reg signed [AW-1:0] one;
        begin
            one = 1;
            if (s > 0) rescale = (v + (one <<< (s - 8'sd1))) >>> s;     // add half, round down
            else       rescale = v <<< (-s);                           // exact
        end
    endfunction

    function signed [VW-1:0] saturate(input signed [AW-1:0] v, input [4:0] bits);
        reg signed [AW-1:0] one, hi, lo;
        begin
            one = 1;
            hi = (one <<< (bits - 1)) - one;
            lo = -hi - one;
            saturate = (v > hi) ? hi[VW-1:0] : (v < lo) ? lo[VW-1:0] : v[VW-1:0];
        end
    endfunction

    // ---------------------------------------------------------------- the current layer's description
    reg [31:0] f [0:19];
    wire [3:0]  kind      = f[0][3:0];
    wire        use_silu  = f[1][0];
    wire        residual  = f[1][1];
    wire [31:0] P         = f[2];         // positions (LINEAR, AFFINE); HOP: T * NODES
    wire [31:0] K         = f[3];         // inputs per output (AFFINE: channels)
    wire [31:0] G         = f[4];         // groups of N output channels (LINEAR)
    wire [31:0] C_out     = f[5];         // values written per position
    wire [31:0] in_base   = f[6], out_base = f[7], res_base = f[8], w_base = f[9];
    wire [31:0] b_base    = f[10], s_base = f[11], t_base = f[12];
    wire [4:0]  sat_bits  = f[13][4:0];
    wire signed [7:0] res_shift = f[14][7:0];
    wire [4:0]  res_bits  = f[14][12:8];
    wire [7:0]  T         = f[15][7:0], NODES = f[15][15:8], C_IN = f[15][23:16], T_SRC = f[15][31:24];
    wire [31:0] g_base    = f[16], chan_base = f[17];

    // ---------------------------------------------------------------- state
    localparam [4:0] IDLE = 0, DESC0 = 1, DESC1 = 2, DESC2 = 3, POS0 = 4, POS1 = 5, ACC = 6, DRAIN = 7,
                     WR0 = 8, WR1 = 9, WR2 = 10, NEXT = 11, AF0 = 12, AF1 = 13, AF2 = 14, AF3 = 15, FINISH = 16;
    reg [4:0]  state;
    reg [DA-1:0] layer;
    reg [31:0] p, g, e, o;                 // position, group; element and its channel (AFFINE)
    reg [7:0]  tau, t;                     // HOP: time and target node of the position
    reg [31:0] k;                          // input counter
    reg        h; reg [2:0] slot; reg [7:0] i;   // HOP: k = h * 6 * C_IN + slot * C_IN + i
    reg [7:0]  lane;
    reg [3:0]  drain;
    reg signed [AW-1:0] acc [0:N-1];
    reg signed [7:0]    sh  [0:N-1];
    reg signed [AW-1:0] sel;
    reg signed [7:0]    sel_sh;
    reg signed [VW-1:0] rq;

    wire blk = (NODES != 0) && (t >= NODES / 2);       // the target's block: X or Z checks, left or right data qubits

    // the accumulate pipeline: stage 0 issues the neighbour look-up, stage 1 the input and weight reads, stage 2 adds
    reg        v0, v1, z0, z1;             // valid; z: the input is missing (counts as zero)
    reg [31:0] k0, k1, srct0, srct1;
    reg [7:0]  i0, i1;
    reg        h0;

    // the SiLU table's address of a value: the table holds one entry per value from -2^(bits-1) up
    function [31:0] table_index(input signed [VW-1:0] v, input [4:0] bits);
        reg signed [31:0] wide;
        begin
            wide = v;                                                  // sign extension
            table_index = wide + (32'sd1 <<< (bits - 1));
        end
    endfunction

    reg signed [AW-1:0] old_q, sum;
    integer l;
    always @(posedge clk) begin
        b_we <= 1'b0;
        if (rst) begin
            state <= IDLE; done <= 1'b0; cycles <= 0; v0 <= 0; v1 <= 0;
        end else begin
            if (state != IDLE) cycles <= cycles + 1;

            // ---- stage 2 of the accumulate pipeline: the reads of stage 1 have arrived; every lane adds its product
            if (v1 && !z1) for (l = 0; l < N; l = l + 1)
                acc[l] <= acc[l] + $signed(w_data[l*WW +: WW]) * a_data;
            // ---- stage 1: the neighbour is known; read the input value and the lanes' weights
            if (v0) begin
                if (kind == LINEAR)
                    a_addr <= in_base + p * K + k0;
                else
                    a_addr <= in_base + (srct0 * NODES + g_data) * C_IN + i0;
                w_addr <= w_base + ((kind == LINEAR) ? g : {31'b0, blk}) * K + k0;
            end
            v1 <= v0; z1 <= z0;

            case (state)
            IDLE: if (start) begin layer <= 0; done <= 1'b0; cycles <= 0; state <= DESC0; end
            DESC0: begin d_addr <= layer; state <= DESC2; end
            DESC2: begin
                for (l = 0; l < 20; l = l + 1) f[l] <= d_data[l*32 +: 32];
                p <= 0; g <= 0; e <= 0; o <= 0; tau <= 0; t <= 0;
                state <= (d_data[3:0] == AFFINE) ? AF0 : POS0;
            end

            // ---- one position (and one group of N channels): biases and shifts first
            POS0: begin
                if (kind == LINEAR)      begin bias_addr <= b_base + g; s_addr <= s_base + g; end
                else if (kind == HOP_CD) begin bias_addr <= b_base + blk; s_addr <= s_base; end
                else                     begin bias_addr <= b_base + ((tau == 0) ? 0 : (tau == T - 1) ? 4 : 2) + blk; s_addr <= s_base; end
                state <= POS1;
            end
            POS1: begin                                              // the reads arrive next clock
                k <= 0; h <= 0; slot <= 0; i <= 0; drain <= 0;
                state <= ACC;
            end
            ACC: begin
                if (k == 0) for (l = 0; l < N; l = l + 1) begin
                    acc[l] <= $signed(bias_data[l*AW +: AW]);
                    sh[l]  <= $signed(s_data[l*8 +: 8]);
                end
                // stage 0: issue input k
                v0 <= 1'b1; k0 <= k; i0 <= i; h0 <= h;
                if (kind == HOP_CD) begin
                    srct0 <= tau + h; z0 <= 1'b0;
                end else if (kind == HOP_DC) begin
                    srct0 <= h ? tau - 1 : tau;
                    z0 <= h ? (tau == 0) : (tau >= T_SRC);
                end else z0 <= 1'b0;
                g_addr <= g_base + t * 6 + slot;
                // the next k
                if (i == C_IN - 1) begin
                    i <= 0;
                    if (slot == 5) begin slot <= 0; h <= ~h; end else slot <= slot + 1'b1;
                end else i <= i + 1'b1;
                if (k == K - 1) begin state <= DRAIN; end
                k <= k + 1;
            end
            DRAIN: begin                                               // two clocks: the last input passes stages 1 and 2
                v0 <= 1'b0;
                drain <= drain + 1'b1;
                if (drain == 1) begin lane <= 0; state <= WR0; end
            end

            // ---- write the N lanes one by one
            WR0: begin
                sel <= acc[lane]; sel_sh <= sh[lane];
                state <= WR1;
            end
            WR1: begin
                rq <= saturate(rescale(sel, sel_sh), residual ? res_bits : sat_bits);
                t_addr <= t_base + table_index(saturate(rescale(sel, sel_sh), sat_bits), sat_bits);
                b_addr <= res_base + ((kind == LINEAR) ? p * C_out + g * N : (tau * NODES + t) * N) + lane;
                state <= WR2;
            end
            WR2: begin                                                 // the table and the old stream arrive now
                b_addr  <= out_base + ((kind == LINEAR) ? p * C_out + g * N : (tau * NODES + t) * N) + lane;
                old_q = saturate(rescale($signed(b_data), res_shift), res_bits);    // sign-extended to AW bits
                sum   = rq;
                sum   = sum + old_q;
                b_wdata <= use_silu ? t_data : residual ? saturate(sum, res_bits) : rq;
                b_we    <= 1'b1;
                if (lane == N - 1) state <= NEXT;
                else begin lane <= lane + 1'b1; state <= WR0; end
            end
            NEXT: begin
                if (kind == LINEAR) begin
                    if (g == G - 1) begin
                        g <= 0;
                        if (p == P - 1) state <= FINISH; else begin p <= p + 1; state <= POS0; end
                    end else begin g <= g + 1; state <= POS0; end
                end else begin
                    if (t == NODES - 1) begin
                        t <= 0;
                        if (tau == T - 1) state <= FINISH; else begin tau <= tau + 1'b1; state <= POS0; end
                    end else begin t <= t + 1'b1; state <= POS0; end
                end
            end

            // ---- AFFINE: one value at a time
            AF0: begin a_addr <= in_base + e; c_addr <= chan_base + o; state <= AF2; end
            AF2: begin                                                 // the value and its channel's a, c, shift
                t_addr <= t_base + table_index(saturate(rescale($signed(c_data[WW-1:0]) * a_data + $signed(c_data[WW +: AW]),
                                                                $signed(c_data[WW+AW +: 8])), sat_bits), sat_bits);
                state <= AF3;
            end
            AF3: begin
                b_addr <= out_base + e; b_wdata <= t_data; b_we <= 1'b1;
                if (e == P * K - 1) state <= FINISH;
                else begin e <= e + 1; o <= (o == K - 1) ? 0 : o + 1; state <= AF0; end
            end

            FINISH: begin
                if (layer == n_layers - 1) begin done <= 1'b1; state <= IDLE; end
                else begin layer <= layer + 1'b1; state <= DESC0; end
            end
            default: state <= IDLE;
            endcase
        end
    end
endmodule
