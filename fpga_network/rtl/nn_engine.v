// A small engine that walks the network layer by layer (fpga_network/README.md, step 4).
//
// N multiply-add units ("lanes") work on N output channels of one position at once. On every clock one input value
// is read from the activation memory and given to all lanes, each with its own weight (the weight memory's word holds
// N weights). After K inputs, each lane's sum gets its bias (preloaded), then the lanes are written, one per clock
// through a three-stage pipeline: round to the output's scale (add half, arithmetic shift right; or shift left),
// saturate, optionally look the result up in a SiLU table, optionally add the old residual stream (rescaled,
// saturated; read through the input port, which is free while writing), and write.
//
// A table of layer descriptions (one DW-bit word per layer) tells the controller what to do. Four kinds:
//   LINEAR  out[p, g*N + l] = f(sum_k W[g][k][l] * in[p*K + k] + bias[g][l])           (down, up)
//   HOP_CD  checks -> data qubits: target (cycle c, data node t) reads, for 6 neighbour slots s, the c_in values of
//           check node nbr[t][s] in round c ("same") and round c + 1 ("after"): K = 2 * 6 * c_in
//   HOP_DC  data qubits -> checks: target (round r, check t) reads cycle r ("same", if r < T_src) and cycle r - 1
//           ("after", if r >= 1); a missing input counts as zero; the first, middle and last rounds have own biases
//   AFFINE  per channel o, for every value: f(a[o] * in[e] + c[o]): the normalisation before a SiLU
//   TSUM    out[s, t, c] = sum over T rounds from s of in[r, t, c], rescaled and saturated: the readout's time sums
// The two target blocks of a collecting step (X/Z checks, left/right data qubits) have their own weights and biases.
// Flags: bit 0 SiLU table, bit 1 residual sum, bit 2 strided input (LINEAR: in[k * stride + p]), bit 3 transposed
// output (LINEAR: out[(g*N + l) * P + p]), bit 4 raw output: the 48-bit sum as three 16-bit words (the final logits).
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
    localparam [3:0] LINEAR = 0, HOP_CD = 1, HOP_DC = 2, AFFINE = 3, TSUM = 4;

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
    wire        strided   = f[1][2];
    wire        transpose = f[1][3];
    wire        raw       = f[1][4];
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
    wire [31:0] g_base    = f[16], chan_base = f[17], stride = f[18];

    // ---------------------------------------------------------------- state
    localparam [4:0] IDLE = 0, DESC2 = 3, POS0 = 4, POS1 = 5, ACC = 6, DRAIN = 7, DESC0 = 1,
                     WR0 = 8, WR1 = 9, WR2 = 10, NEXT = 11, AF0 = 12, AF2 = 14, AF3 = 15, FINISH = 16,
                     WR3 = 17, WR4 = 18, TS0 = 19, TSA = 20, TSB = 21, TSW = 22, WRP = 23, AFP = 24;
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

    reg signed [AW-1:0] old_q, sum, tsum;
    reg        tsv;
    reg [7:0]  r, sl;                       // TSUM: round inside the sum, and which sum
    reg [31:0] rest;                        // TSUM: node * CH + channel
    reg signed [AW-1:0] raw_acc;
    // where a lane of this position and group goes (and where its old residual value is)
    function [31:0] offset(input [7:0] ln);
        offset = (kind == LINEAR) ? (transpose ? (g * N + ln) * P + p : p * C_out + g * N + ln)
                                  : (tau * NODES + t) * C_out + g * N + ln;
    endfunction
    wire [31:0] pos_off = offset(lane);
    // the write pipeline: A takes lane li's sum, B rounds it and starts the table and residual reads, C writes
    reg [7:0]  li, lane_a, lane_b;
    reg        va, vb;
    reg signed [AW-1:0] sum_a; reg signed [7:0] sh_a;
    reg signed [VW-1:0] rq_b;
    // the AFFINE pipeline: A reads the value and its channel, B rounds and reads the table, C writes
    reg        fa_v, fb_v; reg [31:0] e_a, e_b; reg issued;
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
            if (tsv) tsum <= tsum + a_data;
            tsv <= (state == TSA);
            // ---- stage 1: the neighbour is known; read the input value and the lanes' weights
            if (v0) begin
                if (kind == LINEAR)
                    a_addr <= strided ? in_base + k0 * stride + p : in_base + p * K + k0;
                else
                    a_addr <= in_base + (srct0 * NODES + g_data) * C_IN + i0;
                w_addr <= w_base + ((kind == LINEAR) ? g : {31'b0, blk} * G + g) * K + k0;
            end
            v1 <= v0; z1 <= z0;

            case (state)
            IDLE: if (start) begin layer <= 0; done <= 1'b0; cycles <= 0; state <= DESC0; end
            DESC0: begin d_addr <= layer; state <= DESC2; end
            DESC2: begin
                for (l = 0; l < 20; l = l + 1) f[l] <= d_data[l*32 +: 32];
                p <= 0; g <= 0; e <= 0; o <= 0; tau <= 0; t <= 0;
                state <= (d_data[3:0] == AFFINE) ? AFP : (d_data[3:0] == TSUM) ? TS0 : POS0;
                issued <= 1'b0; fa_v <= 1'b0; fb_v <= 1'b0;
                sl <= 0; rest <= 0;
            end

            // ---- one position (and one group of N channels): biases and shifts first
            POS0: begin
                if (kind == LINEAR)      begin bias_addr <= b_base + g; s_addr <= s_base + g; end
                else if (kind == HOP_CD) begin bias_addr <= b_base + {31'b0, blk} * G + g; s_addr <= s_base + g; end
                else                     begin bias_addr <= b_base + (((tau == 0) ? 0 : (tau == T - 1) ? 4 : 2) + blk) * G + g;
                                               s_addr <= s_base + g; end
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
                if (drain == 1) begin lane <= 0; li <= 0; va <= 1'b0; vb <= 1'b0; state <= raw ? WR0 : WRP; end
            end

            // ---- write the N lanes, one per clock
            WRP: begin
                if (vb) begin                                          // stage C: the table and the old stream are here
                    old_q = saturate(rescale($signed(a_data), res_shift), res_bits);
                    sum   = rq_b;
                    sum   = sum + old_q;
                    b_addr  <= out_base + offset(lane_b);
                    b_wdata <= use_silu ? t_data : residual ? saturate(sum, res_bits) : rq_b;
                    b_we    <= 1'b1;
                end
                vb <= va;                                              // stage B
                if (va) begin
                    rq_b   <= saturate(rescale(sum_a, sh_a), residual ? res_bits : sat_bits);
                    t_addr <= t_base + table_index(saturate(rescale(sum_a, sh_a), sat_bits), sat_bits);
                    a_addr <= res_base + offset(lane_a);
                    lane_b <= lane_a;
                end
                va <= (li < N);                                        // stage A
                if (li < N) begin sum_a <= acc[li]; sh_a <= sh[li]; lane_a <= li; li <= li + 1'b1; end
                if (li == N && !va && !vb) state <= NEXT;
            end

            // ---- write the N lanes one by one
            WR0: begin
                sel <= acc[lane]; sel_sh <= sh[lane];
                state <= WR1;
            end
            WR1: begin
                rq <= saturate(rescale(sel, sel_sh), residual ? res_bits : sat_bits);
                t_addr <= t_base + table_index(saturate(rescale(sel, sel_sh), sat_bits), sat_bits);
                b_addr <= res_base + pos_off;
                state <= WR2;
            end
            WR2: begin                                                 // the table and the old stream arrive now
                old_q = saturate(rescale($signed(b_data), res_shift), res_bits);    // sign-extended to AW bits
                sum   = rq;
                sum   = sum + old_q;
                if (raw) begin
                    b_addr <= out_base + pos_off * 3; b_wdata <= sel[15:0]; b_we <= 1'b1; state <= WR3;
                end else begin
                    b_addr  <= out_base + pos_off;
                    b_wdata <= use_silu ? t_data : residual ? saturate(sum, res_bits) : rq;
                    b_we    <= 1'b1;
                    if (lane == N - 1) state <= NEXT;
                    else begin lane <= lane + 1'b1; state <= WR0; end
                end
            end
            WR3: begin b_addr <= out_base + pos_off * 3 + 1; b_wdata <= sel[31:16]; b_we <= 1'b1; state <= WR4; end
            WR4: begin
                b_addr <= out_base + pos_off * 3 + 2; b_wdata <= sel[47:32]; b_we <= 1'b1;
                if (lane == N - 1) state <= NEXT;
                else begin lane <= lane + 1'b1; state <= WR0; end
            end
            NEXT: begin
                if (g != G - 1) begin g <= g + 1; state <= POS0; end
                else begin
                    g <= 0;
                    if (kind == LINEAR) begin
                        if (p == P - 1) state <= FINISH; else begin p <= p + 1; state <= POS0; end
                    end else if (t == NODES - 1) begin
                        t <= 0;
                        if (tau == T - 1) state <= FINISH; else begin tau <= tau + 1'b1; state <= POS0; end
                    end else begin t <= t + 1'b1; state <= POS0; end
                end
            end

            // ---- TSUM: out[(sl * NODES * CH) + rest] = sum over r < T of in[((sl + r) * NODES * CH) + rest]
            TS0: begin tsum <= 0; r <= 0; state <= TSA; end
            TSA: begin                                                 // one read per clock; the sum follows one clock later
                a_addr <= in_base + (sl + r) * NODES * K + rest;
                if (r == T - 1) state <= TSB; else r <= r + 1'b1;
            end
            TSB: state <= TSW;
            TSW: begin
                b_addr <= out_base + sl * NODES * K + rest;
                b_wdata <= saturate(rescale(tsum, res_shift), sat_bits); b_we <= 1'b1;
                if (rest == NODES * K - 1) begin
                    rest <= 0;
                    if (sl == P - 1) state <= FINISH; else begin sl <= sl + 1'b1; state <= TS0; end
                end else begin rest <= rest + 1; state <= TS0; end
            end

            // ---- AFFINE: one value per clock through a three-stage pipeline
            AFP: begin
                if (fb_v) begin b_addr <= out_base + e_b; b_wdata <= t_data; b_we <= 1'b1; end      // stage C
                fb_v <= fa_v;                                                                    // stage B
                if (fa_v) begin
                    t_addr <= t_base + table_index(saturate(rescale($signed(c_data[WW-1:0]) * a_data + $signed(c_data[WW +: AW]),
                                                                    $signed(c_data[WW+AW +: 8])), sat_bits), sat_bits);
                    e_b <= e_a;
                end
                fa_v <= !issued;                                                                 // stage A
                if (!issued) begin
                    a_addr <= in_base + e; c_addr <= chan_base + o; e_a <= e;
                    if (e == P * K - 1) issued <= 1'b1;
                    else begin e <= e + 1; o <= (o == K - 1) ? 0 : o + 1; end
                end
                if (issued && !fa_v && !fb_v) state <= FINISH;
            end
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
