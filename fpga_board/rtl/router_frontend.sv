// 124-byte packet: 117 syndrome bytes (13 x 72 bits), then
// iterations u16 LE, converged u8, clipped posterior Q8.8 u16 LE,
// correction weight u16 LE. Only syndrome popcounts enter the model.
module router_frontend (
    input  logic clk,
    input  logic rst,
    input  logic byte_valid,
    input  logic [7:0] byte_data,
    output logic x_valid,
    output logic [17*16-1:0] x_flat
);
    logic [3:0] cycle = 0;
    logic [3:0] position = 0;
    logic tail;
    assign tail = cycle == 13;

    function automatic logic [3:0] ones8(input logic [7:0] b);
        ones8 = {3'b0,b[0]} + {3'b0,b[1]} + {3'b0,b[2]} + {3'b0,b[3]}
              + {3'b0,b[4]} + {3'b0,b[5]} + {3'b0,b[6]} + {3'b0,b[7]};
    endfunction

    always_ff @(posedge clk) begin
        x_valid <= 1'b0;
        if (rst) begin
            cycle <= 0;
            position <= 0;
            x_flat <= '0;
        end else if (byte_valid) begin
            if (!tail) begin
                if (position == 0)
                    x_flat[(cycle+4)*16 +: 16] <= ones8(byte_data);
                else
                    x_flat[(cycle+4)*16 +: 16] <= x_flat[(cycle+4)*16 +: 16] + ones8(byte_data);
                if (position == 8) begin
                    position <= 0;
                    cycle <= cycle + 1'b1;
                end else position <= position + 1'b1;
            end else begin
                case (position)
                    0: x_flat[0*16 +: 8] <= byte_data;
                    1: x_flat[0*16+8 +: 8] <= byte_data;
                    2: x_flat[1*16 +: 16] <= {15'b0,byte_data[0]};
                    3: x_flat[2*16 +: 8] <= byte_data;
                    4: x_flat[2*16+8 +: 8] <= byte_data;
                    5: x_flat[3*16 +: 8] <= byte_data;
                    6: x_flat[3*16+8 +: 8] <= byte_data;
                endcase
                if (position == 6) begin
                    position <= 0;
                    cycle <= 0;
                    x_valid <= 1'b1;
                end else position <= position + 1'b1;
            end
        end
    end
endmodule
