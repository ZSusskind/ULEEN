############################################################
## interface.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for interface.sv
############################################################
<%!
    import math
%>\

module device_interface(
    clk, rst,
    inp_vld, outp_vld,
    inp_stall, outp_stall,
    inp,
    outp
);
<%
    input_size = config["bus_width"]
    if not config["compressed_input"]:
        model_input_size = info["num_inputs"]
    else:
        model_input_size = int(info["num_inputs"] / info["bits_per_input"]) * math.ceil(math.log2(info["bits_per_input"]+1))
    output_size = model_input_size
    cycles_to_fill = math.ceil(model_input_size / input_size)

    padded_size = cycles_to_fill * input_size
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    input  inp_stall;
    output outp_stall;

    input  [${input_size-1}:0]  inp;
    output [${output_size-1}:0] outp;

    reg [${padded_size-1}:0] pingpong_buffer [1:0];
// synthesis translate_off
`ifndef SYNTHESIS
    initial $vcdplusmemon(pingpong_buffer);
`endif
// synthesis translate_on
    reg  [0:0] r_front_buffer_id;
    reg  [0:0] r_outp_vld;
    reg  [0:0] last_dispatch;
    reg  [0:0] last_fill;
    wire [0:0] front_buffer_id;
    wire [0:0] front_buffer_done;
    wire [0:0] back_buffer_done;

    genvar g;

% if cycles_to_fill > 1:
    reg  [${math.ceil(math.log2(cycles_to_fill))-1}:0] state;
% endif
    
    assign front_buffer_id = r_front_buffer_id ^ (front_buffer_done && back_buffer_done);
    assign front_buffer_done = (outp_vld && !inp_stall) || (last_dispatch == r_front_buffer_id);
% if cycles_to_fill > 1:
    assign back_buffer_done = (inp_vld && (state == ${cycles_to_fill-1})) || (last_fill == !r_front_buffer_id);
% else:
    assign back_buffer_done = inp_vld || (last_fill == !r_front_buffer_id);
% endif

    assign outp_vld = r_outp_vld;
    assign outp_stall = inp_vld && back_buffer_done && !front_buffer_done;
    assign outp = pingpong_buffer[r_front_buffer_id][${model_input_size-1}:0];
    
    always_ff @(posedge clk) begin
        if (rst) begin
            r_front_buffer_id <= 1'b0;
            r_outp_vld <= 1'b0;
            last_dispatch <= 1'b0;
            last_fill <= 1'b0;
% if cycles_to_fill > 1:
            state <= 0;
% endif
        end else begin
            r_front_buffer_id <= front_buffer_id;
            r_outp_vld <= inp_stall || (front_buffer_done && back_buffer_done);
            if (outp_vld && !inp_stall) begin
                last_dispatch <= r_front_buffer_id;
            end
            if (inp_vld && (last_fill == r_front_buffer_id)) begin
% if cycles_to_fill > 1:
                pingpong_buffer[!r_front_buffer_id][${padded_size-1}-:${input_size}] <= inp;
                pingpong_buffer[!r_front_buffer_id][${padded_size-input_size-1}:0] <= pingpong_buffer[!r_front_buffer_id][${padded_size-1}:${input_size}];
                if (state == ${cycles_to_fill-1}) begin
                    last_fill <= !r_front_buffer_id;
                end
                state <= (state == ${cycles_to_fill-1}) ? 0 : state + 1;
% else:
                pingpong_buffer[!r_front_buffer_id] <= inp;
                last_fill <= !r_front_buffer_id;
%endif
            end
        end
    end
endmodule

