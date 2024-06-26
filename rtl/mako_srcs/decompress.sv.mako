############################################################
## decompress.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for decompress.sv
############################################################
<%!
    import math
%>\

<%
    num_unique_inputs = int(info["num_inputs"] / info["bits_per_input"])
    compressed_input_size = math.ceil(math.log2(info["bits_per_input"]+1))
    compressed_size = compressed_input_size * num_unique_inputs
    uncompressed_size = info["num_inputs"]
    bus_cycles = math.ceil(compressed_size / config["bus_width"])
    target_cycles = bus_cycles if (config["throughput"] <= 0) else config["throughput"]
    num_decompression_units = math.ceil(num_unique_inputs / max((target_cycles - 1), 1)) #// "-1" since the decompression unit adds a cycle of latency
    decompression_cycles = math.ceil(num_unique_inputs / num_decompression_units) + 1    #// "+1" for the same reason
    
    array_input_width = num_decompression_units*compressed_input_size
    array_output_width = num_decompression_units*info["bits_per_input"]

    padded_input_size = array_input_width * (decompression_cycles-1)
    padded_output_size = array_output_width * (decompression_cycles-1)

    inp_shr_size = array_input_width * (decompression_cycles-2)
%>\
module decompression_unit (
    clk, rst,
    inp,
    outp
);
    input  clk;
    input  rst;

    input  [${compressed_input_size-1}:0] inp;
    output [${info["bits_per_input"]-1}:0] outp;
    
    reg [${compressed_input_size-1}:0] r_inp;

    genvar g;

    generate for (g = 0; g < ${info["bits_per_input"]}; g = g + 1) begin: gen_decompress
        assign outp[g] = (r_inp > g);
    end endgenerate

    always_ff @(posedge clk) begin
        r_inp <= inp;
    end
endmodule

module decompression_block (
    clk, rst,
    inp_vld, outp_vld,
    inp_stall, outp_stall,
    inp,
    outp
);
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    input  inp_stall;
    output outp_stall;

    input  [${compressed_size-1}:0]   inp;
    output [${uncompressed_size-1}:0] outp;

    //reg  [${uncompressed_size-1}:0] pingpong_buffer [1:0];
    reg  [${padded_output_size-1}:0] pingpong_buffer [1:0];
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

    reg  [${math.ceil(math.log2(decompression_cycles))-1}:0] state;
% if decompression_cycles > 2:
    reg  [${inp_shr_size-1}:0]                               inp_shr;
% endif
    wire [${padded_input_size-1}:0]                          padded_inp;
    wire [${array_input_width-1}:0]                          array_inps; 
    wire [${array_output_width-1}:0]                         array_outps;

    
    assign front_buffer_id = r_front_buffer_id ^ (front_buffer_done && back_buffer_done);
    assign front_buffer_done = (outp_vld && !inp_stall) || (last_dispatch == r_front_buffer_id);
    assign back_buffer_done = (inp_vld && (state == ${decompression_cycles-1})) || (last_fill == !r_front_buffer_id);

    assign outp_vld = r_outp_vld;
    assign outp_stall = inp_vld && !(front_buffer_done && back_buffer_done);
    assign outp = pingpong_buffer[r_front_buffer_id][${uncompressed_size-1}:0]; 

% if padded_input_size != compressed_size:
    assign padded_inp = {${padded_input_size-compressed_size}'bx, inp};
% else:
    assign padded_inp = inp;
% endif
% if decompression_cycles > 2:
    assign array_inps = (state == 0) ? padded_inp[${array_input_width-1}:0] : inp_shr[${array_input_width-1}:0];
% else:
    assign array_inps = padded_inp[${array_input_width-1}:0];
% endif

    generate for (g = 0; g < ${num_decompression_units}; g = g + 1) begin: gen_decompression_units
        decompression_unit decomp (
            .clk(clk), .rst(rst),
            .inp(array_inps[g*${compressed_input_size}+:${compressed_input_size}]),
            .outp(array_outps[g*${info["bits_per_input"]}+:${info["bits_per_input"]}])
        );
    end endgenerate
    
    always_ff @(posedge clk) begin
        if (rst) begin
            r_front_buffer_id <= 1'b0;
            r_outp_vld <= 1'b0;
            last_dispatch <= 1'b0;
            last_fill <= 1'b0;
            state <= 0;
        end else begin
            r_front_buffer_id <= front_buffer_id;
            r_outp_vld <= inp_stall || (front_buffer_done && back_buffer_done);
            if (outp_vld && !inp_stall) begin
                last_dispatch <= r_front_buffer_id;
            end
            if (inp_vld && (last_fill == r_front_buffer_id)) begin
                if (state != 0) begin
                    pingpong_buffer[!r_front_buffer_id][${padded_output_size-1}-:${array_output_width}] <= array_outps;
% if decompression_cycles > 2:
                    pingpong_buffer[!r_front_buffer_id][${padded_output_size-array_output_width-1}:0] <= pingpong_buffer[!r_front_buffer_id][${padded_output_size-1}:${array_output_width}];
% endif
                end
                if (state == ${decompression_cycles-1}) begin
                    last_fill <= !r_front_buffer_id;
                end
                state <= (state == ${decompression_cycles-1}) ? 0 : state + 1;
% if decompression_cycles > 2:
                inp_shr <= (state == 0) ? padded_inp[${padded_input_size-1}-:${inp_shr_size}] : (inp_shr >> ${array_input_width});
% endif
            end
        end
    end
endmodule

