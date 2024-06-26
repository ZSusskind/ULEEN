############################################################
## hash.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for hash.sv
############################################################
<%!
    import math
%>\
`include "sv_srcs/model_parameters.svh"

<%
    hash_unit_specs = sorted(list(set((\
        i["num_filter_inputs"],
        int(math.log2(i["num_filter_entries"]))\
        ) for i in info["submodel_info"])))
%>\
% for filter_inpw, filter_addrw in hash_unit_specs:
module hash_unit_${filter_inpw}x${filter_addrw} (
    clk, rst, inp_vld, outp_vld,
    input_value, hash_values,
    hash_result
);
<%
    hash_input_set_size = filter_inpw * filter_addrw;


    intermediate_buffer = True #// If True, inserts an intermediate pipe stage - for critical path problems
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;

    input  [${filter_inpw-1}:0]       input_value;
    input  [${hash_input_set_size-1}:0] hash_values;
    output [${filter_addrw-1}:0]           hash_result;

    genvar g;

    wire    [${hash_input_set_size-1}:0]    w_gated_values;
    generate for (g = 0; g < ${filter_inpw}; g = g + 1) begin: gen_gated_val
        assign w_gated_values[g*${filter_addrw}+:${filter_addrw}] = input_value[g] ? hash_values[g*${filter_addrw}+:${filter_addrw}] : 0;
    end endgenerate

% if intermediate_buffer:
    reg     [${hash_input_set_size-1}:0]    gated_values;
    reg     [0:0]                           gated_values_vld;
% else: 
    wire    [${hash_input_set_size-1}:0]    gated_values;
    wire    [0:0]                           gated_values_vld;
    assign gated_values = w_gated_values;
    assign gated_values_vld = inp_vld;
% endif

    // Assumption: XOR reduction can be performed in at most one cycle
    //  This appears to hold true for reasonably sized filters
    wire    [${filter_addrw-1}:0]  w_hash_result;
    generate for (g = 0; g < ${filter_addrw}; g = g + 1) begin: gen_hash_res
        assign w_hash_result[g:g] = 
% for i in range(filter_inpw-1):
            gated_values[${i*filter_addrw}+g] ^
% endfor
            gated_values[${(filter_inpw-1)*filter_addrw}+g];
    end endgenerate

    reg     [${filter_addrw-1}:0]  r_hash_result;
    reg     r_outp_vld;
    always_ff @(posedge clk) begin
        if (rst) begin
% if intermediate_buffer:
            gated_values_vld <= 0;
% endif
            r_outp_vld <= 0;
        end else begin
% if intermediate_buffer:
            gated_values <= w_gated_values;
            gated_values_vld <= inp_vld;
% endif
            r_hash_result <= w_hash_result;
            r_outp_vld <= gated_values_vld;
        end
    end 

    assign hash_result = r_hash_result;
    assign outp_vld = r_outp_vld;
endmodule

% endfor

<%
    if not config["compressed_input"]:
        input_data_size = info["num_inputs"]
    else:
        input_data_size = int(info["num_inputs"] / info["bits_per_input"]) * math.ceil(math.log2(info["bits_per_input"]+1))
    bus_cycles = math.ceil(input_data_size / config["bus_width"])
    target_cycles = bus_cycles if (config["throughput"] <= 0) else config["throughput"]
    max_num_hashes = max(m["num_filter_hashes"] for m in info["submodel_info"])
    target_hash_cycles = max(math.floor(target_cycles / max_num_hashes), 1)
%>\
// Target hash cycles: ${target_hash_cycles}
% for submodel_idx, submodel_info in enumerate(info["submodel_info"]):
module hash_block_${submodel_idx} (
    clk, rst, inp_vld, outp_vld, stall,
    inp,
    hashed
);
<%
    filter_inpw = submodel_info["num_filter_inputs"]
    filter_addrw = int(math.log2(submodel_info["num_filter_entries"]))
    num_hashes = submodel_info["num_filter_hashes"]

    num_filters = submodel_info.get("num_filters", math.ceil(info["num_inputs"] / filter_inpw))
    num_null_bits = submodel_info.get("num_null_bits", (num_filters * filter_inpw) - info["num_inputs"])
    num_hash_units = math.ceil(num_filters / target_hash_cycles)
    hash_steps = math.ceil(num_filters / num_hash_units)
    num_pad_bits = (((hash_steps*num_hash_units)-num_filters)*filter_inpw)
    delay_cycles = target_hash_cycles - hash_steps

    array_input_width = num_hash_units*filter_inpw
    array_output_width = num_hash_units*filter_addrw

    hash_unit_mname = f"hash_unit_{filter_inpw}x{filter_addrw}"

    hash_param_set_size = filter_addrw * filter_inpw
    hash_param_total_size = num_hashes * hash_param_set_size

    partial_buffer_size = ((hash_steps-1)*array_output_width)

    padded_input_size = hash_steps * array_input_width
    inp_shr_size = ((hash_steps-1) * array_input_width)
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    output stall;

    input  [${info["num_inputs"]-1}:0]      inp;
    output [${(num_filters*filter_addrw)-1}:0] hashed;

    genvar g;
 
    reg  [${math.ceil(math.log2(max_num_hashes+1))-1}:0]     hash_inp_count;
    reg  [${math.ceil(math.log2(target_hash_cycles+1))-1}:0] hash_inp_state;
    reg  [${math.ceil(math.log2(hash_steps+1))-1}:0]         hash_outp_state;

    //reg  r_stall;

    wire [${info["num_inputs"]+num_null_bits-1}:0] null_inp;
    wire [${(num_filters*filter_inpw)-1}:0]        reordered_inp;
    wire [${padded_input_size-1}:0]                padded_inp;
    wire [${(array_input_width)-1}:0]              array_inps;
    wire [${(array_output_width)-1}:0]             array_outps;
    wire [${(hash_steps*array_output_width)-1}:0]  padded_hash_outp;

    wire [${hash_param_total_size-1}:0] hash_params;
    wire [${hash_param_set_size-1}:0]   sel_hash_params;
% if hash_steps > 1:
    reg  [${partial_buffer_size-1}:0] partial_hash_buffer;
    reg  [${inp_shr_size-1}:0]        inp_shr;
% endif
    wire hash_inp_vld;
    wire hash_outp_vld;

% if num_null_bits > 0:
    assign null_inp = {${num_null_bits}'b0, inp};
% else:
    assign null_inp = inp;
% endif
    assign reordered_inp = {${", ".join([f"null_inp[{x}]" for x in list(reversed(submodel_info["input_order"]))])}};
% if num_pad_bits > 0:
    assign padded_inp = {${num_pad_bits}'bx, reordered_inp};
% else:
    assign padded_inp = reordered_inp;
% endif

% if hash_steps > 1:
    assign array_inps = (hash_inp_state <= ${delay_cycles}) ? padded_inp[${array_input_width-1}:0] : inp_shr[${array_input_width-1}:0];
    assign padded_hash_outp = {array_outps, partial_hash_buffer};
% else:
    assign array_inps = padded_inp;
    assign padded_hash_outp = array_outps;
% endif
    assign hash_params = `HPARAMS_${submodel_idx};
    assign sel_hash_params = (hash_inp_count >= ${max_num_hashes-num_hashes}) ?
        hash_params[(hash_inp_count-${max_num_hashes-num_hashes})*${hash_param_set_size}+:${hash_param_set_size}] : ${hash_param_set_size}'bx;

    assign hash_inp_vld = inp_vld && (hash_inp_count >= ${max_num_hashes - num_hashes}) && (hash_inp_state >= ${delay_cycles});
    
    ${hash_unit_mname} hash0 (
        .clk(clk), .rst(rst), .inp_vld(hash_inp_vld), .outp_vld(hash_outp_vld),
        .input_value(array_inps[0+:${filter_inpw}]), .hash_values(sel_hash_params),
        .hash_result(array_outps[0+:${filter_addrw}])
    );
    generate for (g = 1; g < ${num_hash_units}; g = g + 1) begin: gen_hash_units
        ${hash_unit_mname} hash(
            .clk(clk), .rst(rst), .inp_vld(hash_inp_vld), .outp_vld(/* Signal intentionally unconnected */),
            .input_value(array_inps[g*${filter_inpw}+:${filter_inpw}]), .hash_values(sel_hash_params),
            .hash_result(array_outps[g*${filter_addrw}+:${filter_addrw}])
        );
    end endgenerate
    
    assign outp_vld = hash_outp_vld && (hash_outp_state == ${hash_steps-1});
    //assign stall = r_stall;
    assign stall = inp_vld && !((hash_inp_count == ${max_num_hashes-1}) && (hash_inp_state == ${target_hash_cycles-1}));
    assign hashed = padded_hash_outp[${(num_filters*filter_addrw)-1}:0];


    always_ff @(posedge clk) begin
        if (rst) begin
            hash_inp_count <= 0;
            hash_inp_state <= 0;
            hash_outp_state <= 0;
            //r_stall <= 0;
        end else begin
            if (inp_vld) begin
                hash_inp_state <= (hash_inp_state == ${target_hash_cycles-1}) ? 0 : hash_inp_state+1;
                hash_inp_count <= (hash_inp_state == ${target_hash_cycles-1}) ? (
                    (hash_inp_count == ${max_num_hashes-1}) ? 0 : hash_inp_count+1
                    ) : hash_inp_count;
% if hash_steps > 1:
                inp_shr <= (hash_inp_state <= ${delay_cycles}) ? padded_inp[${padded_input_size-1}-:${inp_shr_size}] : (inp_shr >> ${array_input_width});
% endif
            end
% if hash_steps > 1:
            if (hash_outp_vld) begin
                if (hash_outp_state < ${hash_steps-1}) begin
                    partial_hash_buffer[${partial_buffer_size-1}-:${array_output_width}] <= array_outps;
% if hash_steps > 2:
                    partial_hash_buffer[${partial_buffer_size-array_output_width-1}:0] <= partial_hash_buffer[${partial_buffer_size-1}:${array_output_width}];
% endif
                end
                hash_outp_state <= (hash_outp_state == ${hash_steps-1}) ? 0 : hash_outp_state + 1;
            end
% endif
            //r_stall <= inp_vld && !((hash_inp_count == ${max_num_hashes-1}) && (hash_inp_state == ${target_hash_cycles-1}));
        end
    end
endmodule

% endfor

