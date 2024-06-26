############################################################
## discriminator.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for discriminator.sv
############################################################
<%!
    import math
%>\
`include "sv_srcs/model_parameters.svh"


% for submodel_idx, submodel_info in enumerate(info["submodel_info"]):
% for discrim_idx in range(info["num_classes"]):
module discriminator_${submodel_idx}_${discrim_idx} (
    clk, rst, inp_vld, outp_vld,
    hashed_inp,
    filter_outps
);
<%
    filter_addrw = int(math.log2(submodel_info["num_filter_entries"]))
    filter_hashes = submodel_info["num_filter_hashes"]
    total_filters = int(len(submodel_info["input_order"]) / submodel_info["num_filter_inputs"])
    input_size = total_filters * filter_addrw
    num_nonsparse_filters = len(submodel_info["nonsparse_filter_idxs"][discrim_idx])
    output_size = math.ceil(math.log2(num_nonsparse_filters+1))
%>
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;

    input  [${input_size-1}:0]            hashed_inp;
    output [${num_nonsparse_filters-1}:0] filter_outps;

<%
    filter_mname = f"filter{filter_addrw}_{filter_hashes}hash"
    nonsparse_idx = 0
%>\
% for filter_idx in range(total_filters):
% if filter_idx in submodel_info["nonsparse_filter_idxs"][discrim_idx]:
<%
    data_idx = filter_idx * filter_addrw 
    outp_vld_signal = "outp_vld" if nonsparse_idx == 0 else "/* Signal intentionally unconnected */"
%>\
    ${filter_mname} #(.DATA(`FDATA_${submodel_idx}_${discrim_idx}_${filter_idx})) filter${filter_idx} (
        .clk(clk), .rst(rst), .inp_vld(inp_vld), .outp_vld(${outp_vld_signal}),
        .hashed_inp(hashed_inp[${data_idx}+:${filter_addrw}]), .result(filter_outps[${nonsparse_idx}])
    );
<% nonsparse_idx += 1 %>\
% else:
    // Pruned filter${filter_idx}
% endif
% endfor

endmodule
% endfor
% endfor

module popcount (
    clk, rst, inp_vld, outp_vld,
    inp,
    sum
);
<%
    popcount_input_size = sum(len(i["nonsparse_filter_idxs"][0]) for i in info["submodel_info"])
    popcount_output_size = math.ceil(math.log2(popcount_input_size+1))
%>\
    input   clk;
    input   rst;
    input   inp_vld;
    output  outp_vld;

    input   [${popcount_input_size-1}:0]     inp;
    output  [${popcount_output_size-1}:0]    sum;

    genvar g;

    wire    [0:0]   reduce_layer0 [${popcount_input_size-1}:0];
    generate for (g = 0; g < ${popcount_input_size}; g = g + 1) begin: gen_popcount_layer0
        assign reduce_layer0[g] = inp[g];
    end endgenerate

    wire    [0:0]   vld_layer0;
    assign vld_layer0 = inp_vld;

<%
    idx = 1
    layer_inputs = popcount_input_size
%>\
% while layer_inputs > 1:
<%
    layer_outputs = math.ceil(layer_inputs / 2)
    odd_input = not (layer_inputs / 2).is_integer()
%>\
    wire    [${idx}:0]  w_reduce_layer${idx}    [${layer_outputs-1}:0];
    reg     [${idx}:0]  reduce_layer${idx}      [${layer_outputs-1}:0];
// synthesis translate_off
`ifndef SYNTHESIS
    initial $vcdplusmemon(reduce_layer${idx});
`endif
// synthesis translate_on
    reg     [0:0]       vld_layer${idx};
    generate for (g = 0; g < ${layer_outputs - (1 if odd_input else 0)}; g = g + 1) begin: gen_popcount_layer${idx}
        assign w_reduce_layer${idx}[g] = reduce_layer${idx-1}[2*g] + reduce_layer${idx-1}[2*g+1];
    end endgenerate
% if odd_input:
    assign w_reduce_layer${idx}[${layer_outputs-1}] = reduce_layer${idx-1}[${layer_inputs-1}];
% endif

<%
    layer_inputs = layer_outputs
    idx += 1
%>\
% endwhile

    always_ff @(posedge clk) begin
        if (rst) begin
% for i in range(1, idx):
            vld_layer${i} <= 0;
% endfor
        end else begin
% for i in range(1, idx):
            reduce_layer${i} <= w_reduce_layer${i};
            vld_layer${i} <= vld_layer${i-1};
% endfor
        end
    end

    assign sum = reduce_layer${idx-1}[0][${popcount_output_size-1}:0];
    assign outp_vld = vld_layer${idx-1};

endmodule

// What if different modules have different latencies (#s of filters)?
// Top-level FSM (above this) should delay giving valid signals for "shallower" filters such that all receive the last set of inputs at the same time
// ALSO: Need bias term
<% use_bias = (max(info["bias"]) > 0) %>\
% for discrim_idx in range(info["num_classes"]):
module ensemble_discriminator_${discrim_idx} (
    clk, rst, outp_vld,
% for i in range(len(info["submodel_info"])):
    model${i}_inp_vld,
    model${i}_hashed_inp,
% endfor
    activation
);
    input  clk;
    input  rst;
    output outp_vld;
<% output_idxs = [0] %>\
% for i, submodel_info in enumerate(info["submodel_info"]):
<%
    filter_addrw = int(math.log2(submodel_info["num_filter_entries"]))
    total_filters = int(len(submodel_info["input_order"]) / submodel_info["num_filter_inputs"])
    input_size = total_filters * filter_addrw
    output_idxs.append(output_idxs[-1] + len(submodel_info["nonsparse_filter_idxs"][discrim_idx]))
%>\
    input  [0:0]               model${i}_inp_vld;
    input  [${input_size-1}:0] model${i}_hashed_inp;
% endfor
    output [15:0]                                     activation;

    wire [0:0]                         model_outps_vld;
    wire [${output_idxs[-1]-1}:0]      model_outputs;
    wire [0:0]                         popcount_outp_vld;
    wire [${popcount_output_size-1}:0] popcount_output;

% if use_bias:
    reg [0:0]  r_outp_vld;
    reg [15:0] r_activation;

    assign outp_vld = r_outp_vld;
    assign activation = r_activation;
% else:
    assign outp_vld = popcount_outp_vld;
    assign activation = {${16-popcount_output_size}'b0, popcount_output};
% endif

% for submodel_idx in range(len(info["submodel_info"])):
<%
    num_model_outps = len(info["submodel_info"][submodel_idx]["nonsparse_filter_idxs"][discrim_idx]) 
    outp_vld_signal = "model_outps_vld" if submodel_idx == 0 else "/* Signal intentionally unconnected */"
%>\
    discriminator_${submodel_idx}_${discrim_idx} discrim${submodel_idx} (
        .clk(clk), .rst(rst), .inp_vld(model${submodel_idx}_inp_vld), .outp_vld(${outp_vld_signal}),
        .hashed_inp(model${submodel_idx}_hashed_inp),
        .filter_outps(model_outputs[${output_idxs[submodel_idx]}+:${num_model_outps}])
    );
% endfor

    popcount count (
        clk, rst, model_outps_vld, popcount_outp_vld,
        model_outputs,
        popcount_output
    );

% if use_bias:
    always_ff @(posedge clk) begin
        r_outp_vld <= popcount_outp_vld;
        r_activation <= popcount_output + 16'd${info["bias"][discrim_idx]};
    end
% endif
endmodule

% endfor

