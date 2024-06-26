############################################################
## wisard.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for wisard.sv
############################################################
<%!
    import math
%>\
`include "sv_srcs/model_parameters.svh"

module max_idx(
    clk, rst, inp_vld, outp_vld,
    inp_values,
    max_index
);
<%
    entry_size = 16
    input_entries = info["num_classes"]
    index_size = math.ceil(math.log2(input_entries))
%>\
    input   clk;
    input   rst;
    input   inp_vld;
    output  outp_vld;

    input   [${entry_size-1}:0] inp_values  [${input_entries-1}:0];
// synthesis translate_off
`ifndef SYNTHESIS
    initial $vcdplusmemon(inp_values);
`endif
// synthesis translate_on
    output  [${index_size-1}:0] max_index;

    genvar g;

    wire    [${entry_size-1}:0] reduce_layer0   [${input_entries-1}:0];
    wire    [${index_size-1}:0] index_layer0    [${input_entries-1}:0];
    wire    [0:0]               vld_layer0;
    assign reduce_layer0 = inp_values;
    generate for (g = 0; g < ${input_entries}; g = g + 1) begin: gen_max_layer0
        assign index_layer0[g] = g;
    end endgenerate
    assign vld_layer0 = inp_vld;

<%
    idx = 1
    layer_inputs = input_entries
%>\
% while layer_inputs > 1:
<%
    layer_outputs = math.ceil(layer_inputs / 2)
    odd_input = not (layer_inputs / 2).is_integer()
%>\
    reg     [${entry_size-1}:0] reduce_layer${idx}      [${layer_outputs-1}:0];
    reg     [${index_size-1}:0] index_layer${idx}       [${layer_outputs-1}:0];
    wire    [${entry_size-1}:0] w_reduce_layer${idx}    [${layer_outputs-1}:0];
    wire    [${index_size-1}:0] w_index_layer${idx}     [${layer_outputs-1}:0];
    reg     [0:0]               vld_layer${idx};
// synthesis translate_off
`ifndef SYNTHESIS
        initial $vcdplusmemon(reduce_layer${idx});
        initial $vcdplusmemon(index_layer${idx});
`endif
// synthesis translate_on
    // In the event of a tie, the lower-indexed (left-hand) term should win
    generate for (g = 0; g < ${layer_outputs - (1 if odd_input else 0)}; g = g + 1) begin: gen_max_layer${idx}
        assign w_reduce_layer${idx}[g] = (reduce_layer${idx-1}[2*g] >= reduce_layer${idx-1}[2*g+1]) ? reduce_layer${idx-1}[2*g] : reduce_layer${idx-1}[2*g+1];
        assign w_index_layer${idx}[g] = (reduce_layer${idx-1}[2*g] >= reduce_layer${idx-1}[2*g+1]) ? index_layer${idx-1}[2*g] : index_layer${idx-1}[2*g+1];
    end endgenerate
% if odd_input:
    assign w_reduce_layer${idx}[${layer_outputs-1}] = reduce_layer${idx-1}[${layer_inputs-1}];
    assign w_index_layer${idx}[${layer_outputs-1}] = index_layer${idx-1}[${layer_inputs-1}];
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
            index_layer${i} <= w_index_layer${i};
            vld_layer${i} <= vld_layer${i-1};
% endfor
        end
    end

    assign max_index = index_layer${idx-1}[0];
    assign outp_vld = vld_layer${idx-1};

endmodule

module wisard_ensemble(
    clk, rst, inp_vld, outp_vld, stall,
    inp,
    outp
);
<%
    input_size = info["num_inputs"]
    output_size = math.ceil(math.log2(info["num_classes"]))
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    output stall;

    input  [${input_size-1}:0]     inp;
    output [${output_size-1}:0]    outp;

% for submodel_idx, submodel_info in enumerate(info["submodel_info"]):
<%
    filter_inpw = submodel_info["num_filter_inputs"]
    filter_addrw = int(math.log2(submodel_info["num_filter_entries"]))
    num_filters = math.ceil(info["num_inputs"] / filter_inpw)
%>\
    wire [${(num_filters*filter_addrw)-1}:0] model${submodel_idx}_hash;
    wire hash${submodel_idx}_outp_vld;
% endfor
    wire lookup_outp_vld;
    wire [15:0] activations [${info["num_classes"]-1}:0];

    // Instantiate hash blocks
% for submodel_idx in range(len(info["submodel_info"])):
<% stall_signal = "stall" if submodel_idx == 0 else "/* Signal intentionally unconnected */" %>\
    hash_block_${submodel_idx} hb${submodel_idx} (
        .clk(clk), .rst(rst), .inp_vld(inp_vld), .outp_vld(hash${submodel_idx}_outp_vld), .stall(${stall_signal}),
        .inp(inp),
        .hashed(model${submodel_idx}_hash)
    );
% endfor
    
    // Instantiate ensemble discriminators
% for discrim_idx in range(info["num_classes"]):
<% outp_vld_signal = "lookup_outp_vld" if discrim_idx == 0 else "/* Signal intentionally unconnected */" %>\
    ensemble_discriminator_${discrim_idx} discrim${discrim_idx} (
        .clk(clk), .rst(rst), .outp_vld(${outp_vld_signal}),
% for i in range(len(info["submodel_info"])):
        .model${i}_inp_vld(hash${i}_outp_vld), .model${i}_hashed_inp(model${i}_hash),
% endfor
        .activation(activations[${discrim_idx}])
    );
% endfor
    
    max_idx max(
        .clk(clk), .rst(rst), .inp_vld(lookup_outp_vld), .outp_vld(outp_vld),
        .inp_values(activations),
        .max_index(outp)
    );
endmodule

