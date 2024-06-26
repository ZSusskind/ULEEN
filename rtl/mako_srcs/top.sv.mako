############################################################
## top.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for top.sv
############################################################
<%!
    import math
%>\

module device_top (
    clk, rst, inp_vld, outp_vld, stall,
    inp,
    outp
);
<%
    input_size = config["bus_width"]
    output_size = math.ceil(math.log2(info["num_classes"]))
    if not config["compressed_input"]:
        sample_size = info["num_inputs"]
    else:
        sample_size = int(info["num_inputs"] / info["bits_per_input"]) * math.ceil(math.log2(info["bits_per_input"]+1))
    decompressed_sample_size = info["num_inputs"]
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    output stall;

    input  [${input_size-1}:0]     inp;
    output [${output_size-1}:0]    outp;

    wire                      interface_outp_vld;
    wire [${sample_size-1}:0] interface_outp;

    wire                                   decompress_outp_vld;
    wire                                   decompress_outp_stall;
    wire [${decompressed_sample_size-1}:0] decompress_outp;

    wire model_stall;


    device_interface ifc (
        .clk(clk), .rst(rst),
        .inp_vld(inp_vld), .outp_vld(interface_outp_vld),
        .inp_stall(decompress_outp_stall), .outp_stall(stall),
        .inp(inp),
        .outp(interface_outp)
    );

% if config["compressed_input"]:
    decompression_block decompress ( 
        .clk(clk), .rst(rst),
        .inp_vld(interface_outp_vld), .outp_vld(decompress_outp_vld),
        .inp_stall(model_stall), .outp_stall(decompress_outp_stall),
        .inp(interface_outp),
        .outp(decompress_outp)
    );
% else:
    assign decompress_outp_vld = interface_outp_vld;
    assign decompress_outp_stall = model_stall;
    assign decompress_outp = interface_outp;
% endif

    wisard_ensemble model (
        .clk(clk), .rst(rst),
        .inp_vld(decompress_outp_vld), .outp_vld(outp_vld),
        .stall(model_stall),
        .inp(decompress_outp),
        .outp(outp)
    );
endmodule

