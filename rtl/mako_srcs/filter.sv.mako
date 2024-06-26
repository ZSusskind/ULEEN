############################################################
## filter.sv.mako
## Author: Zachary Susskind (ZSusskind@utexas.edu)
##
## Generator file for filter.sv
############################################################
<%!
    import math
%>\

<%
    lut_sizes = sorted(list(set(int(math.log2(i["num_filter_entries"])) for i in info["submodel_info"])))
%>\
% for size in lut_sizes:
module lut${size} (
    address,
    data
);
    parameter reg [${(1<<size)-1}:0] DATA = ${1<<size}'b0;

    input  [${size-1}:0] address;
    output [0:0]         data;

    wire [${(1<<size)-1}:0] data_arr;

    assign data_arr = DATA;
    assign data = data_arr[address];
endmodule

% endfor

<%
    bloom_filter_specs = sorted(list(set((\
        int(math.log2(i["num_filter_entries"])),\
        i["num_filter_hashes"]\
        ) for i in info["submodel_info"])))
%>\
% for size, hashes in bloom_filter_specs:
<%
    max_state = hashes - 1
%>\
module filter${size}_${hashes}hash (
    clk, rst, inp_vld, outp_vld,
    hashed_inp,
    result
);
    parameter reg [${(1<<size)-1}:0] DATA = ${1<<size}'b0;

    input   clk;
    input   rst;
    input   inp_vld;
    output  outp_vld;

    input   [${size-1}:0] hashed_inp;
    output  [0:0]         result;

% if max_state > 0:
    reg  [${math.ceil(math.log2(max_state+1))-1}:0] state;
    wire [${math.ceil(math.log2(max_state+1))-1}:0] vld_next_state;
% endif
    reg  [0:0] accumulator;
    wire [0:0] vld_next_accumulator;
    reg  [0:0] r_outp_vld;
    wire [0:0] next_r_outp_vld;

    wire[0:0] table_data;
    lut${size} #(.DATA(DATA)) lut(.address(hashed_inp), .data(table_data));

% if max_state > 0:
    assign vld_next_state = (state < ${max_state}) ? state+1 : 0;
    assign vld_next_accumulator = (state == 0) ? table_data : (accumulator & table_data);
    assign next_r_outp_vld = (state == ${max_state}) && inp_vld;
% else:
    assign vld_next_accumulator = table_data;
    assign next_r_outp_vld = inp_vld;
% endif

    always_ff @(posedge clk) begin
        if (rst) begin
% if max_state > 0:
            state <= 0;
% endif
            r_outp_vld <= 0;
        end else begin
            if (inp_vld) begin
% if max_state > 0:
                state <= vld_next_state;
% endif
                accumulator <= vld_next_accumulator;
            end
            r_outp_vld <= next_r_outp_vld;
        end
    end

    assign outp_vld = r_outp_vld;
    assign result = accumulator;
endmodule

% endfor

