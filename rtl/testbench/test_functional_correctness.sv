// test_functional_correctness.sv
// Author: Zachary Susskind (ZSusskind@utexas.edu)
// Main ULEEN Synopsys testbench

`include "sv_srcs/config.svh"

// To reduce simulation time, just run the first 10k samples
// For a dataset with <10k samples, you'll need to reduce it
// Sorry, it's a bit of a hack - couldn't come up with a better way to get
// the number of entries in the dataset
`define SAMPLES 10000
`define SAMPLE_SIZE_BYTES ((`INPUT_SIZE_BITS + 7) / 8)
`define SAMPLE_BUS_CYCLES ((`INPUT_SIZE_BITS+`INPUT_BUS_WIDTH-1)/`INPUT_BUS_WIDTH)
`define FILE_DATA_BYTES (((`SAMPLE_SIZE_BYTES + 1) * `SAMPLES) + 6)

module Top;
    reg  [0:0] clk;
    reg  [0:0] rst;
    reg  [0:0] inp_vld;
    wire [0:0] outp_vld;
    wire [0:0] stall;

    wire [`INPUT_BUS_WIDTH-1:0] inp;
    wire [3:0] outp;

    wire [(`SAMPLE_BUS_CYCLES*`INPUT_BUS_WIDTH)-1:0] padded_sample;
    wire [7:0] label;
    
    integer i, j;
    integer correct, total;
    string dset_fname;
    integer file;
    reg [(`FILE_DATA_BYTES*8)-1:0] fdata;
    integer f_sample_size_bytes;
    integer f_unused;
    integer f_bits_per_input; 
    reg [`INPUT_SIZE_BITS-1:0] input_samples [`SAMPLES-1:0];
    reg [7:0]                  input_labels  [`SAMPLES-1:0];

    generate if ((`SAMPLE_BUS_CYCLES*`INPUT_BUS_WIDTH) > `INPUT_SIZE_BITS) begin
        assign padded_sample[(`SAMPLE_BUS_CYCLES*`INPUT_BUS_WIDTH)-1:`INPUT_SIZE_BITS] = 'bx;
    end endgenerate
    assign padded_sample[`INPUT_SIZE_BITS-1:0] = input_samples[i];
    assign inp = padded_sample[`INPUT_BUS_WIDTH*j+:`INPUT_BUS_WIDTH];
    assign label = input_labels[total];

    real accuracy;
    initial begin
        if ($value$plusargs("DSET=%s", dset_fname)) begin
            $display("Using dataset file %s", dset_fname);
        end else begin
            $fatal("No dataset specified; use +DSET=<fname>");
        end
        $display("Reading dataset...");
        file = $fopen(dset_fname, "rb");
        $fread(fdata, file);
        f_sample_size_bytes = fdata[(`FILE_DATA_BYTES*8)-1-:32];
        $display(f_sample_size_bytes);
        f_unused = fdata[(`FILE_DATA_BYTES*8)-33-:8];
        $display(f_unused);
        f_bits_per_input = fdata[(`FILE_DATA_BYTES*8)-41-:8];
        $display(f_bits_per_input);
        for (i = 0; i < `SAMPLES; i = i + 1) begin
            input_samples[i] = fdata[((`FILE_DATA_BYTES - (((`SAMPLE_SIZE_BYTES + 1) * i) + 6)) * 8) - 1 -: `INPUT_SIZE_BITS];
            input_labels[i] = fdata[((`FILE_DATA_BYTES - (((`SAMPLE_SIZE_BYTES + 1) * (i+1)) + 5)) * 8) - 1 -: 8];
        end
        i = 0;
        j = 0;
        fork
        begin
            clk = 0;
            forever begin
                #5 begin end
                clk = !clk;    
            end
        end
        begin
            rst = 1'b1;
            #16 begin end

            rst = 1'b0;
            inp_vld = 1'b1;
            forever begin
                #0 begin end
                if (!stall) begin
                    #10 begin end
                    j = j + 1;
                    if (j == `SAMPLE_BUS_CYCLES) begin
                        j = 0;
                        i = i + 1;
                    end
                end else begin
                    #10 begin end
                end
                if (i == `SAMPLES) break;
            end
            inp_vld = 1'b0;
        end
        begin
            correct = 0;
            total = 0;
            while (total < `SAMPLES) begin
                if (outp_vld) begin
                    if (outp == label) correct = correct + 1;
                    total = total + 1;
                    if ((total % 1000) == 0) $display(total);
                end
                #10 begin end
            end
            accuracy = correct * 100;
            accuracy = accuracy / total;
            $display("Model accuracy: %f", accuracy);
            
            $finish;
        end
        join
    end
    
    initial begin // timeout
        #100000000 begin end
        $fatal("Simulation timed out");
    end

    initial begin
        $vcdplusfile("test_functional_correctness.dump.vpd");
        $vcdpluson(1, Top.dut);
    end // initial begin

    device_top dut (
        .clk(clk), .rst(rst), .inp_vld(inp_vld), .outp_vld(outp_vld), .stall(stall),
        .inp(inp),
        .outp(outp)
    );
endmodule

