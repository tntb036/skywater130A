`timescale 1ns/1ps

module tb_top_module;

    parameter INPUT_SIZE = 784;
    parameter HIDDEN_SIZE = 128;
    parameter OUTPUT_SIZE = 10;
    parameter DATA_WIDTH = 16;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1];
    reg [15:0] check_label [0:99];
    wire valid_out;
    wire signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1];
    wire [3:0] predicted_label;
    string filename_img ;
	string filename_label ;

    // Instantiate DUT (Device Under Test)
    top_module #(
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_data(input_data),
        .valid_out(valid_out),
        .final_out(final_out),
        .predicted_label(predicted_label)
    );
    integer img_num;
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end

    integer max_o, checkin, j, accuraccy, i;
    // Stimulus
    initial begin
        $monitor("predicted_label: %d",predicted_label);
        $dumpfile("output.vcd"); // Open the VCD file for writing
        $dumpvars(0, tb_top_module);
        // Reset sequence
        accuraccy = 0;
        img_num = 10;
        for (j = 0; j < img_num;j = j + 1) begin
            rst_n = 0;
            valid_in = 0;
            #20;
            rst_n = 1;
            #20;
            // Load input image & label
            filename_img = $sformatf("/VNCHIP/WORK/thienbao/Neural_network/data/mem_test/img/input_img_%0d.mem", j);
			$display("Reading file: %s", filename_img);
			$readmemh(filename_img, input_data);
			filename_label = $sformatf("/VNCHIP/WORK/thienbao/Neural_network/data/mem_test/label/label_%0d.mem",j);
			$display("Reading file: %s", filename_label);
			$readmemh(filename_label, check_label);
            // Apply input
            @(posedge clk);
            valid_in <= 1;
            @(posedge clk);
            valid_in <= 0;
            
            // Wait for valid_out
            wait (valid_out == 1);
            checkin = 0;
            max_o = final_out[0];
            // Display output
            $display("Prediction result:");
            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                $display("final_out[%0d] = %0d", i, final_out[i]);
                if (max_o < final_out[i]) begin
                    max_o = final_out[i];
                    checkin = i;
                end
            end
            if (check_label[0] == checkin) begin
                accuraccy = accuraccy + 1;
            end
            $display("label_real: %d", check_label[0]);
            $display("label measure: %d",check_label[0]);
        end
        $display("accuraccy:%d/%d",accuraccy,img_num);
        #100;
        $finish;
    end

endmodule