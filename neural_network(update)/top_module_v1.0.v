`timescale 1ns/1ps

module memory_W1_b1_init #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 784,
    parameter HIDDEN_SIZE = 128
)(
    // No clk or rst_n needed if it's just for initial block loading ROM content
    output reg signed [DATA_WIDTH-1:0] W1_data_out [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1],
    output reg signed [DATA_WIDTH-1:0] b1_data_out [0:HIDDEN_SIZE-1]
);

    initial begin
        $display("Initializing W1 and b1 memories...");
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W1.mem", W1_data_out);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b1.mem", b1_data_out);
        $display("W1 and b1 memories initialized.");
    end
endmodule

`timescale 1ns/1ps

module memory_W2_b2_init #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 128, // This is the HIDDEN_SIZE from the perspective of top_module
    parameter OUTPUT_SIZE = 10
)(
    // No clk or rst_n needed if it's just for initial block loading ROM content
    output reg signed [DATA_WIDTH-1:0] W2_data_out [0:INPUT_SIZE-1][0:OUTPUT_SIZE-1],
    output reg signed [DATA_WIDTH-1:0] b2_data_out [0:OUTPUT_SIZE-1]
);

    initial begin
        $display("Initializing W2 and b2 memories...");
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W2.mem", W2_data_out);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b2.mem", b2_data_out);
        $display("W2 and b2 memories initialized.");
    end

endmodule

// top_module.v (Modified)

`timescale 1ns/1ps

module top_module #(
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 128,
    parameter OUTPUT_SIZE = 10,
    parameter DATA_WIDTH = 16
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],
    output valid_out,
    output signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1],
    output [3:0] predicted_label
);

    // Wires for memory data (outputs from memory init modules)
    wire signed [DATA_WIDTH-1:0] w1_mem_data_internal [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1];
    wire signed [DATA_WIDTH-1:0] b1_mem_data_internal [0:HIDDEN_SIZE-1];
    wire signed [DATA_WIDTH-1:0] w2_mem_data_internal [0:HIDDEN_SIZE-1][0:OUTPUT_SIZE-1]; // Note: Input size for W2 is HIDDEN_SIZE
    wire signed [DATA_WIDTH-1:0] b2_mem_data_internal [0:OUTPUT_SIZE-1];

    // Instantiate memory initialization modules
    memory_W1_b1_init #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE)
    ) u_memory_W1_b1 (
        .W1_data_out(w1_mem_data_internal),
        .b1_data_out(b1_mem_data_internal)
    );

    memory_W2_b2_init #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(HIDDEN_SIZE), // This is the input dimension for the second layer's weights
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_memory_W2_b2 (
        .W2_data_out(w2_mem_data_internal),
        .b2_data_out(b2_mem_data_internal)
    );

    // Instantiate the neural_network_core module
    neural_network_core #(
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_neural_network_core_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_data(input_data),

        // Pass memory data to the core
        .w1_mem_data_in(w1_mem_data_internal),
        .b1_mem_data_in(b1_mem_data_internal),
        .w2_mem_data_in(w2_mem_data_internal),
        .b2_mem_data_in(b2_mem_data_internal),

        // Connect outputs from the core
        .valid_out(valid_out),
        .final_out(final_out),
        .predicted_label(predicted_label)
    );

endmodule


// input_layer.v (Modified)
`timescale 1ns/1ps

module input_layer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 784,
    parameter HIDDEN_SIZE = 128
)(
    input  clk,
    input  rst_n,
    input  valid_in,
    input  signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],
    // NEW INPUTS for weights and biases
    input  signed [DATA_WIDTH-1:0] W1_in [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1],
    input  signed [DATA_WIDTH-1:0] b1_in [0:HIDDEN_SIZE-1],
    output reg valid_out,
    output reg signed [DATA_WIDTH-1:0] output_data [0:HIDDEN_SIZE-1]
);
    localparam PARALLEL_FACTOR = 4;
    localparam FIXED_NUM_CHUNKS = 16;
    localparam FIXED_CHUNK_SIZE = 49;
    localparam ACC_WIDTH = 24;
    localparam CHUNK_CNT_WIDTH = 4;
    localparam HIDDEN_CNT_WIDTH = 7;
    localparam DELTA_VAL_WIDTH = 23;

    localparam K_LOOP_CNT_WIDTH = $clog2(FIXED_CHUNK_SIZE);
    reg [K_LOOP_CNT_WIDTH-1:0] k_loop_cnt;
    reg [HIDDEN_CNT_WIDTH-1:0] j_proc_cnt;

    // REMOVED: reg signed [DATA_WIDTH-1:0] W1 [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1];
    // REMOVED: reg signed [DATA_WIDTH-1:0] b1 [0:HIDDEN_SIZE-1];
    reg signed [ACC_WIDTH-1:0] acc [0:HIDDEN_SIZE-1];

    localparam [3:0] S_IDLE             = 4'b0000,
                     S_INIT_ALL         = 4'b0001,
                     S_CALC_DELTA_SETUP = 4'b0010,
                     S_CALC_DELTA_LOOP  = 4'b0011,
                     S_ACCUMULATE_DELTA = 4'b0100,
                     S_NEXT_HIDDEN_OR_CHUNK = 4'b0101,
                     S_ADD_BIAS_SETUP   = 4'b0110,
                     S_ADD_BIAS_LOOP    = 4'b0111,
                     S_RELU_SETUP       = 4'b1000,
                     S_RELU_LOOP        = 4'b1001,
                     S_DONE             = 4'b1010;
    reg [3:0] state, next_state;

    reg [CHUNK_CNT_WIDTH-1:0] chunk_cnt;
    reg [HIDDEN_CNT_WIDTH-1:0] hidden_node_cnt;
    reg signed [DELTA_VAL_WIDTH-1:0] current_delta_sum_for_node;

    reg signed [2*DATA_WIDTH-1:0]     product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DATA_WIDTH:0]         scaled_product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DELTA_VAL_WIDTH-1:0]  sum_of_scaled_products;

    integer current_i_base;

    // REMOVED: initial begin ... $readmemh ... end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            valid_out         <= 1'b0;
            chunk_cnt         <= 0;
            hidden_node_cnt   <= 0;
            k_loop_cnt        <= 0;
            j_proc_cnt        <= 0;
            current_delta_sum_for_node <= 0;
            for (integer signed j_rst = 0; j_rst < HIDDEN_SIZE; j_rst = j_rst + 1) begin
                acc[j_rst]         <= '0;
                output_data[j_rst] <= '0;
            end
        end else begin
            state <= next_state;
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        next_state <= S_INIT_ALL;
                    end else begin
                        next_state <= S_IDLE;
                    end
                end
                S_INIT_ALL: begin
                    for (integer signed j_init = 0; j_init < HIDDEN_SIZE; j_init = j_init + 1) begin
                        acc[j_init] <= '0;
                    end
                    chunk_cnt         <= 0;
                    hidden_node_cnt   <= 0;
                    next_state        <= S_CALC_DELTA_SETUP;
                end
                S_CALC_DELTA_SETUP: begin
                    k_loop_cnt <= 0;
                    current_delta_sum_for_node <= 0;
                    next_state <= S_CALC_DELTA_LOOP;
                end
                S_CALC_DELTA_LOOP: begin
                    current_i_base = (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt;
                    sum_of_scaled_products = '0;
                    for (integer signed m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (k_loop_cnt + m < FIXED_CHUNK_SIZE) begin
                            // MODIFIED: W1 -> W1_in
                            product_seq[m] = input_data[current_i_base + m] * W1_in[current_i_base + m][hidden_node_cnt];
                            scaled_product_seq[m] = product_seq[m] >>> 12;
                            sum_of_scaled_products = sum_of_scaled_products +
                                                     {{(DELTA_VAL_WIDTH - (DATA_WIDTH+1)){scaled_product_seq[m][DATA_WIDTH]}}, scaled_product_seq[m]};
                        end
                    end
                    current_delta_sum_for_node <= current_delta_sum_for_node + sum_of_scaled_products;

                    if (k_loop_cnt + PARALLEL_FACTOR < FIXED_CHUNK_SIZE) begin
                        k_loop_cnt <= k_loop_cnt + PARALLEL_FACTOR;
                        next_state <= S_CALC_DELTA_LOOP;
                    end else begin
                        next_state <= S_ACCUMULATE_DELTA;
                    end
                end
                S_ACCUMULATE_DELTA: begin
                    acc[hidden_node_cnt] <= acc[hidden_node_cnt] +
                                           {{(ACC_WIDTH - DELTA_VAL_WIDTH){current_delta_sum_for_node[DELTA_VAL_WIDTH-1]}}, current_delta_sum_for_node};
                    next_state <= S_NEXT_HIDDEN_OR_CHUNK;
                end
                S_NEXT_HIDDEN_OR_CHUNK: begin
                    if (hidden_node_cnt < HIDDEN_SIZE - 1) begin
                        hidden_node_cnt <= hidden_node_cnt + 1;
                        next_state <= S_CALC_DELTA_SETUP;
                    end else begin
                        hidden_node_cnt <= 0;
                        if (chunk_cnt < FIXED_NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1;
                            next_state <= S_CALC_DELTA_SETUP;
                        end else begin
                            next_state <= S_ADD_BIAS_SETUP;
                        end
                    end
                end
                S_ADD_BIAS_SETUP: begin
                    j_proc_cnt <= 0;
                    next_state <= S_ADD_BIAS_LOOP;
                end
                S_ADD_BIAS_LOOP: begin
                    for (integer signed m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (j_proc_cnt + m < HIDDEN_SIZE) begin
                            // MODIFIED: b1 -> b1_in
                            acc[j_proc_cnt + m] <= acc[j_proc_cnt + m] + {{ (ACC_WIDTH - DATA_WIDTH){b1_in[j_proc_cnt + m][DATA_WIDTH-1]} }, b1_in[j_proc_cnt + m]};
                        end
                    end

                    if (j_proc_cnt + PARALLEL_FACTOR < HIDDEN_SIZE) begin
                        j_proc_cnt <= j_proc_cnt + PARALLEL_FACTOR;
                        next_state <= S_ADD_BIAS_LOOP;
                    end else begin
                        next_state <= S_RELU_SETUP;
                    end
                end
                S_RELU_SETUP: begin
                    j_proc_cnt <= 0;
                    next_state <= S_RELU_LOOP;
                end
                S_RELU_LOOP: begin
                    for (integer signed m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (j_proc_cnt + m < HIDDEN_SIZE) begin
                            if (acc[j_proc_cnt + m][ACC_WIDTH-1]) begin
                                output_data[j_proc_cnt + m] <= '0;
                            end else begin
                                if (acc[j_proc_cnt + m] > ((1 << (DATA_WIDTH-1)) - 1) ) begin
                                    output_data[j_proc_cnt + m] <= ((1 << (DATA_WIDTH-1)) - 1);
                                end else begin
                                    output_data[j_proc_cnt + m] <= acc[j_proc_cnt + m][DATA_WIDTH-1:0];
                                end
                            end
                        end
                    end

                    if (j_proc_cnt + PARALLEL_FACTOR < HIDDEN_SIZE) begin
                        j_proc_cnt <= j_proc_cnt + PARALLEL_FACTOR;
                        next_state <= S_RELU_LOOP;
                    end else begin
                        next_state <= S_DONE;
                    end
                end
                S_DONE: begin
                    valid_out <= 1'b1;
                    next_state <= S_IDLE;
                end
                default: begin
                    next_state <= S_IDLE;
                end
            endcase
        end
    end
endmodule

// output_layer.v (Modified)
`timescale 1ns/1ps

module output_layer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 128,
    parameter OUTPUT_SIZE = 10
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],
    // NEW INPUTS for weights and biases
    input signed [DATA_WIDTH-1:0] W2_in [0:INPUT_SIZE-1][0:OUTPUT_SIZE-1],
    input signed [DATA_WIDTH-1:0] b2_in [0:OUTPUT_SIZE-1],
    output valid_out,
    output signed [DATA_WIDTH-1:0] output_data [0:OUTPUT_SIZE-1],
    output [3:0] predicted_label
);

    localparam PARALLEL_FACTOR_OL = 4;
    localparam ACC_OL_WIDTH = DATA_WIDTH + 8;

    // REMOVED: reg signed [DATA_WIDTH-1:0] W2 [0:INPUT_SIZE-1][0:OUTPUT_SIZE-1];
    // REMOVED: reg signed [DATA_WIDTH-1:0] b2 [0:OUTPUT_SIZE-1];
    reg signed [ACC_OL_WIDTH-1:0] acc [0:OUTPUT_SIZE-1];
    reg valid_softmax_in_internal;

    localparam OL_S_IDLE      = 3'b000,
               OL_S_LOAD_BIAS = 3'b001,
               OL_S_MAC_LOOP  = 3'b010,
               OL_S_NEXT_NODE = 3'b011,
               OL_S_SOFTMAX   = 3'b100;

    reg [2:0] ol_state, ol_next_state;

    localparam J_CNT_WIDTH = $clog2(OUTPUT_SIZE);
    localparam I_CNT_WIDTH = $clog2(INPUT_SIZE);
    reg [J_CNT_WIDTH-1:0] j_cnt;
    reg [I_CNT_WIDTH-1:0] i_cnt;

    reg signed [2*DATA_WIDTH-1:0] ol_product [0:PARALLEL_FACTOR_OL-1];
    reg signed [DATA_WIDTH:0]     ol_scaled_product [0:PARALLEL_FACTOR_OL-1];
    reg signed [ACC_OL_WIDTH-1:0] ol_sum_of_scaled_products;

    // REMOVED: initial begin ... $readmemh ... end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ol_state <= OL_S_IDLE;
            valid_softmax_in_internal <= 1'b0;
            j_cnt <= 0;
            i_cnt <= 0;
            for (integer k_rst = 0; k_rst < OUTPUT_SIZE; k_rst = k_rst + 1) begin
                acc[k_rst] <= 0;
            end
        end else begin
            ol_state <= ol_next_state;
            valid_softmax_in_internal <= 1'b0;

            case (ol_state)
                OL_S_IDLE: begin
                    if (valid_in) begin
                        j_cnt <= 0;
                        ol_next_state <= OL_S_LOAD_BIAS;
                    end else begin
                        ol_next_state <= OL_S_IDLE;
                    end
                end
                OL_S_LOAD_BIAS: begin
                    // MODIFIED: b2 -> b2_in
                    acc[j_cnt] <= {{ (ACC_OL_WIDTH - DATA_WIDTH){b2_in[j_cnt][DATA_WIDTH-1]} }, b2_in[j_cnt]};
                    i_cnt <= 0;
                    ol_next_state <= OL_S_MAC_LOOP;
                end
                OL_S_MAC_LOOP: begin
                    ol_sum_of_scaled_products = '0;
                    for (integer signed m = 0; m < PARALLEL_FACTOR_OL; m = m + 1) begin
                        if (i_cnt + m < INPUT_SIZE) begin
                            // MODIFIED: W2 -> W2_in
                            ol_product[m] = input_data[i_cnt + m] * W2_in[i_cnt + m][j_cnt];
                            ol_scaled_product[m] = ol_product[m] >>> 12;
                            ol_sum_of_scaled_products = ol_sum_of_scaled_products +
                                                       {{ (ACC_OL_WIDTH-(DATA_WIDTH+1)){ol_scaled_product[m][DATA_WIDTH]} }, ol_scaled_product[m]};
                        end
                    end
                    acc[j_cnt] <= acc[j_cnt] + ol_sum_of_scaled_products;

                    if (i_cnt + PARALLEL_FACTOR_OL < INPUT_SIZE) begin
                        i_cnt <= i_cnt + PARALLEL_FACTOR_OL;
                        ol_next_state <= OL_S_MAC_LOOP;
                    end else begin
                        ol_next_state <= OL_S_NEXT_NODE;
                    end
                end
                OL_S_NEXT_NODE: begin
                    if (j_cnt < OUTPUT_SIZE - 1) begin
                        j_cnt <= j_cnt + 1;
                        ol_next_state <= OL_S_LOAD_BIAS;
                    end else begin
                        ol_next_state <= OL_S_SOFTMAX;
                    end
                end
                OL_S_SOFTMAX: begin
                    valid_softmax_in_internal <= 1'b1;
                    ol_next_state <= OL_S_IDLE;
                end
                default: ol_next_state <= OL_S_IDLE;
            endcase
        end
    end

    softmax_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .SIZE(OUTPUT_SIZE)
    ) u_softmax_unit (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_softmax_in_internal),
        .acc_data(acc),
        .valid_out(valid_out),
        .softmax_result(output_data),
        .predicted_label(predicted_label)
    );

endmodule

// Include the softmax_unit module as it was before
module softmax_unit #(
    parameter DATA_WIDTH = 16, 
    parameter SIZE = 10
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH+8-1:0] acc_data [0:SIZE-1], 
    output reg valid_out,
    output reg signed [DATA_WIDTH-1:0] softmax_result [0:SIZE-1], 
    output reg [3:0] predicted_label
);

    localparam LABEL_WIDTH = $clog2(SIZE); 
    localparam SIGNED_MAX_VAL = (1 << (DATA_WIDTH-1)) - 1;
    integer i_comb;

    localparam SM_IDLE = 1'b0,
               SM_DONE = 1'b1;
    reg state_sm;
    reg signed [DATA_WIDTH+8-1:0] temp_current_max_val_reg;
    reg [LABEL_WIDTH-1:0] temp_max_idx_reg;
    reg signed [DATA_WIDTH-1:0] temp_softmax_result_regs [0:SIZE-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out       <= 1'b0;
            predicted_label <= {LABEL_WIDTH{1'b0}};
            state_sm        <= SM_IDLE;
            for (integer k_rst = 0; k_rst < SIZE; k_rst = k_rst + 1) begin
                softmax_result[k_rst] <= 0;
            end
            temp_current_max_val_reg <= 0;
            temp_max_idx_reg <= 0;
        end else begin
            valid_out <= 1'b0; 

            case (state_sm)
                SM_IDLE: begin
                    if (valid_in) begin
                        reg signed [DATA_WIDTH+8-1:0] current_max_val_comb;
                        reg [LABEL_WIDTH-1:0]         max_idx_comb;
                        // temp_softmax_result_regs is already a reg, no need for _comb version if assigned directly for next cycle

                        current_max_val_comb = acc_data[0];
                        max_idx_comb         = 0;

                        for (i_comb = 1; i_comb < SIZE; i_comb = i_comb + 1) begin
                            if (acc_data[i_comb] > current_max_val_comb) begin
                                current_max_val_comb = acc_data[i_comb];
                                max_idx_comb         = i_comb;
                            end
                        end
                        
                        temp_current_max_val_reg <= current_max_val_comb; 
                        temp_max_idx_reg         <= max_idx_comb;

                        for (i_comb = 0; i_comb < SIZE; i_comb = i_comb + 1) begin
                            if (i_comb == max_idx_comb) begin 
                                temp_softmax_result_regs[i_comb] = SIGNED_MAX_VAL;
                            end else begin
                                temp_softmax_result_regs[i_comb] = 0;
                            end
                        end
                        
                        predicted_label <= max_idx_comb; 
                        softmax_result  <= temp_softmax_result_regs; 

                        state_sm <= SM_DONE;
                    end else begin
                        state_sm <= SM_IDLE;
                    end
                end
                SM_DONE: begin
                    valid_out <= 1'b1;
                    state_sm  <= SM_IDLE;
                end
                default: begin
                    state_sm <= SM_IDLE;
                end
            endcase
        end
    end
endmodule

`timescale 1ns/1ps

module neural_network_core #(
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 128,
    parameter OUTPUT_SIZE = 10,
    parameter DATA_WIDTH = 16
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],

    // Inputs for weights and biases from memory init modules
    input signed [DATA_WIDTH-1:0] w1_mem_data_in [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1],
    input signed [DATA_WIDTH-1:0] b1_mem_data_in [0:HIDDEN_SIZE-1],
    input signed [DATA_WIDTH-1:0] w2_mem_data_in [0:HIDDEN_SIZE-1][0:OUTPUT_SIZE-1],
    input signed [DATA_WIDTH-1:0] b2_mem_data_in [0:OUTPUT_SIZE-1],

    output valid_out,
    output signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1],
    output [3:0] predicted_label
);

    // Internal wires connecting input_layer and output_layer
    wire valid_mid_core;
    wire signed [DATA_WIDTH-1:0] hidden_out_core [0:HIDDEN_SIZE-1];

    // Instantiate input_layer
    input_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE)
    ) u_input_layer_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_data(input_data),
        .W1_in(w1_mem_data_in),     // Connect W1 from core input
        .b1_in(b1_mem_data_in),     // Connect b1 from core input
        .valid_out(valid_mid_core),
        .output_data(hidden_out_core)
    );

    // Instantiate output_layer (which contains softmax_unit)
    output_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(HIDDEN_SIZE), // Input to this layer is the output of hidden layer
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_output_layer_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_mid_core),
        .input_data(hidden_out_core),
        .W2_in(w2_mem_data_in),     // Connect W2 from core input
        .b2_in(b2_mem_data_in),     // Connect b2 from core input
        .valid_out(valid_out),       // Connect to core's output
        .output_data(final_out),     // Connect to core's output
        .predicted_label(predicted_label) // Connect to core's output
    );

endmodule