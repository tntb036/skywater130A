`timescale 1ns/1ps
//****************************************************************************//
// Module: top_module
// Description: Top-level wrapper for the neural network accelerator.
//              It interfaces with the memory module and the main computational
//              core, while maintaining a stable I/O for the testbench.
//****************************************************************************//
module top_module #(
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 128,
    parameter OUTPUT_SIZE = 10,s
    parameter DATA_WIDTH = 16,
    parameter PARALLEL_FACTOR_L1 = 4,
    parameter PARALLEL_FACTOR_L2 = 4
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],
    output valid_out,
    output signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1], 
    output [3:0] predicted_label
);

    // --- Memory Module Instantiation ---
    wire signed [DATA_WIDTH-1:0] W1_from_mem [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1];
    wire signed [DATA_WIDTH-1:0] b1_from_mem [0:HIDDEN_SIZE-1];
    wire signed [DATA_WIDTH-1:0] W2_from_mem [0:HIDDEN_SIZE-1][0:OUTPUT_SIZE-1];
    wire signed [DATA_WIDTH-1:0] b2_from_mem [0:OUTPUT_SIZE-1];

    memory_module #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_memory_module (
        .W1_out(W1_from_mem),
        .b1_out(b1_from_mem),
        .W2_out(W2_from_mem),
        .b2_out(b2_from_mem)
    );

    // --- Wires for Core Communication ---
    // Wires to carry requests FROM core TO memory/slicing logic
    wire [($clog2(INPUT_SIZE))-1:0]  w1_req_base_idx_from_core;
    wire [($clog2(HIDDEN_SIZE))-1:0] w1_req_hidden_idx_from_core;
    wire [($clog2(HIDDEN_SIZE))-1:0] b1_req_base_idx_from_core;
    wire [($clog2(HIDDEN_SIZE))-1:0] w2_req_input_act_base_idx_from_core;
    wire [($clog2(OUTPUT_SIZE))-1:0] w2_req_output_node_idx_from_core;

    // Wires to carry selected chunks TO the core
    wire signed [DATA_WIDTH-1:0] input_data_chunk_to_core [0:PARALLEL_FACTOR_L1-1];
    wire signed [DATA_WIDTH-1:0] w1_chunk_to_core [0:PARALLEL_FACTOR_L1-1];
    wire signed [DATA_WIDTH-1:0] b1_chunk_to_core [0:PARALLEL_FACTOR_L1-1];
    wire signed [DATA_WIDTH-1:0] w2_chunk_to_core [0:PARALLEL_FACTOR_L2-1];
    wire signed [DATA_WIDTH-1:0] b2_element_to_core;

    // --- Data/Weight/Bias Slicing Logic (Generate Blocks) ---

    // Logic to select input_data chunk
    genvar m_input_data;
    generate
        for (m_input_data = 0; m_input_data < PARALLEL_FACTOR_L1; m_input_data = m_input_data + 1) begin : gen_input_data_chunk_sel
            assign input_data_chunk_to_core[m_input_data] = (w1_req_base_idx_from_core + m_input_data < INPUT_SIZE) ? 
                                                            input_data[w1_req_base_idx_from_core + m_input_data] : 
                                                            '0;
        end
    endgenerate

    // Logic to select W1 chunks
    genvar m_l1_w;
    generate
        for (m_l1_w = 0; m_l1_w < PARALLEL_FACTOR_L1; m_l1_w = m_l1_w + 1) begin : gen_w1_chunk_sel
            assign w1_chunk_to_core[m_l1_w] = (w1_req_base_idx_from_core + m_l1_w < INPUT_SIZE) ?
                                              W1_from_mem[w1_req_base_idx_from_core + m_l1_w][w1_req_hidden_idx_from_core] :
                                              '0;
        end
    endgenerate

    // Logic to select b1 chunks
    genvar m_l1_b;
    generate
        for (m_l1_b = 0; m_l1_b < PARALLEL_FACTOR_L1; m_l1_b = m_l1_b + 1) begin : gen_b1_chunk_sel
            assign b1_chunk_to_core[m_l1_b] = (b1_req_base_idx_from_core + m_l1_b < HIDDEN_SIZE) ?
                                              b1_from_mem[b1_req_base_idx_from_core + m_l1_b] :
                                              '0;
        end
    endgenerate

    // Logic to select W2 chunks
    genvar m_l2_w;
    generate
        for (m_l2_w = 0; m_l2_w < PARALLEL_FACTOR_L2; m_l2_w = m_l2_w + 1) begin : gen_w2_chunk_sel
            assign w2_chunk_to_core[m_l2_w] = (w2_req_input_act_base_idx_from_core + m_l2_w < HIDDEN_SIZE) ?
                                              W2_from_mem[w2_req_input_act_base_idx_from_core + m_l2_w][w2_req_output_node_idx_from_core] :
                                              '0;
        end
    endgenerate

    // Logic to select b2 element
    assign b2_element_to_core = b2_from_mem[w2_req_output_node_idx_from_core];


    // --- Neural Network Core Instantiation ---
    neural_network_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .PARALLEL_FACTOR_L1(PARALLEL_FACTOR_L1),
        .PARALLEL_FACTOR_L2(PARALLEL_FACTOR_L2)
    ) u_neural_network_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),

        // Connect chunks to the core
        .input_data_chunk_in(input_data_chunk_to_core),
        .w1_chunk_in(w1_chunk_to_core),
        .b1_chunk_in(b1_chunk_to_core),
        .w2_chunk_in(w2_chunk_to_core),
        .b2_element_in(b2_element_to_core),

        // Get final results from the core
        .valid_out(valid_out),
        .final_out(final_out),
        .predicted_label(predicted_label),

        // Connect request signals from the core
        .w1_request_base_idx(w1_req_base_idx_from_core),
        .w1_request_hidden_idx(w1_req_hidden_idx_from_core),
        .b1_request_base_idx(b1_req_base_idx_from_core),
        .w2_request_input_act_base_idx(w2_req_input_act_base_idx_from_core),
        .w2_request_output_node_idx(w2_req_output_node_idx_from_core)
    );

endmodule


module neural_network_core #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 128,
    parameter OUTPUT_SIZE = 10,
    parameter PARALLEL_FACTOR_L1 = 4,
    parameter PARALLEL_FACTOR_L2 = 4
)(
    input clk,
    input rst_n,
    input valid_in,

    // Chunked Data Inputs from top_modules
    input signed [DATA_WIDTH-1:0] input_data_chunk_in [0:PARALLEL_FACTOR_L1-1],
    input signed [DATA_WIDTH-1:0] w1_chunk_in [0:PARALLEL_FACTOR_L1-1],
    input signed [DATA_WIDTH-1:0] b1_chunk_in [0:PARALLEL_FACTOR_L1-1],
    input signed [DATA_WIDTH-1:0] w2_chunk_in [0:PARALLEL_FACTOR_L2-1],
    input signed [DATA_WIDTH-1:0] b2_element_in,

    // Final Outputs to top_module
    output valid_out,
    output signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1],
    output [3:0] predicted_label,

    // Request Outputs to top_module/memory
    output [($clog2(INPUT_SIZE))-1:0]  w1_request_base_idx,
    output [($clog2(HIDDEN_SIZE))-1:0] w1_request_hidden_idx,
    output [($clog2(HIDDEN_SIZE))-1:0] b1_request_base_idx,
    output [($clog2(HIDDEN_SIZE))-1:0] w2_request_input_act_base_idx,
    output [($clog2(OUTPUT_SIZE))-1:0] w2_request_output_node_idx
);

    // --- Internal Wires connecting Layer 1 and Layer 2 ---
    wire valid_mid;
    wire signed [DATA_WIDTH-1:0] hidden_out [0:HIDDEN_SIZE-1];

    // --- Input Layer Instantiation ---
    input_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .PARALLEL_FACTOR(PARALLEL_FACTOR_L1)
    ) u_input_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_data_chunk(input_data_chunk_in),
        .w1_chunk_in(w1_chunk_in),
        .b1_chunk_in(b1_chunk_in),
        .valid_out(valid_mid), // Output connects to output_layer's input
        .output_data(hidden_out), // Output connects to output_layer's input
        .w1_request_base_idx(w1_request_base_idx),
        .w1_request_hidden_idx(w1_request_hidden_idx),
        .b1_request_base_idx(b1_request_base_idx)
    );

    // --- Output Layer Instantiation ---
    output_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .PARALLEL_FACTOR_OL(PARALLEL_FACTOR_L2)
    ) u_output_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_mid), // Input from input_layer's output
        .input_data(hidden_out), // Input from input_layer's output
        .w2_chunk_in(w2_chunk_in),
        .b2_element_in(b2_element_in),
        .valid_out(valid_out), // Final valid signal
        .output_data(final_out), // Final raw output
        .predicted_label(predicted_label), // Final prediction
        .w2_request_input_act_base_idx(w2_request_input_act_base_idx),
        .w2_request_output_node_idx(w2_request_output_node_idx)
    );

endmodule

`timescale 1ns/1ps

module softmax_unit #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH_IN = DATA_WIDTH + 8, // Derived from your example code
    parameter SIZE = 10
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [ACC_WIDTH_IN-1:0] acc_data [0:SIZE-1],
    output reg valid_out,
    output reg signed [DATA_WIDTH-1:0] softmax_result [0:SIZE-1],
    output reg [$clog2(SIZE)-1:0] predicted_label // Ensure width matches LABEL_WIDTH
);

    localparam LABEL_WIDTH = $clog2(SIZE);
    localparam SIGNED_MAX_VAL = (1 << (DATA_WIDTH-1)) - 1;

    integer i_comb; // For combinational loops in SM_IDLE (for temp_softmax_result_regs)

    localparam SM_IDLE = 1'b0,
               SM_DONE = 1'b1;
    reg state_sm;

    // Temporary registers for values to be passed to the next stage or output
    // temp_current_max_val_reg was in your code, but not used for output. Can be removed if not needed.
    // reg signed [ACC_WIDTH_IN-1:0] temp_current_max_val_reg; 
    reg [LABEL_WIDTH-1:0]         temp_max_idx_reg; // To hold max_idx if predicted_label needs to align with valid_out
    reg signed [DATA_WIDTH-1:0]   temp_softmax_result_regs [0:SIZE-1];


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out       <= 1'b0;
            predicted_label <= {LABEL_WIDTH{1'b0}}; // Use LABEL_WIDTH for consistency
            state_sm        <= SM_IDLE;
            for (integer k_rst = 0; k_rst < SIZE; k_rst = k_rst + 1) begin
                softmax_result[k_rst] <= 0;
                temp_softmax_result_regs[k_rst] <= 0; 
            end
            // temp_current_max_val_reg <= 0;
            temp_max_idx_reg <= {LABEL_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0; // Default

            case (state_sm)
                SM_IDLE: begin
                    if (valid_in) begin
                        // --- Combinational block to find max and its index ---
                        // These are temporary variables for the combinational logic within this clock cycle.
                        // Declared as 'reg' type because they are assigned in a procedural block.
                        // Synthesis will infer combinational logic.
                        reg signed [ACC_WIDTH_IN-1:0] current_max_val_comb;
                        reg [LABEL_WIDTH-1:0]         max_idx_comb;
                        // The 'softmax_result_comb' from your snippet is not strictly needed if
                        // temp_softmax_result_regs are updated directly based on max_idx_comb.

                        current_max_val_comb = acc_data[0];
                        max_idx_comb         = {LABEL_WIDTH{1'b0}}; // Initialize to 0 with proper width

                        // Loop for finding max. Using a distinct loop variable.
                        for (integer find_max_loop_idx = 1; find_max_loop_idx < SIZE; find_max_loop_idx = find_max_loop_idx + 1) begin
                            if (acc_data[find_max_loop_idx] > current_max_val_comb) begin
                                current_max_val_comb = acc_data[find_max_loop_idx];
                                max_idx_comb         = find_max_loop_idx;
                            end
                        end
                        // --- End of combinational block ---

                        // Register the combinational result if needed for alignment in SM_DONE
                        // temp_current_max_val_reg <= current_max_val_comb; 
                        temp_max_idx_reg         <= max_idx_comb; 

                        // Update the temporary register array for softmax output (these become FFs)
                        // Use non-blocking assignments.
                        for (i_comb = 0; i_comb < SIZE; i_comb = i_comb + 1) begin
                            if (i_comb == max_idx_comb) begin 
                                temp_softmax_result_regs[i_comb] <= SIGNED_MAX_VAL; // Corrected: Non-blocking
                            end else begin
                                temp_softmax_result_regs[i_comb] <= 0;              // Corrected: Non-blocking
                            end
                        end
                        
                        // predicted_label gets the combinational max_idx_comb.
                        // This means predicted_label is available 1 cycle before valid_out and softmax_result.
                        // This is often acceptable or even desired.
                        predicted_label <= max_idx_comb; 
                        
                        state_sm <= SM_DONE;
                    end else begin
                        state_sm <= SM_IDLE;
                    end
                end

                SM_DONE: begin
                    valid_out <= 1'b1;
                    // Assign pipelined results to outputs
                    softmax_result  <= temp_softmax_result_regs; // Corrected: Moved from SM_IDLE

                    // If predicted_label needs to align with valid_out and softmax_result, assign it here:
                    // predicted_label <= temp_max_idx_reg; 
                    // Otherwise, it's already assigned in SM_IDLE and is available earlier.

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

module output_layer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 128, // Hidden layer size
    parameter OUTPUT_SIZE = 10,
    parameter PARALLEL_FACTOR_OL = 4 // Parallel factor for output layer
)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1], // activations from hidden layer

    // Weight and bias inputs (small chunks/element)
    input signed [DATA_WIDTH-1:0] w2_chunk_in [0:PARALLEL_FACTOR_OL-1], // W2[i_cnt + m][j_cnt]
    input signed [DATA_WIDTH-1:0] b2_element_in,                      // b2[j_cnt]

    output valid_out, // To top module (from softmax)
    output signed [DATA_WIDTH-1:0] output_data [0:OUTPUT_SIZE-1], // To top (raw logits for softmax)
    output [3:0] predicted_label, // To top from softmax/argmax

    // Outputs to request specific weight/bias from top_module/memory_module
    output reg [($clog2(INPUT_SIZE))-1:0]  w2_request_input_act_base_idx, // i_cnt for W2 rows
    output reg [($clog2(OUTPUT_SIZE))-1:0] w2_request_output_node_idx    // j_cnt for W2 columns and b2 index
);

    localparam ACC_OL_WIDTH = DATA_WIDTH + 8;

    // Removed W2, b2 internal registers
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
    reg [J_CNT_WIDTH-1:0] j_cnt; // Current output node being processed
    reg [I_CNT_WIDTH-1:0] i_cnt; // Current input activation index for MAC

    reg signed [2*DATA_WIDTH-1:0] ol_product [0:PARALLEL_FACTOR_OL-1];
    reg signed [DATA_WIDTH:0]     ol_scaled_product [0:PARALLEL_FACTOR_OL-1];
    reg signed [ACC_OL_WIDTH-1:0] ol_sum_of_scaled_products;

    // Assign control signals for weight fetching
    always_comb begin
        w2_request_input_act_base_idx = i_cnt;
        w2_request_output_node_idx    = j_cnt;
    end

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
                    // b2_element_in is b2[j_cnt] provided by top_module
                    acc[j_cnt] <= {{ (ACC_OL_WIDTH - DATA_WIDTH){b2_element_in[DATA_WIDTH-1]} }, b2_element_in};
                    i_cnt <= 0;
                    ol_next_state <= OL_S_MAC_LOOP;
                end

                OL_S_MAC_LOOP: begin
                    ol_sum_of_scaled_products = '0;
                    // Top module ensures w2_chunk_in corresponds to W2[i_cnt+m][j_cnt]
                    for (integer signed m = 0; m < PARALLEL_FACTOR_OL; m = m + 1) begin
                        if (i_cnt + m < INPUT_SIZE) begin
                            // W2[i_cnt + m][j_cnt] is now w2_chunk_in[m]
                            ol_product[m] = input_data[i_cnt + m] * w2_chunk_in[m];
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
        .DATA_WIDTH(DATA_WIDTH), // Output data width
        .ACC_WIDTH_IN(ACC_OL_WIDTH), // Specify input accumulator width
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

`timescale 1ns/1ps

module input_layer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 784,
    parameter HIDDEN_SIZE = 128,
    parameter PARALLEL_FACTOR = 4 // M?c ?? song song hóa
)(
    input  clk,
    input  rst_n,
    input  valid_in,
    input  signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],

    // Weight and bias inputs (small chunks)
    input  signed [DATA_WIDTH-1:0] w1_chunk_in [0:PARALLEL_FACTOR-1], // W1[current_i_base + m][hidden_node_cnt]
    input  signed [DATA_WIDTH-1:0] b1_chunk_in [0:PARALLEL_FACTOR-1], // b1[j_proc_cnt + m]

    output reg valid_out,
    output reg signed [DATA_WIDTH-1:0] output_data [0:HIDDEN_SIZE-1],

    // Outputs to request specific weight/bias chunks from top_module/memory_module
    output reg [($clog2(INPUT_SIZE))-1:0] w1_request_base_idx, // current_i_base for W1
    output reg [($clog2(HIDDEN_SIZE))-1:0] w1_request_hidden_idx, // hidden_node_cnt for W1
    output reg [($clog2(HIDDEN_SIZE))-1:0] b1_request_base_idx   // j_proc_cnt for b1
);

    localparam FIXED_NUM_CHUNKS = 16;
    localparam FIXED_CHUNK_SIZE = INPUT_SIZE / FIXED_NUM_CHUNKS; // Should be 49 if INPUT_SIZE=784
    localparam ACC_WIDTH = 24;
    localparam CHUNK_CNT_WIDTH = $clog2(FIXED_NUM_CHUNKS);
    localparam HIDDEN_CNT_WIDTH = $clog2(HIDDEN_SIZE);
    localparam DELTA_VAL_WIDTH = 23;

    localparam K_LOOP_CNT_WIDTH = $clog2(FIXED_CHUNK_SIZE);
    reg [K_LOOP_CNT_WIDTH-1:0] k_loop_cnt;

    reg [HIDDEN_CNT_WIDTH-1:0] j_proc_cnt; // For bias and ReLU processing

    // Removed W1, b1 internal registers
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
    reg [HIDDEN_CNT_WIDTH-1:0] hidden_node_cnt; // Current hidden node being processed for MAC
    reg signed [DELTA_VAL_WIDTH-1:0] current_delta_sum_for_node;

    reg signed [2*DATA_WIDTH-1:0]     product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DATA_WIDTH:0]         scaled_product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DELTA_VAL_WIDTH-1:0]  sum_of_scaled_products;

    integer current_i_base; // Base index for parallel access for input_data and W1 rows

    // Assign control signals for weight fetching
    // These are combinational based on current state counters
    // For W1: We need W1[ (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt + m ][ hidden_node_cnt ]
    // The base row index is (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt
    // The column index is hidden_node_cnt
    always_comb begin
        w1_request_base_idx   = (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt;
        w1_request_hidden_idx = hidden_node_cnt;
        b1_request_base_idx   = j_proc_cnt; // For b1[j_proc_cnt + m]
    end

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
            valid_out <= 1'b0; // Default

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
                    // current_i_base is implicitly w1_request_base_idx
                    // hidden_node_cnt is w1_request_hidden_idx
                    // The top module ensures w1_chunk_in corresponds to these indices
                    current_i_base = (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt; // Recalculate for input_data access
                    sum_of_scaled_products = '0;

                    for (integer signed m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (k_loop_cnt + m < FIXED_CHUNK_SIZE) begin
                            // W1[current_i_base + m][hidden_node_cnt] is now w1_chunk_in[m]
                            product_seq[m] = input_data[current_i_base + m] * w1_chunk_in[m];
                            scaled_product_seq[m] = product_seq[m] >>> 12;
                            sum_of_scaled_products = sum_of_scaled_products +
                                                     {{(DELTA_VAL_WIDTH - (DATA_WIDTH+1)){scaled_product_seq[m][DATA_WIDTH]}}, scaled_product_seq[m]};
                        end
                    end
                    current_delta_sum_for_node <= current_delta_sum_for_node + sum_of_scaled_products;

                    if (k_loop_cnt + PARALLEL_FACTOR < FIXED_CHUNK_SIZE) begin
                        k_loop_cnt <= k_loop_cnt + PARALLEL_FACTOR;
                        next_state <= S_CALC_DELTA_LOOP; // Stay to process next part of chunk with new k_loop_cnt
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
                        next_state <= S_CALC_DELTA_SETUP; // New hidden node, restart k_loop, current_delta_sum
                    end else begin // All hidden nodes processed for current chunk
                        hidden_node_cnt <= 0; // Reset for next chunk
                        if (chunk_cnt < FIXED_NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1;
                            next_state <= S_CALC_DELTA_SETUP; // Process next chunk for all hidden nodes
                        end else begin // All chunks and all hidden nodes' MACs are done
                            next_state <= S_ADD_BIAS_SETUP;
                        end
                    end
                end

                S_ADD_BIAS_SETUP: begin
                    j_proc_cnt <= 0; // This j_proc_cnt is for iterating through HIDDEN_SIZE for bias/ReLU
                    next_state <= S_ADD_BIAS_LOOP;
                end

                S_ADD_BIAS_LOOP: begin
                    // top_module ensures b1_chunk_in corresponds to b1_request_base_idx (j_proc_cnt)
                    for (integer signed m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (j_proc_cnt + m < HIDDEN_SIZE) begin
                            // b1[j_proc_cnt + m] is now b1_chunk_in[m]
                            acc[j_proc_cnt + m] <= acc[j_proc_cnt + m] + {{ (ACC_WIDTH - DATA_WIDTH){b1_chunk_in[m][DATA_WIDTH-1]} }, b1_chunk_in[m]};
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
                            if (acc[j_proc_cnt + m][ACC_WIDTH-1]) begin // Negative
                                output_data[j_proc_cnt + m] <= '0;
                            end else begin // Positive
                                if (acc[j_proc_cnt + m] > ((1 << (DATA_WIDTH-1)) - 1) ) begin // Saturate
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

`timescale 1ns/1ps

module memory_module #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 128,
    parameter OUTPUT_SIZE = 10
)(
    // No clock or reset needed for this simple ROM-like behavior
    // Outputs are combinational reads from the initialized memories.

    // Weights and biases for input_layer
    output reg signed [DATA_WIDTH-1:0] W1_out [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1],
    output reg signed [DATA_WIDTH-1:0] b1_out [0:HIDDEN_SIZE-1],

    // Weights and biases for output_layer
    // Note: output_layer's INPUT_SIZE is HIDDEN_SIZE of the network
    output reg signed [DATA_WIDTH-1:0] W2_out [0:HIDDEN_SIZE-1][0:OUTPUT_SIZE-1],
    output reg signed [DATA_WIDTH-1:0] b2_out [0:OUTPUT_SIZE-1]
);

    initial begin
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W1.mem", W1_out);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b1.mem", b1_out);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W2.mem", W2_out);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b2.mem", b2_out);
        $display("Memory Module: Weights and biases loaded at time %0t.", $time);
    end

endmodule