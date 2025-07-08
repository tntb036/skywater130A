`timescale 1ns/1ps

//********************************************************************************//
// Module: top_module (FINAL VERILOG-2001 VERSION)                                //
// Description: Giao tiếp với memory_module đã được thiết kế lại để đọc từng      //
//              phần tử và tạo các chunk dữ liệu.                                 //
//********************************************************************************//
module top_module (
    input clk,
    input rst_n,
    input valid_in,
    // input_data_flat: 784 * 16 = 12544 bits
    input signed [12543:0] input_data_flat,
    output valid_out,
    // final_out_flat: 10 * 16 = 160 bits
    output signed [159:0] final_out_flat,
    output [3:0] predicted_label
);
    // --- Unpack flattened input into an internal wire array ---
    wire signed [15:0] input_data [0:783];
    genvar i_unpack_input;
    generate
        for (i_unpack_input = 0; i_unpack_input < 784; i_unpack_input = i_unpack_input + 1) begin : gen_unpack_input
            assign input_data[i_unpack_input] = input_data_flat[(i_unpack_input+1)*16-1 -: 16];
        end
    endgenerate

    // --- Wires for Core Communication (requests from core) ---
    wire [9:0]  w1_req_base_idx_from_core;       // clog2(784) = 10
    wire [6:0]  w1_req_hidden_idx_from_core;     // clog2(128) = 7
    wire [6:0]  b1_req_base_idx_from_core;       // clog2(128) = 7
    wire [6:0]  w2_req_input_act_base_idx_from_core; // clog2(128) = 7
    wire [3:0]  w2_req_output_node_idx_from_core;    // clog2(10)  = 4

    // --- Wires to carry selected chunks TO the core ---
    wire signed [15:0] input_data_chunk_to_core [0:3];
    wire signed [15:0] w1_chunk_to_core [0:3];
    wire signed [15:0] b1_chunk_to_core [0:3];
    wire signed [15:0] w2_chunk_to_core [0:3];
    wire signed [15:0] b2_element_to_core;
    
    // --- Data from Memory Module ---
    wire signed [15:0] w1_data_from_mem [0:3];
    wire signed [15:0] b1_data_from_mem [0:3];
    wire signed [15:0] w2_data_from_mem [0:3];

    // --- Memory Module Instantiation ---
    memory_module u_memory_module (
        // W1 read ports (4 parallel reads for one column)
        .w1_addr_row_0(w1_req_base_idx_from_core + 0),
        .w1_addr_row_1(w1_req_base_idx_from_core + 1),
        .w1_addr_row_2(w1_req_base_idx_from_core + 2),
        .w1_addr_row_3(w1_req_base_idx_from_core + 3),
        .w1_addr_col(w1_req_hidden_idx_from_core),
        .w1_data_out_0(w1_data_from_mem[0]),
        .w1_data_out_1(w1_data_from_mem[1]),
        .w1_data_out_2(w1_data_from_mem[2]),
        .w1_data_out_3(w1_data_from_mem[3]),
        
        // b1 read ports (4 parallel reads)
        .b1_addr_0(b1_req_base_idx_from_core + 0),
        .b1_addr_1(b1_req_base_idx_from_core + 1),
        .b1_addr_2(b1_req_base_idx_from_core + 2),
        .b1_addr_3(b1_req_base_idx_from_core + 3),
        .b1_data_out_0(b1_data_from_mem[0]),
        .b1_data_out_1(b1_data_from_mem[1]),
        .b1_data_out_2(b1_data_from_mem[2]),
        .b1_data_out_3(b1_data_from_mem[3]),

        // W2 read ports (4 parallel reads for one column)
        .w2_addr_row_0(w2_req_input_act_base_idx_from_core + 0),
        .w2_addr_row_1(w2_req_input_act_base_idx_from_core + 1),
        .w2_addr_row_2(w2_req_input_act_base_idx_from_core + 2),
        .w2_addr_row_3(w2_req_input_act_base_idx_from_core + 3),
        .w2_addr_col(w2_req_output_node_idx_from_core),
        .w2_data_out_0(w2_data_from_mem[0]),
        .w2_data_out_1(w2_data_from_mem[1]),
        .w2_data_out_2(w2_data_from_mem[2]),
        .w2_data_out_3(w2_data_from_mem[3]),
        
        // b2 read port (single read)
        .b2_addr(w2_req_output_node_idx_from_core),
        .b2_data_out(b2_element_to_core)
    );

    // --- Data/Weight/Bias Slicing Logic ---
    genvar m_chunk;
    generate
        for (m_chunk = 0; m_chunk < 4; m_chunk = m_chunk + 1) begin : gen_chunk_sel
            // Input data chunk
            assign input_data_chunk_to_core[m_chunk] = (w1_req_base_idx_from_core + m_chunk < 784) ? 
                                                       input_data[w1_req_base_idx_from_core + m_chunk] : 16'h0;
            // W1 chunk
            assign w1_chunk_to_core[m_chunk] = (w1_req_base_idx_from_core + m_chunk < 784) ? 
                                               w1_data_from_mem[m_chunk] : 16'h0;
            // b1 chunk
            assign b1_chunk_to_core[m_chunk] = (b1_req_base_idx_from_core + m_chunk < 128) ? 
                                               b1_data_from_mem[m_chunk] : 16'h0;
            // W2 chunk
            assign w2_chunk_to_core[m_chunk] = (w2_req_input_act_base_idx_from_core + m_chunk < 128) ? 
                                               w2_data_from_mem[m_chunk] : 16'h0;
        end
    endgenerate

    // --- Flatten chunks before passing to core ---
    wire signed [63:0] input_data_chunk_flat; // 4 * 16 = 64
    wire signed [63:0] w1_chunk_flat;
    wire signed [63:0] b1_chunk_flat;
    wire signed [63:0] w2_chunk_flat;

    genvar p_l;
    generate
        for (p_l = 0; p_l < 4; p_l = p_l + 1) begin : gen_pack_chunks
            assign input_data_chunk_flat[(p_l+1)*16-1 -: 16] = input_data_chunk_to_core[p_l];
            assign w1_chunk_flat[(p_l+1)*16-1 -: 16]         = w1_chunk_to_core[p_l];
            assign b1_chunk_flat[(p_l+1)*16-1 -: 16]         = b1_chunk_to_core[p_l];
            assign w2_chunk_flat[(p_l+1)*16-1 -: 16]         = w2_chunk_to_core[p_l];
        end
    endgenerate

    // --- Neural Network Core Instantiation ---
    neural_network_core u_neural_network_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_data_chunk_in_flat(input_data_chunk_flat),
        .w1_chunk_in_flat(w1_chunk_flat),
        .b1_chunk_in_flat(b1_chunk_flat),
        .w2_chunk_in_flat(w2_chunk_flat),
        .b2_element_in(b2_element_to_core),
        .valid_out(valid_out),
        .final_out_flat(final_out_flat),
        .predicted_label(predicted_label),
        .w1_request_base_idx(w1_req_base_idx_from_core),
        .w1_request_hidden_idx(w1_req_hidden_idx_from_core),
        .b1_request_base_idx(b1_req_base_idx_from_core),
        .w2_request_input_act_base_idx(w2_req_input_act_base_idx_from_core),
        .w2_request_output_node_idx(w2_req_output_node_idx_from_core)
    );

endmodule


//****************************************************************************//
// Module: neural_network_core                                                //
//****************************************************************************//
module neural_network_core (
    input clk,
    input rst_n,
    input valid_in,
    input signed [63:0] input_data_chunk_in_flat,
    input signed [63:0] w1_chunk_in_flat,
    input signed [63:0] b1_chunk_in_flat,
    input signed [63:0] w2_chunk_in_flat,
    input signed [15:0] b2_element_in,
    output valid_out,
    output signed [159:0] final_out_flat,
    output [3:0] predicted_label,
    output [9:0]  w1_request_base_idx,
    output [6:0]  w1_request_hidden_idx,
    output [6:0]  b1_request_base_idx,
    output [6:0]  w2_request_input_act_base_idx,
    output [3:0]  w2_request_output_node_idx
);
    reg signed [15:0] hidden_activations_ram [0:127];
    wire valid_mid;
    wire hidden_ram_we_from_l1;
    wire [6:0] hidden_ram_addr_from_l1;
    wire signed [15:0] hidden_ram_wdata_from_l1;
    wire signed [15:0] hidden_data_chunk_to_l2 [0:3];
    wire signed [63:0] hidden_data_chunk_to_l2_flat;

    always @(posedge clk) begin
        if (hidden_ram_we_from_l1) begin
            hidden_activations_ram[hidden_ram_addr_from_l1] <= hidden_ram_wdata_from_l1;
        end
    end

    genvar m_ram_read;
    generate
        for (m_ram_read = 0; m_ram_read < 4; m_ram_read = m_ram_read + 1) begin : gen_ram_read_chunk
            assign hidden_data_chunk_to_l2[m_ram_read] = hidden_activations_ram[w2_request_input_act_base_idx + m_ram_read];
            assign hidden_data_chunk_to_l2_flat[(m_ram_read+1)*16-1 -: 16] = hidden_data_chunk_to_l2[m_ram_read];
        end
    endgenerate

    input_layer u_input_layer (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .input_data_chunk_flat(input_data_chunk_in_flat),
        .w1_chunk_in_flat(w1_chunk_in_flat),
        .b1_chunk_in_flat(b1_chunk_in_flat),
        .valid_out(valid_mid),
        .hidden_ram_we(hidden_ram_we_from_l1),
        .hidden_ram_addr(hidden_ram_addr_from_l1),
        .hidden_ram_wdata(hidden_ram_wdata_from_l1),
        .w1_request_base_idx(w1_request_base_idx),
        .w1_request_hidden_idx(w1_request_hidden_idx),
        .b1_request_base_idx(b1_request_base_idx)
    );

    output_layer u_output_layer (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_mid),
        .hidden_data_chunk_in_flat(hidden_data_chunk_to_l2_flat),
        .w2_chunk_in_flat(w2_chunk_in_flat),
        .b2_element_in(b2_element_in),
        .valid_out(valid_out),
        .output_data_flat(final_out_flat),
        .predicted_label(predicted_label),
        .hidden_ram_read_base_addr(w2_request_input_act_base_idx),
        .w2_request_output_node_idx(w2_request_output_node_idx)
    );
endmodule


//****************************************************************************//
// Module: input_layer                                                        //
//****************************************************************************//
module input_layer (
    input clk,
    input rst_n,
    input valid_in,
    input signed [63:0] input_data_chunk_flat,
    input signed [63:0] w1_chunk_in_flat,
    input signed [63:0] b1_chunk_in_flat,
    output valid_out,
    output hidden_ram_we,
    output [6:0] hidden_ram_addr,
    output signed [15:0] hidden_ram_wdata,
    output [9:0] w1_request_base_idx,
    output [6:0] w1_request_hidden_idx,
    output [6:0] b1_request_base_idx
);
    reg valid_out_reg;
    reg hidden_ram_we_reg;
    reg [6:0] hidden_ram_addr_reg;
    reg signed [15:0] hidden_ram_wdata_reg;
    reg [9:0] w1_request_base_idx_reg;
    reg [6:0] w1_request_hidden_idx_reg;
    reg [6:0] b1_request_base_idx_reg;
    
    assign valid_out = valid_out_reg;
    assign hidden_ram_we = hidden_ram_we_reg;
    assign hidden_ram_addr = hidden_ram_addr_reg;
    assign hidden_ram_wdata = hidden_ram_wdata_reg;
    assign w1_request_base_idx = w1_request_base_idx_reg;
    assign w1_request_hidden_idx = w1_request_hidden_idx_reg;
    assign b1_request_base_idx = b1_request_base_idx_reg;

    wire signed [15:0] input_data_chunk [0:3];
    wire signed [15:0] w1_chunk_in [0:3];
    wire signed [15:0] b1_chunk_in [0:3];

    genvar u_in;
    generate
        for(u_in = 0; u_in < 4; u_in = u_in + 1) begin: gen_unpack_in_chunks
            assign input_data_chunk[u_in] = input_data_chunk_flat[(u_in+1)*16-1 -: 16];
            assign w1_chunk_in[u_in]      = w1_chunk_in_flat[(u_in+1)*16-1 -: 16];
            assign b1_chunk_in[u_in]      = b1_chunk_in_flat[(u_in+1)*16-1 -: 16];
        end
    endgenerate

    localparam FIXED_NUM_CHUNKS = 16;
    localparam FIXED_CHUNK_SIZE = 49; // 784 / 16 = 49
    localparam ACC_WIDTH = 24;
    localparam DELTA_VAL_WIDTH = 23;
    localparam CHUNK_CNT_WIDTH = 4;      // clog2(16)
    localparam HIDDEN_CNT_WIDTH = 7;     // clog2(128)
    localparam K_LOOP_CNT_WIDTH = 6;     // clog2(49)

    reg [K_LOOP_CNT_WIDTH-1:0] k_loop_cnt;
    reg [HIDDEN_CNT_WIDTH-1:0] j_proc_cnt;
    reg signed [ACC_WIDTH-1:0] acc [0:127];

    localparam [3:0] S_IDLE=4'h0, S_INIT_ALL=4'h1, S_CALC_DELTA_SETUP=4'h2, S_CALC_DELTA_LOOP=4'h3,
                     S_ACCUMULATE_DELTA=4'h4, S_NEXT_HIDDEN_OR_CHUNK=4'h5, S_ADD_BIAS_SETUP=4'h6,
                     S_ADD_BIAS_LOOP=4'h7, S_RELU_SETUP=4'h8, S_RELU_LOOP=4'h9, S_DONE=4'hA;
    reg [3:0] state, next_state;

    reg [CHUNK_CNT_WIDTH-1:0] chunk_cnt;
    reg [HIDDEN_CNT_WIDTH-1:0] hidden_node_cnt;
    reg signed [DELTA_VAL_WIDTH-1:0] current_delta_sum_for_node;

    reg signed [31:0] product_seq [0:3];
    reg signed [16:0] scaled_product_seq [0:3];
    reg signed [DELTA_VAL_WIDTH-1:0] sum_of_scaled_products;
    reg signed [15:0] relu_result;
    
    integer m, j_rst, j_init;

    always @(*) begin
        w1_request_base_idx_reg   = (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt;
        w1_request_hidden_idx_reg = hidden_node_cnt;
        b1_request_base_idx_reg   = j_proc_cnt; 
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; valid_out_reg <= 1'b0; hidden_ram_we_reg <= 1'b0;
            chunk_cnt <= 0; hidden_node_cnt <= 0; k_loop_cnt <= 0; j_proc_cnt <= 0;
            current_delta_sum_for_node <= 0;
            for (j_rst=0; j_rst<128; j_rst=j_rst+1) acc[j_rst] <= 0;
            hidden_ram_addr_reg <= 0; hidden_ram_wdata_reg <= 0;
        end else begin
            state <= next_state;
            valid_out_reg <= 1'b0;
            hidden_ram_we_reg <= 1'b0;

            case (state)
                S_IDLE: if(valid_in) next_state <= S_INIT_ALL; else next_state <= S_IDLE;
                S_INIT_ALL: begin
                    for(j_init=0; j_init<128; j_init=j_init+1) acc[j_init] <= 0;
                    chunk_cnt <= 0; hidden_node_cnt <= 0; next_state <= S_CALC_DELTA_SETUP;
                end
                S_CALC_DELTA_SETUP: begin k_loop_cnt<=0; current_delta_sum_for_node<=0; next_state<=S_CALC_DELTA_LOOP; end
                S_CALC_DELTA_LOOP: begin
                    sum_of_scaled_products = 0;
                    for (m=0; m<4; m=m+1) begin
                        if(k_loop_cnt+m < FIXED_CHUNK_SIZE) begin
                            product_seq[m] = input_data_chunk[m] * w1_chunk_in[m];
                            scaled_product_seq[m] = product_seq[m] >> 12; 
                            sum_of_scaled_products = sum_of_scaled_products + {{(DELTA_VAL_WIDTH-17){scaled_product_seq[m][16]}}, scaled_product_seq[m]};
                        end
                    end
                    current_delta_sum_for_node <= current_delta_sum_for_node + sum_of_scaled_products;
                    if(k_loop_cnt+4 < FIXED_CHUNK_SIZE) begin k_loop_cnt<=k_loop_cnt+4; next_state<=S_CALC_DELTA_LOOP; end
                    else begin next_state <= S_ACCUMULATE_DELTA; end
                end
                S_ACCUMULATE_DELTA: begin
                    acc[hidden_node_cnt] <= acc[hidden_node_cnt] + {{(ACC_WIDTH-DELTA_VAL_WIDTH){current_delta_sum_for_node[DELTA_VAL_WIDTH-1]}}, current_delta_sum_for_node};
                    next_state <= S_NEXT_HIDDEN_OR_CHUNK;
                end
                S_NEXT_HIDDEN_OR_CHUNK: begin
                    if(hidden_node_cnt < 127) begin hidden_node_cnt<=hidden_node_cnt+1; next_state<=S_CALC_DELTA_SETUP; end
                    else begin 
                        hidden_node_cnt <= 0;
                        if(chunk_cnt < 15) begin chunk_cnt<=chunk_cnt+1; next_state<=S_CALC_DELTA_SETUP; end
                        else begin next_state <= S_ADD_BIAS_SETUP; end
                    end
                end
                S_ADD_BIAS_SETUP: begin j_proc_cnt<=0; next_state<=S_ADD_BIAS_LOOP; end
                S_ADD_BIAS_LOOP: begin
                    acc[j_proc_cnt] <= acc[j_proc_cnt] + {{8{b1_chunk_in[0][15]}}, b1_chunk_in[0]};
                    if(j_proc_cnt < 127) begin j_proc_cnt<=j_proc_cnt+1; next_state<=S_ADD_BIAS_LOOP; end
                    else begin next_state <= S_RELU_SETUP; end
                end
                S_RELU_SETUP: begin j_proc_cnt<=0; next_state<=S_RELU_LOOP; end
                S_RELU_LOOP: begin
                    if(acc[j_proc_cnt][ACC_WIDTH-1]) relu_result = 0;
                    else begin
                        if(acc[j_proc_cnt] > 32767) relu_result = 32767; // 2^(15)-1
                        else relu_result = acc[j_proc_cnt][15:0];
                    end
                    hidden_ram_we_reg<=1'b1; hidden_ram_addr_reg<=j_proc_cnt; hidden_ram_wdata_reg<=relu_result;
                    if(j_proc_cnt < 127) begin j_proc_cnt<=j_proc_cnt+1; next_state<=S_RELU_LOOP; end
                    else begin next_state <= S_DONE; end
                end
                S_DONE: begin valid_out_reg<=1'b1; next_state<=S_IDLE; end
                default: next_state <= S_IDLE;
            endcase
        end
    end
endmodule


//****************************************************************************//
// Module: output_layer                                                       //
//****************************************************************************//
module output_layer (
    input clk,
    input rst_n,
    input valid_in,
    input signed [63:0] hidden_data_chunk_in_flat,
    input signed [63:0] w2_chunk_in_flat,
    input signed [15:0] b2_element_in,
    output valid_out,
    output signed [159:0] output_data_flat,
    output [3:0] predicted_label,
    output [6:0] hidden_ram_read_base_addr,
    output [3:0] w2_request_output_node_idx
);
    reg [6:0] hidden_ram_read_base_addr_reg;
    reg [3:0] w2_request_output_node_idx_reg;
    
    assign hidden_ram_read_base_addr = hidden_ram_read_base_addr_reg;
    assign w2_request_output_node_idx = w2_request_output_node_idx_reg;

    wire signed [15:0] hidden_data_chunk_in [0:3];
    wire signed [15:0] w2_chunk_in [0:3];
    
    genvar u_out;
    generate
        for(u_out=0; u_out<4; u_out=u_out+1) begin: gen_unpack_out_chunks
            assign hidden_data_chunk_in[u_out] = hidden_data_chunk_in_flat[(u_out+1)*16-1 -: 16];
            assign w2_chunk_in[u_out] = w2_chunk_in_flat[(u_out+1)*16-1 -: 16];
        end
    endgenerate

    localparam ACC_OL_WIDTH = 24;
    reg signed [ACC_OL_WIDTH-1:0] acc [0:9];
    reg valid_softmax_in_internal;

    localparam [2:0] OL_S_IDLE=3'b000, OL_S_LOAD_BIAS=3'b001, OL_S_MAC_LOOP=3'b010,
                     OL_S_NEXT_NODE=3'b011, OL_S_SOFTMAX=3'b100;
    reg [2:0] ol_state, ol_next_state;

    reg [3:0] j_cnt; // for 10
    reg [6:0] i_cnt; // for 128

    reg signed [31:0] ol_product [0:3];
    reg signed [16:0] ol_scaled_product [0:3];
    reg signed [ACC_OL_WIDTH-1:0] ol_sum_of_scaled_products;
    
    integer m, k_rst;

    always @(*) begin
        hidden_ram_read_base_addr_reg = i_cnt;
        w2_request_output_node_idx_reg = j_cnt;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ol_state <= OL_S_IDLE; valid_softmax_in_internal <= 1'b0;
            j_cnt <= 0; i_cnt <= 0;
            for (k_rst=0; k_rst<10; k_rst=k_rst+1) acc[k_rst] <= 0;
        end else begin
            ol_state <= ol_next_state;
            valid_softmax_in_internal <= 1'b0;

            case (ol_state)
                OL_S_IDLE: if(valid_in) begin j_cnt<=0; ol_next_state<=OL_S_LOAD_BIAS; end else ol_next_state<=OL_S_IDLE;
                OL_S_LOAD_BIAS: begin
                    acc[j_cnt] <= {{8{b2_element_in[15]}}, b2_element_in};
                    i_cnt <= 0; ol_next_state <= OL_S_MAC_LOOP;
                end
                OL_S_MAC_LOOP: begin
                    ol_sum_of_scaled_products = 0;
                    for (m=0; m<4; m=m+1) begin
                        if(i_cnt+m < 128) begin
                            ol_product[m] = hidden_data_chunk_in[m] * w2_chunk_in[m];
                            ol_scaled_product[m] = ol_product[m] >> 12;
                            ol_sum_of_scaled_products = ol_sum_of_scaled_products + {{(ACC_OL_WIDTH-17){ol_scaled_product[m][16]}}, ol_scaled_product[m]};
                        end
                    end
                    acc[j_cnt] <= acc[j_cnt] + ol_sum_of_scaled_products;
                    if(i_cnt+4 < 128) begin i_cnt<=i_cnt+4; ol_next_state<=OL_S_MAC_LOOP; end
                    else begin ol_next_state <= OL_S_NEXT_NODE; end
                end
                OL_S_NEXT_NODE: begin
                    if(j_cnt < 9) begin j_cnt<=j_cnt+1; ol_next_state<=OL_S_LOAD_BIAS; end
                    else begin ol_next_state <= OL_S_SOFTMAX; end
                end
                OL_S_SOFTMAX: begin valid_softmax_in_internal<=1'b1; ol_next_state<=OL_S_IDLE; end
                default: ol_next_state <= OL_S_IDLE;
            endcase
        end
    end

    wire signed [239:0] acc_flat; // 10 * 24
    genvar p_acc;
    generate 
        for(p_acc=0; p_acc<10; p_acc=p_acc+1) begin: gen_pack_acc
            assign acc_flat[(p_acc+1)*24-1 -: 24] = acc[p_acc];
        end
    endgenerate

    softmax_unit u_softmax_unit (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_softmax_in_internal),
        .acc_data_flat(acc_flat),
        .valid_out(valid_out),
        .softmax_result_flat(output_data_flat),
        .predicted_label(predicted_label)
    );
endmodule


//****************************************************************************//
// Module: softmax_unit                                                       //
//****************************************************************************//
module softmax_unit (
    input clk,
    input rst_n,
    input valid_in,
    input signed [239:0] acc_data_flat, // 10 * 24
    output valid_out,
    output signed [159:0] softmax_result_flat, // 10 * 16
    output [3:0] predicted_label
);
    reg valid_out_reg;
    reg [3:0] predicted_label_reg;
    reg signed [15:0] softmax_result_reg [0:9];
    
    assign valid_out = valid_out_reg;
    assign predicted_label = predicted_label_reg;
    
    genvar p_smax;
    generate
        for(p_smax=0; p_smax<10; p_smax=p_smax+1) begin: gen_pack_softmax
            assign softmax_result_flat[(p_smax+1)*16-1 -: 16] = softmax_result_reg[p_smax];
        end
    endgenerate

    wire signed [23:0] acc_data [0:9];
    genvar u_smax;
    generate
        for(u_smax=0; u_smax<10; u_smax=u_smax+1) begin: gen_unpack_smax
            assign acc_data[u_smax] = acc_data_flat[(u_smax+1)*24-1 -: 24];
        end
    endgenerate

    localparam SIGNED_MAX_VAL = 16'h7FFF; // 32767

    integer i_comb, k_rst, find_max_loop_idx;
    reg signed [23:0] current_max_val_comb;
    reg [3:0] max_idx_comb;

    localparam SM_IDLE=1'b0, SM_DONE=1'b1;
    reg state_sm;

    reg [3:0] temp_max_idx_reg;
    reg signed [15:0] temp_softmax_result_regs [0:9];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out_reg <= 1'b0; predicted_label_reg <= 4'h0;
            state_sm <= SM_IDLE;
            for(k_rst=0; k_rst<10; k_rst=k_rst+1) begin
                softmax_result_reg[k_rst] <= 0;
                temp_softmax_result_regs[k_rst] <= 0;
            end
            temp_max_idx_reg <= 4'h0;
        end else begin
            valid_out_reg <= 1'b0;

            case (state_sm)
                SM_IDLE: begin
                    if(valid_in) begin
                        current_max_val_comb = acc_data[0]; max_idx_comb = 4'h0;
                        for(find_max_loop_idx=1; find_max_loop_idx<10; find_max_loop_idx=find_max_loop_idx+1) begin
                            if(acc_data[find_max_loop_idx] > current_max_val_comb) begin
                                current_max_val_comb = acc_data[find_max_loop_idx];
                                max_idx_comb = find_max_loop_idx;
                            end
                        end
                        temp_max_idx_reg <= max_idx_comb;
                        for(i_comb=0; i_comb<10; i_comb=i_comb+1) begin
                            if(i_comb == max_idx_comb) temp_softmax_result_regs[i_comb] <= SIGNED_MAX_VAL;
                            else temp_softmax_result_regs[i_comb] <= 0;
                        end
                        predicted_label_reg <= max_idx_comb;
                        state_sm <= SM_DONE;
                    end else state_sm <= SM_IDLE;
                end
                SM_DONE: begin
                    valid_out_reg <= 1'b1;
                    for(i_comb=0; i_comb<10; i_comb=i_comb+1) begin
                         softmax_result_reg[i_comb] <= temp_softmax_result_regs[i_comb];
                    end
                    state_sm <= SM_IDLE;
                end
                default: state_sm <= SM_IDLE;
            endcase
        end
    end
endmodule


//****************************************************************************//
// Module: memory_module (FINAL VERILOG-2001 VERSION)                         //
// Description: Hoạt động như một ROM đa cổng (multi-ported ROM). Nhận địa chỉ //
//              và trả về dữ liệu. Không còn cổng mảng.                       //
//****************************************************************************//
module memory_module (
    // W1 Read Ports (4 parallel ports)
    input [9:0] w1_addr_row_0,
    input [9:0] w1_addr_row_1,
    input [9:0] w1_addr_row_2,
    input [9:0] w1_addr_row_3,
    input [6:0] w1_addr_col,
    output signed [15:0] w1_data_out_0,
    output signed [15:0] w1_data_out_1,
    output signed [15:0] w1_data_out_2,
    output signed [15:0] w1_data_out_3,
    
    // b1 Read Ports (4 parallel ports)
    input [6:0] b1_addr_0,
    input [6:0] b1_addr_1,
    input [6:0] b1_addr_2,
    input [6:0] b1_addr_3,
    output signed [15:0] b1_data_out_0,
    output signed [15:0] b1_data_out_1,
    output signed [15:0] b1_data_out_2,
    output signed [15:0] b1_data_out_3,

    // W2 Read Ports (4 parallel ports)
    input [6:0] w2_addr_row_0,
    input [6:0] w2_addr_row_1,
    input [6:0] w2_addr_row_2,
    input [6:0] w2_addr_row_3,
    input [3:0] w2_addr_col,
    output signed [15:0] w2_data_out_0,
    output signed [15:0] w2_data_out_1,
    output signed [15:0] w2_data_out_2,
    output signed [15:0] w2_data_out_3,
    
    // b2 Read Port (single port)
    input [3:0] b2_addr,
    output signed [15:0] b2_data_out
);
    // Khai báo bộ nhớ ROM
    reg signed [15:0] W1_mem [0:783][0:127];
    reg signed [15:0] b1_mem [0:127];
    reg signed [15:0] W2_mem [0:127][0:9];
    reg signed [15:0] b2_mem [0:9];

    // Khởi tạo bộ nhớ từ file
    initial begin
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W1.mem", W1_mem);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b1.mem", b1_mem);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/W2.mem", W2_mem);
        $readmemh("/VNCHIP/WORK/thienbao/Neural_network/data/mem/b2.mem", b2_mem);
    end

    // Logic đọc tổ hợp (combinational read)
    assign w1_data_out_0 = W1_mem[w1_addr_row_0][w1_addr_col];
    assign w1_data_out_1 = W1_mem[w1_addr_row_1][w1_addr_col];
    assign w1_data_out_2 = W1_mem[w1_addr_row_2][w1_addr_col];
    assign w1_data_out_3 = W1_mem[w1_addr_row_3][w1_addr_col];

    assign b1_data_out_0 = b1_mem[b1_addr_0];
    assign b1_data_out_1 = b1_mem[b1_addr_1];
    assign b1_data_out_2 = b1_mem[b1_addr_2];
    assign b1_data_out_3 = b1_mem[b1_addr_3];

    assign w2_data_out_0 = W2_mem[w2_addr_row_0][w2_addr_col];
    assign w2_data_out_1 = W2_mem[w2_addr_row_1][w2_addr_col];
    assign w2_data_out_2 = W2_mem[w2_addr_row_2][w2_addr_col];
    assign w2_data_out_3 = W2_mem[w2_addr_row_3][w2_addr_col];

    assign b2_data_out = b2_mem[b2_addr];

endmodule