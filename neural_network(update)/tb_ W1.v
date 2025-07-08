module input_layer #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_SIZE  = 784,a
    parameter HIDDEN_SIZE = 128
)(
    input  clk,
    input  rst_n,
    // Tín hiệu điều khiển từ neural_network_core
    input  start_processing_layer,
    input  signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1], // Dữ liệu đầu vào không đổi

    // Trọng số cho MỘT hidden node hiện tại đang được xử lý
    input  signed [DATA_WIDTH-1:0] W1_current_node_weights_in [0:INPUT_SIZE-1],
    input  signed [DATA_WIDTH-1:0] b1_current_node_bias_in,
    input  weights_for_current_node_valid, // Báo rằng W1_..._in và b1_..._in là mới và hợp lệ

    // Tín hiệu điều khiển tới neural_network_core và kết quả
    output reg layer_processing_done,        // Báo cho core biết đã xử lý xong toàn bộ lớp
    output reg request_data_for_next_node,   // Yêu cầu core cung cấp trọng số cho node tiếp theo
    output reg [$clog2(HIDDEN_SIZE)-1:0] next_node_to_process_idx, // Chỉ số của hidden node tiếp theo

    output reg signed [DATA_WIDTH-1:0] output_data [0:HIDDEN_SIZE-1] // Kết quả của lớp
);

    localparam PARALLEL_FACTOR = 4;
    localparam FIXED_NUM_CHUNKS = 16;
    // Đảm bảo INPUT_SIZE chia hết cho FIXED_NUM_CHUNKS
    localparam FIXED_CHUNK_SIZE = INPUT_SIZE / FIXED_NUM_CHUNKS;
    localparam ACC_WIDTH = 24;
    localparam CHUNK_CNT_WIDTH = $clog2(FIXED_NUM_CHUNKS);
    // HIDDEN_CNT_WIDTH được xử lý bằng $clog2(HIDDEN_SIZE) trực tiếp
    localparam DELTA_VAL_WIDTH = 23;
    localparam K_LOOP_CNT_WIDTH = $clog2(FIXED_CHUNK_SIZE);

    reg [K_LOOP_CNT_WIDTH-1:0] k_loop_cnt;
    reg signed [ACC_WIDTH-1:0] acc_for_single_node; // Tích lũy cho hidden node hiện tại
    reg signed [ACC_WIDTH-1:0] temp_acc_with_bias;  // Để lưu kết quả sau khi cộng bias

    localparam S_IDLE                        = 5'b00000,
               S_INIT_LAYER                  = 5'b00001,
               S_WAIT_FOR_NODE_DATA          = 5'b00010,
               S_INIT_NODE_PROCESSING        = 5'b00011,
               S_CALC_DELTA_SETUP            = 5'b00100,
               S_CALC_DELTA_LOOP             = 5'b00101,
               S_ACCUMULATE_CHUNK_DELTA      = 5'b00110,
               S_NEXT_CHUNK_OR_FINISH_CHUNKS = 5'b00111,
               S_ADD_BIAS                    = 5'b01000, // State riêng cho cộng bias
               S_RELU_STORE                  = 5'b01001, // State riêng cho ReLU và lưu
               S_CHECK_FOR_NEXT_NODE         = 5'b01010,
               S_LAYER_DONE                  = 5'b01011;
    reg [4:0] state, next_state;

    reg [CHUNK_CNT_WIDTH-1:0] chunk_cnt;
    reg [$clog2(HIDDEN_SIZE)-1:0] current_hidden_node_idx;
    reg signed [DELTA_VAL_WIDTH-1:0] current_delta_sum_for_chunk;

    reg signed [2*DATA_WIDTH-1:0]     product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DATA_WIDTH:0]         scaled_product_seq [0:PARALLEL_FACTOR-1];
    reg signed [DELTA_VAL_WIDTH-1:0]  sum_of_scaled_products;

    integer current_i_base;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                       <= S_IDLE;
            layer_processing_done       <= 1'b0;
            request_data_for_next_node  <= 1'b0;
            next_node_to_process_idx    <= '0;
            current_hidden_node_idx     <= '0;
            chunk_cnt                   <= '0;
            k_loop_cnt                  <= '0;
            acc_for_single_node         <= '0;
            current_delta_sum_for_chunk <= '0;
            temp_acc_with_bias          <= '0;
            for (integer j_rst = 0; j_rst < HIDDEN_SIZE; j_rst = j_rst + 1) begin
                output_data[j_rst]      <= '0;
            end
        end else begin
            state <= next_state;
            layer_processing_done      <= 1'b0;
            request_data_for_next_node <= 1'b0; // Mặc định tắt, chỉ bật khi cần

            case (state)
                S_IDLE: begin
                    if (start_processing_layer) begin
                        next_state <= S_INIT_LAYER;
                    end else begin
                        next_state <= S_IDLE;
                    end
                end

                S_INIT_LAYER: begin
                    for (integer j_init = 0; j_init < HIDDEN_SIZE; j_init = j_init + 1) begin
                        output_data[j_init] <= '0;
                    end
                    current_hidden_node_idx  <= 0;
                    next_node_to_process_idx <= 0; // Yêu cầu dữ liệu cho node đầu tiên
                    request_data_for_next_node <= 1'b1;
                    next_state               <= S_WAIT_FOR_NODE_DATA;
                end

                S_WAIT_FOR_NODE_DATA: begin
                    // request_data_for_next_node đã được set ở state trước hoặc sẽ được giữ nếu cần
                    if (weights_for_current_node_valid) begin
                        next_state <= S_INIT_NODE_PROCESSING;
                    end else begin
                        // Tiếp tục yêu cầu nếu chưa nhận được và chưa xử lý hết node
                        if (current_hidden_node_idx < HIDDEN_SIZE) begin
                             request_data_for_next_node <= 1'b1;
                             next_node_to_process_idx   <= current_hidden_node_idx;
                        end
                        next_state <= S_WAIT_FOR_NODE_DATA;
                    end
                end

                S_INIT_NODE_PROCESSING: begin
                    acc_for_single_node <= '0; // Reset tích lũy cho node mới
                    chunk_cnt           <= 0;  // Reset bộ đếm chunk
                    next_state          <= S_CALC_DELTA_SETUP;
                end

                S_CALC_DELTA_SETUP: begin
                    k_loop_cnt                  <= 0;
                    current_delta_sum_for_chunk <= '0;
                    next_state                  <= S_CALC_DELTA_LOOP;
                end

                S_CALC_DELTA_LOOP: begin
                    current_i_base = (chunk_cnt * FIXED_CHUNK_SIZE) + k_loop_cnt;
                    sum_of_scaled_products = '0;
                    for (integer m = 0; m < PARALLEL_FACTOR; m = m + 1) begin
                        if (k_loop_cnt + m < FIXED_CHUNK_SIZE) begin
                            product_seq[m] = input_data[current_i_base + m] * W1_current_node_weights_in[current_i_base + m];
                            scaled_product_seq[m] = product_seq[m] >>> 12;
                            sum_of_scaled_products = sum_of_scaled_products +
                                                     {{(DELTA_VAL_WIDTH - (DATA_WIDTH+1)){scaled_product_seq[m][DATA_WIDTH]}}, scaled_product_seq[m]};
                        end
                    end
                    current_delta_sum_for_chunk <= current_delta_sum_for_chunk + sum_of_scaled_products;

                    if (k_loop_cnt + PARALLEL_FACTOR < FIXED_CHUNK_SIZE) begin
                        k_loop_cnt <= k_loop_cnt + PARALLEL_FACTOR;
                        next_state <= S_CALC_DELTA_LOOP;
                    end else begin
                        next_state <= S_ACCUMULATE_CHUNK_DELTA;
                    end
                end

                S_ACCUMULATE_CHUNK_DELTA: begin
                    acc_for_single_node <= acc_for_single_node +
                                           {{(ACC_WIDTH - DELTA_VAL_WIDTH){current_delta_sum_for_chunk[DELTA_VAL_WIDTH-1]}}, current_delta_sum_for_chunk};
                    next_state          <= S_NEXT_CHUNK_OR_FINISH_CHUNKS;
                end

                S_NEXT_CHUNK_OR_FINISH_CHUNKS: begin
                    if (chunk_cnt < FIXED_NUM_CHUNKS - 1) begin
                        chunk_cnt  <= chunk_cnt + 1;
                        next_state <= S_CALC_DELTA_SETUP;
                    end else begin
                        next_state <= S_ADD_BIAS; // Đã xử lý hết chunk, chuyển sang cộng bias
                    end
                end

                S_ADD_BIAS: begin
                    temp_acc_with_bias <= acc_for_single_node + {{ (ACC_WIDTH - DATA_WIDTH){b1_current_node_bias_in[DATA_WIDTH-1]} }, b1_current_node_bias_in};
                    next_state <= S_RELU_STORE;
                end

                S_RELU_STORE: begin
                    // Sử dụng temp_acc_with_bias đã được tính ở state trước
                    if (temp_acc_with_bias[ACC_WIDTH-1]) begin // Số âm
                        output_data[current_hidden_node_idx] <= '0;
                    end else begin // Số không âm
                        if (temp_acc_with_bias > ((1 << (DATA_WIDTH-1)) - 1) ) begin // Bão hòa dương
                            output_data[current_hidden_node_idx] <= ((1 << (DATA_WIDTH-1)) - 1);
                        end else begin
                            output_data[current_hidden_node_idx] <= temp_acc_with_bias[DATA_WIDTH-1:0];
                        end
                    end
                    next_state <= S_CHECK_FOR_NEXT_NODE;
                end
                 
                S_CHECK_FOR_NEXT_NODE: begin
                    if (current_hidden_node_idx < HIDDEN_SIZE - 1) begin
                        current_hidden_node_idx    <= current_hidden_node_idx + 1;
                        next_node_to_process_idx   <= current_hidden_node_idx + 1;
                        request_data_for_next_node <= 1'b1; // Yêu cầu cho node tiếp theo
                        next_state                 <= S_WAIT_FOR_NODE_DATA;
                    end else begin
                        // Đã xử lý tất cả các hidden node
                        next_state <= S_LAYER_DONE;
                    end
                end

                S_LAYER_DONE: begin
                    layer_processing_done <= 1'b1;
                    next_state            <= S_IDLE;
                end

                default: begin
                    next_state <= S_IDLE;
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
    input valid_in, // Từ top_module, báo hiệu bắt đầu xử lý một ảnh mới
    input signed [DATA_WIDTH-1:0] input_data [0:INPUT_SIZE-1],

    // Inputs for weights and biases from memory init modules (VẪN NHẬN TOÀN BỘ)
    input signed [DATA_WIDTH-1:0] w1_mem_data_in [0:INPUT_SIZE-1][0:HIDDEN_SIZE-1],
    input signed [DATA_WIDTH-1:0] b1_mem_data_in [0:HIDDEN_SIZE-1],
    input signed [DATA_WIDTH-1:0] w2_mem_data_in [0:HIDDEN_SIZE-1][0:OUTPUT_SIZE-1],
    input signed [DATA_WIDTH-1:0] b2_mem_data_in [0:OUTPUT_SIZE-1],

    output valid_out, // valid_out của toàn bộ core (từ output_layer)
    output signed [DATA_WIDTH-1:0] final_out [0:OUTPUT_SIZE-1],
    output [3:0] predicted_label
);

    // Tín hiệu điều khiển và dữ liệu cho input_layer
    reg  core_start_input_layer_processing; // Đổi tên từ start_processing_layer
    reg  core_weights_for_il_node_valid;
    // Dây để trích xuất cột W1 cho input_layer
    wire signed [DATA_WIDTH-1:0] core_W1_node_weights_to_il [0:INPUT_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] core_b1_node_bias_to_il;

    // Tín hiệu từ input_layer
    wire il_layer_processing_done;
    wire il_request_data_for_next_node;
    wire [$clog2(HIDDEN_SIZE)-1:0] il_next_node_to_process_idx;

    // Kết quả từ input_layer -> đầu vào của output_layer
    wire signed [DATA_WIDTH-1:0] hidden_out_core [0:HIDDEN_SIZE-1];

    // FSM cho việc cung cấp trọng số cho input_layer
    localparam CORE_CTRL_IDLE                     = 3'b000,
               CORE_CTRL_START_IL                 = 3'b001, // Kích hoạt input layer và cung cấp data node 0
               CORE_CTRL_PROVIDE_NEXT_NODE_DATA   = 3'b010, // Cung cấp data cho node được yêu cầu
               CORE_CTRL_WAIT_IL_FINISH_NODE      = 3'b011, // Chờ input layer xử lý xong node hiện tại
               CORE_CTRL_WAIT_IL_FINISH_LAYER     = 3'b100, // Chờ input layer xử lý xong toàn bộ layer
               CORE_CTRL_OUTPUT_LAYER_ACTIVE      = 3'b101; // Output layer đang chạy
    reg [2:0] core_ctrl_state, core_ctrl_next_state;
    reg [$clog2(HIDDEN_SIZE)-1:0] core_supplying_node_idx; // Node mà core đang cung cấp data cho

    // Trích xuất cột W1 cho hidden node hiện tại mà core đang cung cấp
    // core_supplying_node_idx là chỉ số cột (hidden node)
    generate
        genvar i_w1_col_slice;
        for (i_w1_col_slice = 0; i_w1_col_slice < INPUT_SIZE; i_w1_col_slice = i_w1_col_slice + 1) begin
            assign core_W1_node_weights_to_il[i_w1_col_slice] = w1_mem_data_in[i_w1_col_slice][core_supplying_node_idx];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_ctrl_state <= CORE_CTRL_IDLE;
            core_start_input_layer_processing <= 1'b0;
            core_weights_for_il_node_valid    <= 1'b0;
            core_supplying_node_idx           <= '0;
            core_b1_node_bias_to_il           <= '0;
        end else begin
            core_ctrl_state <= core_ctrl_next_state;
            // Mặc định các tín hiệu điều khiển
            core_start_input_layer_processing <= 1'b0;
            core_weights_for_il_node_valid    <= 1'b0;

            case (core_ctrl_state)
                CORE_CTRL_IDLE: begin
                    if (valid_in) begin // valid_in từ top module
                        core_ctrl_next_state <= CORE_CTRL_START_IL;
                    end else begin
                        core_ctrl_next_state <= CORE_CTRL_IDLE;
                    end
                end

                CORE_CTRL_START_IL: begin
                    core_start_input_layer_processing <= 1'b1; // Kích hoạt input_layer lần đầu
                    core_supplying_node_idx           <= 0;    // Bắt đầu từ node 0
                    // Dữ liệu W1 được gán bằng 'assign' ở trên
                    core_b1_node_bias_to_il           <= b1_mem_data_in[0]; // Bias cho node 0
                    core_weights_for_il_node_valid    <= 1'b1; // Báo data cho node 0 sẵn sàng
                    core_ctrl_next_state              <= CORE_CTRL_WAIT_IL_FINISH_NODE;
                end
                
                CORE_CTRL_WAIT_IL_FINISH_NODE: begin
                    core_weights_for_il_node_valid <= 1'b0; // Tín hiệu valid chỉ có hiệu lực 1 cycle
                    if (il_layer_processing_done) begin
                        // Input layer đã xử lý xong TẤT CẢ các node
                        core_ctrl_next_state <= CORE_CTRL_OUTPUT_LAYER_ACTIVE;
                    end else if (il_request_data_for_next_node) begin
                        // Input layer yêu cầu data cho node tiếp theo
                        core_ctrl_next_state <= CORE_CTRL_PROVIDE_NEXT_NODE_DATA;
                    end else begin
                        // Input layer đang bận xử lý node hiện tại, tiếp tục chờ
                        core_ctrl_next_state <= CORE_CTRL_WAIT_IL_FINISH_NODE;
                    end
                end

                CORE_CTRL_PROVIDE_NEXT_NODE_DATA: begin
                    core_supplying_node_idx        <= il_next_node_to_process_idx;
                    // W1 được cập nhật tự động qua 'assign'
                    core_b1_node_bias_to_il        <= b1_mem_data_in[il_next_node_to_process_idx];
                    core_weights_for_il_node_valid <= 1'b1;
                    core_ctrl_next_state           <= CORE_CTRL_WAIT_IL_FINISH_NODE; // Quay lại chờ IL xử lý
                end
                
                // CORE_CTRL_WAIT_IL_FINISH_LAYER: // State này không cần thiết nữa nếu logic trên đúng

                CORE_CTRL_OUTPUT_LAYER_ACTIVE: begin
                    // Input layer đã xong, output_layer sẽ được kích hoạt bởi il_layer_processing_done
                    // Chờ output_layer hoàn thành (thông qua valid_out của toàn bộ core)
                    if (valid_out) begin // Khi output_layer xong (valid_out của core)
                        core_ctrl_next_state <= CORE_CTRL_IDLE; // Sẵn sàng cho valid_in tiếp theo
                    end else begin
                        core_ctrl_next_state <= CORE_CTRL_OUTPUT_LAYER_ACTIVE;
                    end
                end

                default: core_ctrl_next_state <= CORE_CTRL_IDLE;
            endcase
        end
    end

    // Instantiate input_layer
    input_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE)
    ) u_input_layer_core (
        .clk(clk),
        .rst_n(rst_n),
        .start_processing_layer(core_start_input_layer_processing),
        .input_data(input_data),
        .W1_current_node_weights_in(core_W1_node_weights_to_il),
        .b1_current_node_bias_in(core_b1_node_bias_to_il),
        .weights_for_current_node_valid(core_weights_for_il_node_valid),
        .layer_processing_done(il_layer_processing_done),
        .request_data_for_next_node(il_request_data_for_next_node),
        .next_node_to_process_idx(il_next_node_to_process_idx),
        .output_data(hidden_out_core) // Kết quả từ input_layer
    );

    // Instantiate output_layer
    // output_layer vẫn giữ nguyên cách nạp W2, b2 như cũ (toàn bộ một lần)
    output_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) u_output_layer_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(il_layer_processing_done), // Kích hoạt output_layer khi input_layer xong
        .input_data(hidden_out_core),        // Dùng kết quả từ input_layer
        .W2_in(w2_mem_data_in),              // Vẫn nạp toàn bộ W2
        .b2_in(b2_mem_data_in),              // Vẫn nạp toàn bộ b2
        .valid_out(valid_out),               // valid_out của toàn bộ core
        .output_data(final_out),
        .predicted_label(predicted_label)
    );

endmodule

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