module cla32(a,b,ci,s); //32-bitcarrylookaheadadder,nog,poutputs
 input [31:0]a,b; //inputs:a,b
 input ci; //input: carry_in
 output[31:0]s; //output:sum
 wire g_out,p_out; //internalwires
 cla_32 cla(a,b,ci,g_out,p_out,s); //usecla_32module
 endmodule
