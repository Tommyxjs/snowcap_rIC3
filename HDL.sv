module OneHotLatch #(
)
(
input wire clk,
input wire [5:0] x,
output wire prop
);

    wire valid_input;
wire done;
reg [5:0] l;

    assign valid_input = (x != 0) && ((x & (x - 1)) == 0);
assign done = (l == 6'b111111);
assign prop = done;

    always @(*) begin
assume(valid_input);
end

    always @(posedge clk) begin
l <= l | x;
end

endmodule
