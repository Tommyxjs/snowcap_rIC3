module OneHotLatch #()
(input wire clk,
input wire [104:0] x,
output wire prop
);

    wire valid_input;
wire done;
reg [104:0] l;
reg assume_failed;

    reg l96_prev;
    reg l101_prev;

    assign valid_input = (x != 0) && ((x & (x - 1)) == 0);
assign done = (l == 105'b111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111);
assign prop = done;
wire assume_ok = !assume_failed;

    always @(*) begin
assume(valid_input);
end
always @(*) begin
assume(assume_ok);
end

    always @(posedge clk) begin
l <= l | x;
end

    always @(posedge clk) begin
if (valid_input && l[89] && !(l96_prev || l101_prev))
assume_failed <= 1;
        l96_prev <= l[96];
        l101_prev <= l[101];
    end

endmodule
