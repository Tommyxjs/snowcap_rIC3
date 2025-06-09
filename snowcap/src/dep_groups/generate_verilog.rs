fn generate_verilog(width: usize, constraints: &[(Vec<usize>, usize)]) -> String {
    let mut verilog = String::new();

    // 模块开头
    verilog.push_str(&format!(
        "module OneHotLatch #(\
)\n(\
    input wire clk,\n\
    input wire [{}:0] x,\n\
    output wire prop\n\
);\n\n",
        width - 1
    ));

    verilog.push_str(&format!(
        "    wire valid_input;\n\
    wire done;\n\
    reg [{}:0] l;\n\
    reg assume_failed;\n\n",
        width - 1
    ));

    // 为所有用到的 latch 添加 prev 变量
    let mut declared_prev = std::collections::HashSet::new();
    for (befores, _) in constraints {
        for &b in befores {
            if declared_prev.insert(b) {
                verilog.push_str(&format!("    reg l{}_prev;\n", b));
            }
        }
    }

    // assign块
    verilog.push_str(&format!(
        "\n    assign valid_input = (x != 0) && ((x & (x - 1)) == 0);\n\
    assign done = (l == {}'b{});\n\
    assign prop = done;\n\
    wire assume_ok = !assume_failed;\n\n",
        width,
        "1".repeat(width)
    ));

    // assume块
    verilog.push_str(
        "    always @(*) begin\n\
        assume(valid_input);\n\
    end\n\
    always @(*) begin\n\
        assume(assume_ok);\n\
    end\n\n",
    );

    // latch 更新逻辑
    verilog.push_str(
        "    always @(posedge clk) begin\n\
        l <= l | x;\n\
    end\n\n",
    );

    // 约束逻辑
    for (befores, after) in constraints {
        let mut conds = vec![];
        for &b in befores {
            conds.push(format!("l{}_prev", b));
        }
        let cond_str = conds.join(" || ");
        for &b in befores {
            verilog.push_str(&format!("    reg l{}_prev;\n", b)); // 确保声明
        }
        verilog.push_str(&format!(
            "    always @(posedge clk) begin\n\
        if (valid_input && l[{}] && !({}))\n\
            assume_failed <= 1;\n",
            after, cond_str
        ));
        for &b in befores {
            verilog.push_str(&format!("        l{}_prev <= l[{}];\n", b, b));
        }
        verilog.push_str("    end\n\n");
    }

    verilog.push_str("endmodule\n");
    verilog
}
