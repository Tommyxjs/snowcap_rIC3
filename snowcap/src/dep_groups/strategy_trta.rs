// Snowcap: Synthesizing Network-Wide Configuration Updates
// Copyright (C) 2021  Tibor Schneider
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

//! # One Strategy To Rule Them All

use super::utils;
use crate::hard_policies::{HardPolicy, PolicyError};
use crate::modifier_ordering::RandomOrdering;
use crate::netsim::config::{ConfigExpr::BgpSession, ConfigModifier};
use crate::netsim::{Network, NetworkError, RouterId};
use crate::strategies::{PushBackTreeStrategy, Strategy};
use crate::{Error, Stopper};
use petgraph::matrix_graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::fmt::format;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::string;
use std::sync::mpsc;
use std::thread;

use log::*;
use rand::prelude::*;
use std::time::{Duration, Instant, SystemTime};
use utils::fmt_err;

/// # One Strategy To Rule Them All
///
/// This is the one strategy to rule them all, combining the best from the
/// [`TreeStrategy`](crate::strategies::TreeStrategy) and the
/// [`DepGroupsStrategy`](crate::strategies::DepGroupsStrategy) into one single strategy.
///
/// ## Description
///
/// The stretegy works by exploring the search space of all possible orderings in a tree-like
/// manner. This means, that it proceeds by taking one valid modifier, while building a tree of
/// which leaves need to be explored later. Once there are no other valid modifiers to try, we are
/// stuck.
///
/// When we are stuck, we try to solve the current problem by finding a dependency group. This
/// procedure is explained in detail [here](#detailed-explenation-of-finding-dependencies). If a
/// dependency with a valid solution could be found, then we reset the exploration tree, but with
/// the group now treated as one single modifier. If however no dependency group could be learned,
/// then backtrack in the current exploration tree, until we have either explored everything, or
/// found a valid solution.
///
/// ### Detailed Explenation of finding dependencies
///
/// When we are stuck, we try to solve the current problem by finding a dependency group. This is
/// done in three distinct phases. The input to all phases is the current ordering of all groups,
/// including the point where applying the first group fails.
///
/// 1. **Reduction Phase**: In reduciton phase, we try to eliminate groups that seem to have no
///    impact on the dependency group. To do this, we iterate over all groups in the orering (except
///    the problematic group), remove it temporarily from the ordering (to obtain `probe_ordering`)
///    and simulate this new ordering. If this removal has absolutely no effect on the outcome, we
///    call this group to be independent of the problem.
///
///    Notice, that it might happen that the new ordering now fails at an earlier position. In this
///    case, we recursively call `reduce` again, but with the probed group removed. Then, we ad the
///    group back to the beginning of the resulting ordering, but only if the call resulted in no
///    additional recursion.
///
///    The following pseudocode illustrates the procedure:
///
///    ```rust,ignore
///    fn reduce(ordering: Vec<Group>, error: Error) -> Vec<Group> {
///
///        for i in 0..ordering.len() - 1 {
///            // generate the probe ordering by removing the group at position i.
///            let probe_group = ordering[i];
///            let probe_ordering = ordering.clone().remove(i);
///
///            // check the new group
///            match check(probe_ordering) {
///                Ok(()) => {
///                    // Current ordering is dependent on the probed group, because it solves the
///                    // problem.
///                }
///                Err(new_error) if new_error.position != error.position() => {
///                    // The probed ordering fails at a different position! Recursive make the
///                    // problem smaller
///                    let reduced_ordering = reduce(probe_ordering[..new_error.position + 1]);
///                    // insert the probe group back (but only if the level of recursion is
///                    // only 1, i.e., the problem was not made smaller by the call to `reduce`
///                    // above.)
///                    reduced_ordering.insert(0, probe_group);
///                    return reduced_ordering;
///                }
///                Err(new_error) if new_error != error => {
///                    // Current ordering is dependent on the probed group, because it changes the
///                    // error
///                }
///                Err(new_error) if new_error == error => {
///                    // Current ordering is independent on the probed group, because it has no
///                    // effect on the outcome of the network. Remove the group indefinately from
///                    // the ordering.
///                    ordering.remove(i)
///                }
///            }
///        }
///        return ordering
///
///    }
///    ```
///
/// 2. **Solving Phase**: In this phase, we try to find a solution to the reduced problem. This is
///    done using the already existing [`TreeStrategy`](crate::strategies::TreeStrategy). If we can
///    find a valid solution to this problem, then we have found a dependency, and we add it to the
///    list of dependency groups, in the ordering that we have determined. However, if we cannot
///    find any valid solution, we go to step 3 and try to expand the group.
///
/// 3. **Expansion Phase**: In this phase, we try to expand the problem in order to still be abe to
///    find a valid solution. To do this, we iterate over all not yet used groups (excluding those
///    who have already been removed in the reduction phase) and try to place this group at every
///    possible position in the ordering. If the error changes at any point, where the probed grou
///    is moved to, then we add this group to the reduced problem. Once we have found one group with
///    which to extend the problem, we exit and go back to step 2.
///
///    There might be the case, where the probed group changes the problematic group (i.e., the
///    group where the problem happens). In this case, we insert the probed group into this position
///    and go back to step 1, reducing the problem even further.
///
///    The following pseudocode illustrates the procedure:
///
///    ```rust,ignore
///    fn expand(ordering: Vec<Group>, unused: Vec<Group>, error: Error) -> Result<Vec<Group>> {
///
///        // iterate over all unused groups
///        for probe_group in unused {
///            // iterate over all positions where this gorup might be added
///            for i in 0..ordering.range() {
///
///                // generate the probe ordering by removing the group at position i.
///                let probe_ordering = ordering.clone().insert(i, probe_group);
///
///                // check the new group
///                match check(probe_ordering) {
///                    Ok(()) => {
///                        // The probed group seems to be dependent. Add it to the group. Since we
///                        // know, that this is already a solved gorup, we can skip the solving
///                        // phase, and directly call this a new group
///                        return Finish(probe_group)
///                    }
///                    Err(new_error) if new_error.position != error.position() => {
///                        // The probed ordering fails at a different position! Go back to the
///                        // reduction phase and make the problem smaller!
///                        let reduced_ordering = reduce(probe_ordering[..new_error.position + 1]);
///                        // Now, continue to the solving phase
///                        return Ok(reduced_ordering);
///                    }
///                    Err(new_error) if new_error != error => {
///                        // Current ordering is dependent on the probed group, because it changes
///                        // the error. Continue to the solving phase
///                        group.insert(i, probe_group);
///                        return Ok(probe_group)
///                    }
///                    Err(new_error) if new_error == error => {
///                        // The probed ordering has the exact same effect as the original one.
///                        // Continue moving the probe group to other positions in the ordering,
///                        // or continue by going to the next unused group.
///                    }
///                }
///            }
///        }
///        return Err
///    }
///    ```
pub struct StrategyTRTA {
    net: Network,
    groups: Vec<Vec<ConfigModifier>>,
    hard_policy: HardPolicy,
    rng: ThreadRng,
    stop_time: Option<SystemTime>,
    max_group_solve_time: Option<Duration>,
    #[cfg(feature = "count-states")]
    num_states: usize,
    #[cfg(feature = "count-states")]
    seen_difficult_dependency: bool,
}

impl Strategy for StrategyTRTA {
    fn new(
        mut net: Network,
        modifiers: Vec<ConfigModifier>,
        mut hard_policy: HardPolicy,
        time_budget: Option<Duration>,
    ) -> Result<Box<Self>, Error> {
        // clear the undo stack
        net.clear_undo_stack();

        // check the state
        hard_policy.set_num_mods_if_none(modifiers.len());
        let mut fw_state = net.get_forwarding_state();
        hard_policy.step(&mut net, &mut fw_state)?;
        //有输出，check了1次
        // println!("Policies checked successfully.");
        // match hard_policy.step(&mut net, &mut fw_state) {
        //     Ok(_) => println!("Policies checked successfully."),
        //     Err(e) => println!("Failed to check policies: {:?}", e),
        // }

        if !hard_policy.check() {
            error!("Initial state errors::\n{}", fmt_err(&hard_policy.get_watch_errors(), &net));
            return Err(Error::InvalidInitialState);
        }

        // prepare the groups
        let mut groups: Vec<Vec<ConfigModifier>> = Vec::with_capacity(modifiers.len());
        for modifier in modifiers {
            groups.push(vec![modifier]);
            // println!("groups: {:?}", groups);
        }

        // prepare the timings
        let max_group_solve_time: Option<Duration> =
            time_budget.as_ref().map(|dur| *dur / super::TIME_FRACTION);
        let stop_time: Option<SystemTime> = time_budget.map(|dur| SystemTime::now() + dur);
        Ok(Box::new(Self {
            net,
            groups,
            hard_policy,
            rng: rand::thread_rng(),
            stop_time,
            max_group_solve_time,
            #[cfg(feature = "count-states")]
            num_states: 0,
            #[cfg(feature = "count-states")]
            seen_difficult_dependency: false,
        }))
    }

    fn work(&mut self, mut abort: Stopper) -> Result<Vec<ConfigModifier>, Error> {
        // setup the stack with a randomized frame
        let mut stack = vec![StackFrame::new(0..self.groups.len(), 0, &mut self.rng)];
        let mut current_sequence: Vec<usize> = vec![];

        // clone the network and the hard policies to work with them for the tree exploration
        let mut net = self.net.clone();
        let mut hard_policy = self.hard_policy.clone();

        // 创建一个空的 Vec 来存储(source, target)元组对
        let mut session_pairs = Vec::new();
        let mut in_session_pairs = Vec::new();
        let mut re_session_pairs = Vec::new();
        let mut mo_session_pairs = Vec::new();

        // 迭代 modifiers，提取 source 和 target
        let first_modifiers: Vec<ConfigModifier> = self
            .groups
            .iter()
            .filter_map(|group| group.first().cloned()) // 获取每个group的第一个modifier
            .collect();
        // println!("first_modifiers: {:?}", first_modifiers);
        for (index, modifier) in first_modifiers.iter().enumerate() {
            match modifier {
                // 处理 Remove 和 Insert 两种情况
                ConfigModifier::Insert(BgpSession { source, target, .. }) => {
                    in_session_pairs.push(index);
                    session_pairs.push((*source, *target));
                }
                ConfigModifier::Remove(BgpSession { source, target, .. }) => {
                    re_session_pairs.push(index);
                    session_pairs.push((*source, *target));
                }
                ConfigModifier::Update { to: BgpSession { source, target, .. }, .. } => {
                    mo_session_pairs.push(index);
                    session_pairs.push((*source, *target));
                }
                _ => {}
            }
        }

        let mut router_counts: HashMap<NodeIndex<u32>, usize> = HashMap::new();
        let total_mo_pairs = mo_session_pairs.len();

        for &session_idx in &mo_session_pairs {
            let (src_router, tgt_router) = session_pairs[session_idx];
            *router_counts.entry(src_router).or_insert(0) += 1;
            *router_counts.entry(tgt_router).or_insert(0) += 1;
        }

        // 查找唯一一个在每个 mo_session_pair 中都出现的路由器：即路由反射器
        let reflector_router = router_counts
            .iter()
            .find(|&(_, &count)| count == total_mo_pairs)
            .map(|(router, _)| *router);

        //打印出生成的列表
        // println!("Session pairs: {:?}", session_pairs);
        // println!("in_session_pairs: {:?}", in_session_pairs);
        // println!("re_session_pairs: {:?}", re_session_pairs);
        // println!("mo_session_pairs: {:?}", mo_session_pairs);
        // println!("reflector_router: {:?}", reflector_router);

        //最终目的是产生aalta_input，送进aalta中，但是循环的是ltl_string
        let mut ltl_string = "True".to_string();
        //构建每个状态只能做一个update的约束
        // let mut formula_parts = Vec::new();
        let mut constraints: Vec<(Vec<usize>, usize)> = vec![];
        let mut old_order = Vec::new();

        //for i in 0..frame.rem_groups.len() {
        // 24.1.9 for i in 0..self.groups.len() {
        // for i in 0..self.groups.len() {
        //     let mut negations = Vec::new();
        //     for j in 0..self.groups.len() {
        //         // for j in 0..self.groups.len() {
        //         if i != j {
        //             negations.push(format!("! x{}", j));
        //         }
        //     }
        //     // let formula = format!("G(x{} -> ({}))", i, negations.join(" & "));
        //     let formula = format!("G((x{} & {}) <-> (e{}))", i, negations.join(" & "), i);
        //     formula_parts.push(formula);
        // }
        // let mut always_formula_parts = formula_parts.join(" & ");
        // always_formula_parts.push_str(&format!(
        //     " & G({}) & {}\n",
        //     (0..self.groups.len())
        //         .map(|i| format!("((e{}) & N(G(! e{})))", i, i))
        //         .collect::<Vec<_>>()
        //         .join(" | "),
        //     (0..self.groups.len() - 1).fold("true".to_string(), |acc, _| format!("X({})", acc))
        // ));
        // always_formula_parts.push_str(&format!(
        //     "{}\n",
        //     (0..20).map(|i| format!("& (F (x{}))", i)).collect::<Vec<_>>().join(" ")
        // ));

        // always_formula_parts.push_str(&format!(
        //     "{}\n",
        //     (0..self.groups.len()).map(|i| format!("& (F (x{}))", i)).collect::<Vec<_>>().join(" ")
        // ));
        // println!("formula_parts: {:?}", always_formula_parts);

        let mut counter = 0;
        loop {
            counter += 1;
            // check for iter overflow检查时间是否已耗尽（即处理时间是否超时）
            if self.stop_time.as_ref().map(|time| time.elapsed().is_ok()).unwrap_or(false) {
                // time budget is used up!
                error!("Time budget is used up! No solution was found yet!");
                return Err(Error::Timeout);
            }

            // check for abort criteria检查是否有中止请求
            if abort.try_is_stop().unwrap_or(false) {
                info!("Operation was aborted!");
                return Err(Error::Abort);
            }

            // get the latest stack frame获取当前堆栈帧（用于管理待处理的组）
            let frame = match stack.last_mut() {
                Some(frame) => {
                    // println!("Current frame: {:?}", frame);
                    frame
                }
                None => {
                    error!("Could not find any valid ordering!");
                    return Err(Error::ProbablyNoSafeOrdering);
                }
            };
            // println!("self.groups.len(): {}", self.groups.len());
            // 假设 done_updates 是一个 Vec<usize>，需要提前定义
            // let mut done_updates = Vec::new();
            //let mut doing_update = *frame.rem_groups.get(frame.idx).unwrap();

            // 将 doing_update 添加到 done_updates 列表中
            // done_updates.push(doing_update);

            // 输出 done_updates 列表
            // println!("Done updates: {:?}", done_updates);
            let mut indices = Vec::new();
            // 构造一个新约束集合，以解决fm2rr出现loop错误无法迭代更新序列
            // search the current stack frame for the next        // 查找当前堆栈帧的下一步操作
            let action: StackAction = match self.get_next_option(&mut net, &mut hard_policy, frame)
            {
                Ok(next_idx) => {
                    // update the current stack frame and prepare the next one
                    frame.idx = next_idx + 1;
                    // There exists a valid next step! Update the current sequence and the stack
                    let next_group_idx = frame.rem_groups[next_idx];
                    current_sequence.push(next_group_idx);
                    // println!("current_sequence.len(): {}", current_sequence.len());
                    // check if all groups have been added to the sequence
                    if current_sequence.len() == self.groups.len() {
                        // We are done! found a valid solution!
                        info!(
                            "Valid solution was found! Learned {} groups",
                            self.groups.iter().filter(|g| g.len() > 1).count()
                        );
                        return Ok(utils::finalize_ordering(&self.groups, &current_sequence));
                    }

                    // Prepare the stack action with the new stack frame
                    StackAction::Push(StackFrame {
                        rem_groups: frame
                            .rem_groups
                            .iter()
                            .cloned()
                            .filter(|x| *x != next_group_idx)
                            .collect(),
                        idx: 0, // 这里我们明确设置 idx 为 0 或者根据逻辑需要的特定值
                        num_undo: self.groups[next_group_idx].len(), // 不使用随机数生成器
                    })
                    // StackAction::Push(StackFrame::new(
                    //     frame.rem_groups.iter().cloned().filter(|x| *x != next_group_idx),
                    //     self.groups[next_group_idx].len(),
                    //     &mut self.rng,
                    // ))
                }
                Err(NetworkError::ForwardingBlackHole(check_idx)) => {
                    println!("Now we have the Extracted NodeIndices: {:?}", check_idx);
                    // let mut formulas = Vec::new();
                    for node in check_idx.iter() {
                        println!("node {:?}", node);
                        // let mut node_formulas = Vec::new();
                        let mut selected_indices = Vec::new();
                        //遍历有问题的节点
                        let mut matched_indices: Vec<usize> = Vec::new(); //把涉及有问题节点的更新的下标取出来
                                                                          // 遍历 session_pairs 并检查是否匹配
                        for (i, (source, target)) in session_pairs.iter().enumerate() {
                            if *source == *node || *target == *node {
                                // 如果匹配，则将下标存储到 matched_indices
                                matched_indices.push(i);
                            }
                        }
                        // println!("Matched session indices: {:?}", matched_indices);
                        //每次取出来，如果不等于已经执行的更新，添加约束
                        for &index in &matched_indices {
                            // 确保 index 不等于 current_sequence 中的任何一项，并且不等于 frame.rem_groups.get(0)并且是一个插入类型的更新
                            if !(current_sequence.contains(&index)
                                || (0..=frame.idx).any(|i| frame.rem_groups.get(i) == Some(&index)))
                            {
                                if in_session_pairs.contains(&index) {
                                    selected_indices.push(index);
                                    // let formula = format!("N(G(! e{:?}))", index);
                                    // node_formulas.push(formula);
                                } else if mo_session_pairs.contains(&index) {
                                    // let formula = format!("N(G(! e{:?}))", index);
                                    // node_formulas.push(formula);
                                    selected_indices.push(index);
                                    // 获取边界路由器（与 node 不同的那一个）
                                    let (source_router, target_router) =
                                        session_pairs[*frame.rem_groups.get(frame.idx).unwrap()];
                                    let border_router = if source_router == *node {
                                        target_router
                                    } else {
                                        source_router
                                    };
                                    if let Some(reflector_router) = reflector_router {
                                        // 在 mo_session_pairs 中寻找是否存在 (border_router, reflector_router) 或其反向
                                        let mut related_update_index: usize = 0;
                                        let mut found = false;

                                        for &mo_index in &mo_session_pairs {
                                            let (s, t) = session_pairs[mo_index];
                                            if (s == border_router && t == reflector_router)
                                                || (s == reflector_router && t == border_router)
                                            {
                                                related_update_index = mo_index;
                                                found = true;
                                                break;
                                            }
                                        }

                                        if found {
                                            println!(
                                                "Found matching MO session update between border router {:?} and reflector {:?} at index {}",
                                                border_router, reflector_router, related_update_index
                                            );

                                            // 你可以在这里进行进一步操作，比如记录、构造公式等
                                            // let formula =
                                            //     format!("N(G(! e{:?}))", related_update_index);
                                            // node_formulas.push(formula);
                                            selected_indices.push(related_update_index);
                                        } else {
                                            println!(
                                                "No matching update found for border router {:?} and reflector {:?}",
                                                border_router, reflector_router
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        if !selected_indices.is_empty() {
                            // 将每个 node 的公式用括号包裹，并连接起来
                            // let combined_node_formula = format!(
                            //     "G(e{} -> ({}))",
                            //     *frame.rem_groups.get(frame.idx).unwrap(),
                            //     node_formulas.join(" | ")
                            // );
                            // formulas.push(combined_node_formula); // 添加到总公式集合
                            constraints.push((
                                selected_indices.clone(),
                                *frame.rem_groups.get(frame.idx).unwrap(),
                            ));
                        }
                    }
                    println!("constraints:{:?}", constraints);

                    //如果formulas为空，那么要么在没做的更新里面没有直连的bgp session可以解决问题，没做的更新集合为空
                    // let combined_formula = if formulas.is_empty() {
                    //     // 初始化 combined_formula
                    //     let mut combined_formula =
                    //         format!("e{}", *frame.rem_groups.get(frame.idx).unwrap());

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len()).rev() {
                    //         combined_formula =
                    //             format!("e{} & X({})", current_sequence[i], combined_formula);
                    //     }

                    //     // 给 combined_formula 添加 ! 外围
                    //     format!("!({})", combined_formula)
                    // } else {
                    //     formulas.join(" & ").to_string()
                    // };

                    // println!("Combined formula: {}", combined_formula);

                    // // 如果当前执行的更新为最后一个更新，但是无效，此时学习不到任何约束，但仍需阻止当前更新序列
                    // if self.groups.len() == current_sequence.len() + 1 {
                    //     // 构建 LTL 公式，从 current_sequence 的最后一个元素开始
                    //     let mut blocked_sequence =
                    //         format!("x{}", *frame.rem_groups.get(frame.idx).unwrap());

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len()).rev() {
                    //         blocked_sequence =
                    //             format!("x{} & X(F({}))", current_sequence[i], blocked_sequence);
                    //     }
                    //     blocked_sequence = format!("!({})", blocked_sequence);

                    //     // 打印构建好的 LTL 公式
                    //     println!("Blocked sequence LTL formula: {}", blocked_sequence);

                    //     combined_formula = format!("{} & {}", combined_formula, blocked_sequence);
                    // }

                    // 构造LTL公式的条件
                    // let prefix = if current_sequence.is_empty() {
                    //     "True".to_string()
                    // } else {
                    //     // 初始的 LTL 表达式是 prefix 中最后一个元素
                    //     let mut prefix_expr =
                    //         format!("x{}", current_sequence[current_sequence.len() - 1]);

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len() - 1).rev() {
                    //         // 这里用 prefix_list.len() - 1
                    //         prefix_expr =
                    //             format!("x{} & X(F({}))", current_sequence[i], prefix_expr);
                    //     }
                    //     prefix_expr
                    // };

                    // 打印生成的 LTL 表达式
                    // println!("Generated LTL Prefix: {}", prefix);
                    // let prefix = "True".to_string();
                    //(prefix -> (combined_formula)) & ltl_string & always_formula_parts & F x0 & F x1 & F x2 & F x3 & F x4 & F x5
                    // ltl_string = format!("({}) & {}", combined_formula, ltl_string);
                    // let aalta_input = format!("({}) & {}", ltl_string, always_formula_parts);
                    // println!("aalta_input: {}", aalta_input);
                    // let mut file = File::create(
                    //     "/home/xu/Documents/snowcap-CDCL/target/debug/aalta_input.txt",
                    // )
                    // .unwrap();
                    // file.write_all(aalta_input.as_bytes()).unwrap();
                    // file.flush().unwrap();

                    //Generate Verilog code
                    let mut verilog = String::new();
                    let width = self.groups.len();
                    println!("self.groups.len(): {}", self.groups.len());
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

                    // 生成所需的 prev 寄存器
                    let mut declared_prev = HashSet::new();
                    for (befores, _) in &constraints {
                        for &b in befores {
                            if declared_prev.insert(b) {
                                verilog.push_str(&format!("    reg l{}_prev;\n", b));
                            }
                        }
                    }

                    verilog.push_str(&format!(
                        "\n    assign valid_input = (x != 0) && ((x & (x - 1)) == 0);\n\
                    assign done = (l == {}'b{});\n\
                    assign prop = done;\n\
                    wire assume_ok = !assume_failed;\n\n",
                        width,
                        "1".repeat(width)
                    ));

                    verilog.push_str(
                        "    always @(*) begin\n\
                        assume(valid_input);\n\
                    end\n\
                    always @(*) begin\n\
                        assume(assume_ok);\n\
                    end\n\n",
                    );

                    verilog.push_str(
                        "    always @(posedge clk) begin\n\
                        l <= l | x;\n\
                    end\n\n",
                    );

                    // 生成顺序约束逻辑
                    for (befores, after) in &constraints {
                        let conds: Vec<String> =
                            befores.iter().map(|b| format!("l{}_prev", b)).collect();
                        let cond_str = conds.join(" || ");
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

                    // 写入文件
                    let path = Path::new("/home/xu/Documents/ver/snowcap-CDCL/HDL.sv");
                    let mut file = File::create(path).expect("无法创建文件");
                    file.write_all(verilog.as_bytes()).expect("无法写入 Verilog 内容");

                    // println!("✅ Verilog 文件已更新：{}", path.display());
                    //已经生成了verilog代码，接下来yosys->rIC3->parser
                    // Step 1: 执行 yosys generate_aiger.ys
                    let yosys_status = Command::new("yosys")
                        .arg("generate_aiger.ys")
                        .current_dir("/home/xu/Documents/ver/snowcap-CDCL")
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .expect("Failed to execute yosys");

                    if !yosys_status.success() {
                        eprintln!("❌ Yosys 执行失败！");
                        std::process::exit(1);
                    }

                    // Step 2: 执行 rIC3 --witness design_aiger.aag，并捕获输出
                    let ric3_output = Command::new("../rIC3/target/release/rIC3")
                        .arg("--witness")
                        .arg("design_aiger.aag")
                        .current_dir("/home/xu/Documents/ver/snowcap-CDCL")
                        .output()
                        .expect("❌ Failed to execute rIC3");

                    let stdout_str = String::from_utf8_lossy(&ric3_output.stdout);
                    let stderr_str = String::from_utf8_lossy(&ric3_output.stderr);

                    // Step 3: 写入日志到指定路径
                    let log_path =
                        Path::new("/home/xu/Documents/ver/snowcap-CDCL/output_to_file.txt");
                    let mut log_file = File::create(log_path).expect("❌ 无法创建日志文件");

                    writeln!(log_file, "=== STDOUT ===\n{}", stdout_str)
                        .expect("❌ 写入 stdout 失败");
                    writeln!(log_file, "\n=== STDERR ===\n{}", stderr_str)
                        .expect("❌ 写入 stderr 失败");

                    // println!("✅ rIC3 执行完成，日志已保存到: {}", log_path.display());

                    // Step 4: parser，解析 STDOUT
                    let output_content = fs::read_to_string(log_path).expect("❌ 无法读取日志文件");

                    let stdout_marker = "=== STDOUT ===";
                    let stdout_start = output_content
                        .find(stdout_marker)
                        .map(|idx| idx + stdout_marker.len())
                        .expect("❌ 未找到 STDOUT 区域");

                    let stdout_end =
                        output_content.find("=== STDERR ===").unwrap_or(output_content.len());
                    let stdout_section = &output_content[stdout_start..stdout_end];

                    // Step 5: 查找 result: safe/unsafe
                    let result_line = stdout_section
                        .lines()
                        .find(|line| line.trim_start().starts_with("result:"))
                        .unwrap_or("");

                    if result_line.contains("safe") || result_line.contains("unsafe") {
                        let lines: Vec<&str> = stdout_section.lines().collect();
                        // println!("✅ STDOUT 区域的所有行如下：");
                        // for (i, line) in lines.iter().enumerate() {
                        //     println!("{}: {}", i, line);
                        // }

                        // 提取 witness 比特串：从第一个全0行之后开始，直到点符号.的前两行
                        let mut start_index = None;
                        let mut dot_index = None;

                        for (i, line) in lines.iter().enumerate() {
                            let trimmed = line.trim();
                            if start_index.is_none()
                                && !trimmed.is_empty()
                                && trimmed.chars().all(|c| c == '0')
                            {
                                start_index = Some(i + 1);
                            }
                            if trimmed == "." {
                                dot_index = Some(i);
                                break;
                            }
                        }

                        let mut extracted_lines = Vec::new();
                        if let (Some(start), Some(dot)) = (start_index, dot_index) {
                            let end = dot.saturating_sub(2);
                            for i in start..=end {
                                let line = lines[i].trim();
                                if line.starts_with('1')
                                    && line.chars().all(|c| c == '0' || c == '1')
                                    && line[1..].contains('1')
                                {
                                    extracted_lines.push(line);
                                }
                            }
                        }

                        // 打印提取结果
                        // println!("✅ 提取比特串如下:");
                        // for line in &extracted_lines {
                        //     println!("{}", line);
                        // }

                        // 提取比特串中除首位外 '1' 的下标
                        for line in &extracted_lines {
                            let bits = line.trim();
                            if bits.len() < 2 {
                                continue;
                            }
                            let tail_bits = &bits[1..];
                            if let Some(pos) = tail_bits.chars().position(|c| c == '1') {
                                indices.push(pos);
                            }
                        }
                        println!("✅ ");

                        // println!("✅ 提取出的 '1' 所在下标为: {:?}", indices);
                    } else {
                        println!("⚠️ 未检测到 safe 或 unsafe 结果。");
                    }

                    // //新建子线程
                    // let timeout = Duration::new(120, 0);
                    // let (tx, rx) = mpsc::channel();

                    // let handle = thread::spawn(move || {
                    //     let mut child = Command::new("/home/xu/Documents/aaltaf/aaltaf") // 替换为你的可执行文件名
                    //         .arg("-e")
                    //         .stdout(Stdio::piped())
                    //         .spawn()
                    //         .expect("Failed to start process");

                    //     let output = child.stdout.take().expect("Failed to open stdout");
                    //     let mut output_str = String::new();
                    //     let mut reader = BufReader::new(output);
                    //     reader.read_to_string(&mut output_str).expect("Failed to read stdout");

                    //     // 将子进程的输出发送到主线程
                    //     tx.send(output_str).expect("Failed to send output");
                    // });
                    // let start_time = Instant::now();
                    // // 主线程等待 5分钟或子进程输出
                    // let result = loop {
                    //     // 检查是否超时
                    //     if start_time.elapsed() >= timeout {
                    //         println!("Timeout reached, returning unsat.");
                    //         return Err(Error::Timeout);
                    //         // 如果超时，发送 unsat 并退出
                    //         break "unsat".to_string();
                    //     }

                    //     // 检查子进程是否已经返回结果
                    //     match rx.try_recv() {
                    //         Ok(output_str) => {
                    //             println!("Process output: {}", output_str);
                    //             break output_str; // 返回子进程的输出
                    //         }
                    //         Err(_) => {
                    //             // 如果没有收到消息，继续循环，等待超时或子进程结果
                    //             thread::sleep(Duration::from_millis(100)); // 避免 CPU 占用过高
                    //         }
                    //     }
                    // };

                    // let mut child = Command::new(
                    //     "/public/home/jwli/workSpace/xjs/sigcomm_exp_24/aaltaf/aaltaf",
                    // ) // 替换为你的可执行文件名
                    // .arg("-e")
                    // // .stdin(Stdio::piped())
                    // .stdout(Stdio::piped())
                    // .spawn()
                    // .expect("Failed to start process");

                    // {
                    //     // 获取子进程的标准输入(aalta_input)
                    //     let stdin = child.stdin.as_mut().expect("Failed to open stdin");
                    //     stdin.write_all(aalta_input.as_bytes()).expect("Failed to write to stdin");
                    //     stdin.flush().expect("Failed to flush stdin");
                    // }
                    // println!("Finish aalta input!");
                    // let output = child.stdout.take().expect("Failed to open stdout");
                    // println!("Begin aalta output!");
                    // 使用 BufReader 来读取输出
                    // let mut output_str = String::new();
                    // let mut reader = BufReader::new(output);
                    // reader
                    //     .read_to_string(&mut output_str)
                    //     .expect("Failed to read stdout");
                    // println!("Output: {}", output_str);

                    // 对aalta的输出进行解析
                    // let lines: Vec<&str> = result.lines().collect();
                    // let mut indices = Vec::new();
                    // 检查结果是否为sat
                    // if lines.len() > 1 && lines[0].trim() == "sat" {
                    // if lines[0].trim() == "sat" {
                    //     // 对从第三行之后的结果进行处理
                    //     let aalta_output_path =
                    //         "/home/xu/Documents/snowcap-CDCL/target/debug/output_to_file.txt";
                    //     let content =
                    //         fs::read_to_string(aalta_output_path).expect("Failed to read file");
                    //     let aalta_output: Vec<&str> = content.lines().collect();
                    //     for line in aalta_output.iter() {
                    //         // skip first two lines (header and "sat")
                    //         // 按，分片
                    //         for part in line.split(",") {
                    //             let trimmed = part.trim();

                    //             // 检索所有以x开始的变量
                    //             if trimmed.starts_with("x") {
                    //                 // Extract the number after "x", whether or not it is preceded by "!"
                    //                 if let Some(index_str) = trimmed.strip_prefix("x") {
                    //                     if let Ok(index) = index_str.trim().parse::<usize>() {
                    //                         // Push the extracted index to the indices vector
                    //                         indices.push(index);
                    //                     }
                    //                 }
                    //                 //检索在开头但是以（x开始的变量
                    //             } else if trimmed.starts_with("(x") {
                    //                 // Handle the case for "(xN" where N is the index we want
                    //                 if let Some(index_str) = trimmed.strip_prefix("(x") {
                    //                     if let Ok(index) = index_str.trim().parse::<usize>() {
                    //                         indices.push(index);
                    //                     }
                    //                 }
                    //             }
                    //         }
                    //     }

                    //     // 输出解析出来的更新序列
                    //     println!("Extracted indices: {:?}", indices);
                    // } else {
                    //     println!("Second line is not 'sat', skipping extraction.");
                    //     return Err(Error::ProbablyNoSafeOrdering);
                    // }
                    // 等待子进程完成
                    // let exit_status = child.wait().expect("Child process wasn't running");

                    // 打印子进程的退出状态
                    // println!("Child exited with status: {}", exit_status);

                    // // 直接退出主进程
                    // std::process::exit(exit_status.code().unwrap_or(1));
                    // let stack_frame = StackFrame { idx: 0, rem_groups: indices, num_undo: 0 };

                    // 将新的栈帧添加到栈中
                    // stack.push(stack_frame);
                    StackAction::Reset
                    // StackAction::Push(stack_frame)

                    // #[cfg(feature = "count-states")]
                    // {
                    //     self.seen_difficult_dependency = true;
                    // }
                    // // There exists no option, that we can take, which would lead to a good result!
                    // // First, we set the next index to the length of the options, in order to
                    // // remember that we have checked everything
                    // frame.idx = frame.rem_groups.len();
                    // // What we do here is try to find a dependency!
                    // match self.find_dependency(
                    //     &mut net,
                    //     &mut hard_policy,
                    //     &current_sequence,
                    //     frame.rem_groups[check_idx],
                    //     abort.clone(),
                    // ) {
                    //     Some((new_group, old_groups)) => {
                    //         info!("Found a new dependency group!");
                    //         // add the new ordering to the known groups
                    //         utils::add_minimal_ordering_as_new_gorup(
                    //             &mut self.groups,
                    //             old_groups,
                    //             Some(new_group),
                    //         );
                    //         // reset the stack frame
                    //         StackAction::Reset
                    //     }
                    //     None => {
                    //         // No dependency group could be found! Continue exploring the search
                    //         // space
                    //         info!("Could not find a new dependency group!");
                    //         StackAction::Pop
                    //     }
                    // }
                }
                Err(NetworkError::ForwardingLoops(check_idx)) => {
                    println!("Now we have the Extracted NodeIndices: {:?}", check_idx);
                    // let mut combined_formula_from_loop = Vec::new();
                    println!("old constraints:{:?}", constraints);
                    if constraints
                        .iter()
                        .any(|(_, target)| *target == *frame.rem_groups.get(frame.idx).unwrap())
                    {
                        println!("1111111111111111111111");
                        for (vec, target) in constraints
                            .iter_mut()
                            .filter(|(_, t)| *t == *frame.rem_groups.get(frame.idx).unwrap())
                        {
                            if let Some(target_pos) = old_order.iter().position(|&x| x == *target) {
                                println!("Target {:?} found at position {}", target, target_pos);

                                vec.retain(|&item| {
                                    let keep = if let Some(item_pos) =
                                        old_order.iter().position(|&x| x == item)
                                    {
                                        println!(
                                            " - Item {:?} at position {} vs target_pos {}",
                                            item, item_pos, target_pos
                                        );
                                        item_pos > target_pos
                                    } else {
                                        println!(
                                            " - Item {:?} not found in old_order, keeping it",
                                            item
                                        );
                                        true
                                    };
                                    keep
                                });
                            } else {
                                println!("Target {:?} 不在 old_order 中，跳过 retain", target);
                            }
                        }
                    } else {
                        println!("2222222222222222");
                        for forwardingloop in check_idx.iter() {
                            println!("forwardingloop {:?}", forwardingloop);
                            // let mut node_formulas = Vec::new();
                            let mut selected_indices = Vec::new();
                            let mut matched_indices: Vec<usize> = Vec::new(); //把涉及有问题节点的更新的下标取出来
                            for node in forwardingloop.iter() {
                                // 遍历有问题的节点
                                // 遍历 session_pairs 并检查是否匹配
                                println!("node {:?}", node);
                                for (i, (source, target)) in session_pairs.iter().enumerate() {
                                    if (*source == *node || *target == *node)
                                        && !matched_indices.contains(&i)
                                    {
                                        // 如果匹配，则将下标存储到 matched_indices
                                        matched_indices.push(i);
                                    }
                                }
                                // println!("Matched session indices: {:?}", matched_indices);
                            }
                            let mut fm2rr = true;
                            //每次取出来，如果不等于已经执行的更新，添加约束
                            for &index in &matched_indices {
                                // 确保 index 不等于 current_sequence 中的任何一项，并且不等于 frame.rem_groups.get(0)
                                // if !(current_sequence.contains(&index)
                                //     || (0..=frame.idx)
                                //         .any(|i| frame.rem_groups.get(i) == Some(&index)))
                                if !(0..=frame.idx).any(|i| frame.rem_groups.get(i) == Some(&index))
                                {
                                    if in_session_pairs.contains(&index) {
                                        selected_indices.push(index);
                                        // let formula = format!("N(G(! e{:?}))", index);
                                        // node_formulas.push(formula);
                                    } else if mo_session_pairs.contains(&index) {
                                        // let formula = format!("N(G(! e{:?}))", index);
                                        // node_formulas.push(formula);
                                        println!("-------------------");
                                        selected_indices.push(index);
                                        fm2rr = true;
                                    }
                                }
                            }
                            if fm2rr == true {
                                println!("3333333333333");
                                // 获取环路中除错误更新相关的路由器外其他路由器的下标
                                let (source_router, target_router) =
                                    session_pairs[*frame.rem_groups.get(frame.idx).unwrap()];
                                let border_router =
                                    if forwardingloop.iter().any(|n| *n == source_router) {
                                        target_router
                                    } else {
                                        source_router
                                    };
                                if let Some(reflector_router) = reflector_router {
                                    // Step 1: 收集所有 forwardingloop 中不等于 source/target 的节点
                                    let other_nodes: Vec<_> = forwardingloop
                                        .iter()
                                        .filter(|n| **n != source_router && **n != target_router)
                                        .collect();

                                    // Step 2: 遍历每个 other_node，再查找匹配的 re_session_pair
                                    let mut related_update_index: Vec<usize> = Vec::new();

                                    for other_node in other_nodes {
                                        for &re_index in &re_session_pairs {
                                            let (s, t) = session_pairs[re_index];

                                            if (s == border_router && t == *other_node)
                                                || (s == *other_node && t == border_router)
                                            {
                                                related_update_index.push(re_index);
                                                break; // 如果你只想每个 other_node 匹配一次，就 break；否则可以删掉这行
                                            }
                                        }
                                    }

                                    // Step 3: 输出结果
                                    if !related_update_index.is_empty() {
                                        println!(
                                            "Found matching RE session updates between border router {:?} and reflector {:?} at indices {:?}",
                                            border_router, reflector_router, related_update_index
                                        );

                                        selected_indices.extend(related_update_index);
                                    } else {
                                        println!(
                                            "No matching update found for border router {:?} and reflector {:?}",
                                            border_router, reflector_router
                                        );
                                    }
                                }
                            }
                            if !selected_indices.is_empty() {
                                // 将每个 node 的公式用括号包裹，并连接起来
                                // let combined_node_formula = format!(
                                //     "G(e{} -> ({}))",
                                //     *frame.rem_groups.get(frame.idx).unwrap(),
                                //     node_formulas.join(" | ")
                                // );
                                // combined_formula_from_loop.push(combined_node_formula);
                                // println!(
                                //     "combined_formula_from_loop: {:?}",
                                //     combined_formula_from_loop
                                // );
                                // 添加到总公式集合
                                constraints.push((
                                    selected_indices.clone(),
                                    *frame.rem_groups.get(frame.idx).unwrap(),
                                ));
                            }
                        }
                    }
                    println!("constraints:{:?}", constraints);

                    //如果combined_formula_from_loop为空，那么要么在没做的更新里面没有直连的bgp session可以解决问题，没做的更新集合为空
                    // let combined_formula_form_loops = if combined_formula_from_loop.is_empty() {
                    //     // 初始化 combined_formula
                    //     let mut combined_formula =
                    //         format!("e{}", *frame.rem_groups.get(frame.idx).unwrap());

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len()).rev() {
                    //         combined_formula =
                    //             format!("e{} & X({})", current_sequence[i], combined_formula);
                    //     }

                    //     // 给 combined_formula 添加 ! 外围
                    //     format!("!({})", combined_formula)
                    // } else {
                    //     combined_formula_from_loop.join(" & ").to_string()
                    // };

                    // println!("combined_formula_form_loops: {}", combined_formula_form_loops);

                    // // 如果当前执行的更新为最后一个更新，但是无效，此时学习不到任何约束，但仍需阻止当前更新序列
                    // if self.groups.len() == current_sequence.len() + 1 {
                    //     // 构建 LTL 公式，从 current_sequence 的最后一个元素开始
                    //     let mut blocked_sequence =
                    //         format!("x{}", *frame.rem_groups.get(frame.idx).unwrap());

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len()).rev() {
                    //         blocked_sequence =
                    //             format!("x{} & X(F({}))", current_sequence[i], blocked_sequence);
                    //     }
                    //     blocked_sequence = format!("!({})", blocked_sequence);

                    //     // 打印构建好的 LTL 公式
                    //     println!("Blocked sequence LTL formula: {}", blocked_sequence);

                    //     combined_formula_form_loops =
                    //         format!("{} & {}", combined_formula_form_loops, blocked_sequence);
                    // }

                    // 构造LTL公式的条件
                    // let prefix = if current_sequence.is_empty() {
                    //     "True".to_string()
                    // } else {
                    //     // 初始的 LTL 表达式是 prefix 中最后一个元素
                    //     let mut prefix_expr =
                    //         format!("x{}", current_sequence[current_sequence.len() - 1]);

                    //     // 从倒数第二个元素到第一个元素构建 LTL 表达式
                    //     for i in (0..current_sequence.len() - 1).rev() {
                    //         // 这里用 prefix_list.len() - 1
                    //         prefix_expr =
                    //             format!("x{} & X(F({}))", current_sequence[i], prefix_expr);
                    //     }
                    //     prefix_expr
                    // };

                    // 打印生成的 LTL 表达式
                    // println!("Generated LTL Prefix: {}", prefix);
                    // let prefix = "True".to_string();

                    //(prefix -> (combined_formula_form_loops)) & ltl_string & always_formula_parts & F x0 & F x1 & F x2 & F x3 & F x4 & F x5
                    // ltl_string = format!("({}) & {}", combined_formula_form_loops, ltl_string);
                    // let aalta_input = format!("({}) & {}", ltl_string, always_formula_parts);
                    // println!("aalta_input: {}", aalta_input);
                    // let mut file = File::create(
                    //     "/home/xu/Documents/snowcap-CDCL/target/debug/aalta_input.txt",
                    // )
                    // .unwrap();
                    // file.write_all(aalta_input.as_bytes()).unwrap();
                    // file.flush().unwrap();

                    // Generate Verilog code
                    let mut verilog = String::new();
                    let width = self.groups.len();
                    println!("self.groups.len(): {}", width);

                    // 模块头
                    verilog.push_str(&format!(
                        "module OneHotLatch #(\n\
                        ) (\n\
                            input wire clk,\n\
                            input wire [{}:0] x,\n\
                            output wire prop\n\
                        );\n\n",
                        width - 1
                    ));

                    // 信号定义
                    verilog.push_str(&format!(
                        "    wire valid_input;\n\
                        wire done;\n\
                        reg [{}:0] l;\n\n",
                        width - 1
                    ));

                    // 校验逻辑
                    verilog.push_str(&format!(
                        "    assign valid_input = (x != 0) && ((x & (x - 1)) == 0);\n\
                        assign done = (l == {}'b{});\n\
                        assign prop = done;\n\n",
                        width,
                        "1".repeat(width)
                    ));

                    // assume 输入合法
                    verilog.push_str(
                        "    always @(*) begin\n\
                        assume(valid_input);\n\
                    end\n\n",
                    );

                    // latch 状态更新逻辑
                    verilog.push_str(
                        "    always @(posedge clk) begin\n\
                        l <= l | x;\n\
                    end\n\n",
                    );

                    // 生成顺序约束逻辑
                    for (befores, after) in &constraints {
                        if befores.is_empty() {
                            // 无前置：after 永远不应被激活
                            verilog.push_str(&format!(
                                "    always @(*) begin\n\
                                assume(!x[{}]);\n\
                            end\n\n",
                                after
                            ));
                        } else {
                            let conds: Vec<String> =
                                befores.iter().map(|b| format!("l[{}]", b)).collect();
                            let or_cond = conds.join(" || ");
                            verilog.push_str(&format!(
                                "    always @(*) begin\n\
                                assume(!(x[{}] && !({})));\n\
                            end\n\n",
                                after, or_cond
                            ));
                        }
                    }

                    // 模块结束
                    verilog.push_str("endmodule\n");

                    // 写入文件
                    let path = Path::new("/home/xu/Documents/ver/snowcap-CDCL/HDL.sv");

                    let mut file = File::create(path).expect("无法创建文件");
                    file.write_all(verilog.as_bytes()).expect("无法写入 Verilog 内容");

                    // println!("✅ Verilog 文件已更新：{}", path.display());
                    //已经生成了verilog代码，接下来yosys->rIC3->parser
                    // Step 1: 执行 yosys generate_aiger.ys
                    let yosys_status = Command::new("yosys")
                        .arg("generate_aiger.ys")
                        .current_dir("/home/xu/Documents/ver/snowcap-CDCL")
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .expect("Failed to execute yosys");

                    if !yosys_status.success() {
                        eprintln!("❌ Yosys 执行失败！");
                        std::process::exit(1);
                    }

                    // Step 2: 执行 rIC3 --witness design_aiger.aag，并捕获输出
                    let ric3_output = Command::new("../rIC3/target/release/rIC3")
                        .arg("--witness")
                        .arg("design_aiger.aag")
                        .current_dir("/home/xu/Documents/ver/snowcap-CDCL")
                        .output()
                        .expect("❌ Failed to execute rIC3");

                    let stdout_str = String::from_utf8_lossy(&ric3_output.stdout);
                    let stderr_str = String::from_utf8_lossy(&ric3_output.stderr);

                    // Step 3: 写入日志到指定路径
                    let log_path =
                        Path::new("/home/xu/Documents/ver/snowcap-CDCL/output_to_file.txt");
                    let mut log_file = File::create(log_path).expect("❌ 无法创建日志文件");

                    writeln!(log_file, "=== STDOUT ===\n{}", stdout_str)
                        .expect("❌ 写入 stdout 失败");
                    writeln!(log_file, "\n=== STDERR ===\n{}", stderr_str)
                        .expect("❌ 写入 stderr 失败");

                    // println!("✅ rIC3 执行完成，日志已保存到: {}", log_path.display());

                    // Step 4: parser，解析 STDOUT
                    let output_content = fs::read_to_string(log_path).expect("❌ 无法读取日志文件");

                    let stdout_marker = "=== STDOUT ===";
                    let stdout_start = output_content
                        .find(stdout_marker)
                        .map(|idx| idx + stdout_marker.len())
                        .expect("❌ 未找到 STDOUT 区域");

                    let stdout_end =
                        output_content.find("=== STDERR ===").unwrap_or(output_content.len());
                    let stdout_section = &output_content[stdout_start..stdout_end];

                    // Step 5: 查找 result: safe/unsafe
                    let result_line = stdout_section
                        .lines()
                        .find(|line| line.trim_start().starts_with("result:"))
                        .unwrap_or("");

                    if result_line.contains("safe") || result_line.contains("unsafe") {
                        let lines: Vec<&str> = stdout_section.lines().collect();
                        // println!("✅ STDOUT 区域的所有行如下：");
                        // for (i, line) in lines.iter().enumerate() {
                        //     println!("{}: {}", i, line);
                        // }

                        // 提取 witness 比特串：从第一个全0行之后开始，直到点符号.的前两行
                        let mut start_index = None;
                        let mut dot_index = None;

                        for (i, line) in lines.iter().enumerate() {
                            let trimmed = line.trim();
                            if start_index.is_none()
                                && !trimmed.is_empty()
                                && trimmed.chars().all(|c| c == '0')
                            {
                                start_index = Some(i + 1);
                            }
                            if trimmed == "." {
                                dot_index = Some(i);
                                break;
                            }
                        }

                        let mut extracted_lines = Vec::new();
                        if let (Some(start), Some(dot)) = (start_index, dot_index) {
                            let end = dot.saturating_sub(2);
                            for i in start..=end {
                                let line = lines[i].trim();
                                if line.starts_with('1')
                                    && line.chars().all(|c| c == '0' || c == '1')
                                    && line[1..].contains('1')
                                {
                                    extracted_lines.push(line);
                                }
                            }
                        }

                        // 打印提取结果
                        // println!("✅ 提取比特串如下:");
                        // for line in &extracted_lines {
                        //     println!("{}", line);
                        // }

                        // 提取比特串中除首位外 '1' 的下标
                        for line in &extracted_lines {
                            let bits = line.trim();
                            if bits.len() < 2 {
                                continue;
                            }
                            let tail_bits = &bits[1..];
                            if let Some(pos) = tail_bits.chars().position(|c| c == '1') {
                                indices.push(pos);
                            }
                        }
                        println!("✅ ");

                        // println!("✅ 提取出的 '1' 所在下标为: {:?}", indices);
                    } else {
                        println!("⚠️ 未检测到 safe 或 unsafe 结果。");
                    }

                    // //time
                    // let start_time = Instant::now();

                    // //新建子线程
                    // let timeout = Duration::new(5, 0);
                    // let (tx, rx) = mpsc::channel();

                    // let handle = thread::spawn(move || {
                    //     let mut child = Command::new("/home/xu/Documents/aaltaf/aaltaf") // 替换为你的可执行文件名
                    //         .arg("-e")
                    //         .stdout(Stdio::piped())
                    //         .spawn()
                    //         .expect("Failed to start process");

                    //     let output = child.stdout.take().expect("Failed to open stdout");
                    //     let mut output_str = String::new();
                    //     let mut reader = BufReader::new(output);
                    //     reader.read_to_string(&mut output_str).expect("Failed to read stdout");

                    //     // 将子进程的输出发送到主线程
                    //     tx.send(output_str).expect("Failed to send output");
                    // });
                    // let start_time = Instant::now();
                    // // 主线程等待 5分钟或子进程输出
                    // let result = loop {
                    //     // 检查是否超时
                    //     if start_time.elapsed() >= timeout {
                    //         println!("Timeout reached, returning unsat.");
                    //         return Err(Error::Timeout);
                    //         // 如果超时，发送 unsat 并退出
                    //         break "unsat".to_string();
                    //     }

                    //     // 检查子进程是否已经返回结果
                    //     match rx.try_recv() {
                    //         Ok(output_str) => {
                    //             println!("Process output: {}", output_str);
                    //             break output_str; // 返回子进程的输出
                    //         }
                    //         Err(_) => {
                    //             // 如果没有收到消息，继续循环，等待超时或子进程结果
                    //             thread::sleep(Duration::from_millis(100)); // 避免 CPU 占用过高
                    //         }
                    //     }
                    // };

                    // let mut child = Command::new(
                    //     "/public/home/jwli/workSpace/xjs/sigcomm_exp_24/aaltaf/aaltaf",
                    // ) // 替换为你的可执行文件名
                    // .arg("-e")
                    // // .stdin(Stdio::piped())
                    // .stdout(Stdio::piped())
                    // .spawn()
                    // .expect("Failed to start process");

                    // {
                    //     // 获取子进程的标准输入(aalta_input)
                    //     let stdin = child.stdin.as_mut().expect("Failed to open stdin");
                    //     stdin.write_all(aalta_input.as_bytes()).expect("Failed to write to stdin");
                    //     stdin.flush().expect("Failed to flush stdin");
                    // }
                    // println!("Finish aalta input!");
                    //let output = child.stdout.take().expect("Failed to open stdout");
                    // println!("Begin aalta output!");
                    // 使用 BufReader 来读取输出
                    //let mut output_str = String::new();
                    //let mut reader = BufReader::new(output);
                    // reader
                    //     .read_to_string(&mut output_str)
                    //     .expect("Failed to read stdout");
                    // println!("Output: {}", output_str);

                    // 对aalta的输出进行解析
                    // let lines: Vec<&str> = result.lines().collect();
                    // let mut indices = Vec::new();
                    // 检查结果是否为sat
                    // if lines.len() > 1 && lines[0].trim() == "sat" {
                    //     // 对从第三行之后的结果进行处理
                    //     for line in lines.iter().skip(1) {
                    //         // skip first two lines (header and "sat")
                    //         // 按，分片
                    //         for part in line.split(",") {
                    //             let trimmed = part.trim();

                    //             // 检索所有以x开始的变量
                    //             if trimmed.starts_with("x") {
                    //                 // Extract the number after "x", whether or not it is preceded by "!"
                    //                 if let Some(index_str) = trimmed.strip_prefix("x") {
                    //                     if let Ok(index) = index_str.trim().parse::<usize>() {
                    //                         // Push the extracted index to the indices vector
                    //                         indices.push(index);
                    //                     }
                    //                 }
                    //                 //检索在开头但是以（x开始的变量
                    //             } else if trimmed.starts_with("(x") {
                    //                 // Handle the case for "(xN" where N is the index we want
                    //                 if let Some(index_str) = trimmed.strip_prefix("(x") {
                    //                     if let Ok(index) = index_str.trim().parse::<usize>() {
                    //                         indices.push(index);
                    //                     }
                    //                 }
                    //             }
                    //         }
                    //     }
                    //     // 输出解析出来的更新序列
                    //     // println!("Extracted indices: {:?}", indices);
                    // } else {
                    //     println!("Second line is not 'sat', skipping extraction.");
                    //     return Err(Error::ProbablyNoSafeOrdering);
                    // }
                    // // 等待子进程完成
                    // let exit_status = child.wait().expect("Child process wasn't running");

                    // 打印子进程的退出状态
                    // println!("Child exited with status: {}", exit_status);

                    // // 直接退出主进程
                    // std::process::exit(exit_status.code().unwrap_or(1));
                    // let stack_frame = StackFrame { idx: 0, rem_groups: indices, num_undo: 0 };

                    // 将新的栈帧添加到栈中
                    // stack.push(stack_frame);
                    StackAction::Reset
                    // StackAction::Push(stack_frame)

                    // #[cfg(feature = "count-states")]
                    // {
                    //     self.seen_difficult_dependency = true;
                    // }
                    // // There exists no option, that we can take, which would lead to a good result!
                    // // First, we set the next index to the length of the options, in order to
                    // // remember that we have checked everything
                    // frame.idx = frame.rem_groups.len();
                    // // What we do here is try to find a dependency!
                    // match self.find_dependency(
                    //     &mut net,
                    //     &mut hard_policy,
                    //     &current_sequence,
                    //     frame.rem_groups[check_idx],
                    //     abort.clone(),
                    // ) {
                    //     Some((new_group, old_groups)) => {
                    //         info!("Found a new dependency group!");
                    //         // add the new ordering to the known groups
                    //         utils::add_minimal_ordering_as_new_gorup(
                    //             &mut self.groups,
                    //             old_groups,
                    //             Some(new_group),
                    //         );
                    //         // reset the stack frame
                    //         StackAction::Reset
                    //     }
                    //     None => {
                    //         // No dependency group could be found! Continue exploring the search
                    //         // space
                    //         info!("Could not find a new dependency group!");
                    //         StackAction::Pop
                    //     }
                    // }
                }
                _ => StackAction::Reset,
            };

            // at this point, the mutable reference to `stack` (i.e., `frame`) is dropped, which
            // means that `stack` is no longer borrowed exclusively.

            match action {
                StackAction::Pop => {
                    // pop the stack, as long as the top frame has no options left
                    'backtrace: while let Some(frame) = stack.last() {
                        if frame.idx < frame.rem_groups.len() {
                            break 'backtrace;
                        } else {
                            // undo the net, the hard policy and pop the current sequence
                            current_sequence.pop();
                            (0..frame.num_undo).for_each(|_| {
                                net.undo_action().expect("Cannot undo the action on the network");
                                hard_policy.undo();
                            });
                            // pop the stack
                            stack.pop();
                        }
                    }
                }
                StackAction::Push(new_frame) => stack.push(new_frame),
                StackAction::Reset => {
                    // reset the stack for the new groups, as well as the sequence, the network and
                    // the hard policies
                    stack = vec![StackFrame::new(0..self.groups.len(), 0, &mut self.rng)];
                    current_sequence.clear();
                    net = self.net.clone();
                    hard_policy = self.hard_policy.clone();
                }
            }

            if !indices.is_empty() {
                // 检查 indices 是否为空
                old_order = indices.clone();
                let stack_frame = StackFrame {
                    idx: 0,
                    rem_groups: indices, // 使用 indices
                    num_undo: 0,
                };
                stack.push(stack_frame); // 将新构造的 stack_frame 推入栈中
            }

            // if let StackAction::Reset = action.clone() {
            //     // 构造新的 StackFrame
            //     let stack_frame = StackFrame {
            //         idx: 0,
            //         rem_groups: indices, // 这里 indices 在上面已经获取
            //         num_undo: 0,
            //     };
            //     stack.push(stack_frame);
            // }
        }
    }

    #[cfg(feature = "count-states")]
    fn num_states(&self) -> usize {
        self.num_states
    }
}

impl StrategyTRTA {
    /// Check all remaining possible choices at the current position in the stack. The first option,
    /// that works is returned (with `Ok(idx)`). However, if none of them seem to work, then one of
    /// the checked and failed groups is returned at random, which should be used to find a
    /// dependency group. The returned index corresponds to the position in `frame.rem_groups`!
    ///
    /// In the OK case, the network and the hard policy will remain in the state of the modification
    /// of which the index is returned
    fn get_next_option(
        &mut self,
        net: &mut Network,
        hard_policy: &mut HardPolicy,
        frame: &StackFrame,
    ) -> Result<usize, NetworkError> {
        assert!(frame.idx < frame.rem_groups.len());
        for group_pos in frame.idx..frame.rem_groups.len() {
            let group_idx = *frame.rem_groups.get(group_pos).unwrap();
            // println!("Done updates: {:?}", group_idx);

            // perform the modification group
            let mut mod_ok: bool = true;
            let mut num_undo: usize = 0;
            let mut num_undo_policy: usize = 0;
            let mut blackhole_error_node_indices: Option<Vec<RouterId>> = None;
            let mut forwarding_loop_error_node_indices: Option<Vec<Vec<RouterId>>> = None;
            'apply_group: for modifier in self.groups[group_idx].iter() {
                #[cfg(feature = "count-states")]
                {
                    self.num_states += 1;
                }
                num_undo += 1;
                if net.apply_modifier(modifier).is_ok() {
                    num_undo_policy += 1;
                    let mut fw_state = net.get_forwarding_state();
                    // hard_policy.step(net, &mut fw_state).expect("cannot check policies!");
                    // println!("--------------1---------------!");
                    let result = hard_policy.step(net, &mut fw_state);
                    // println!("--------------2---------------!");
                    match result {
                        Ok(_) => {} //println!("Policies checked successfully!"),
                        Err(e) => {
                            println!("Error checking policies: {:?}", e);
                            match e {
                                NetworkError::ForwardingBlackHole(node_indices) => {
                                    println!(
                                        "Extracted NodeIndices from BlackHole: {:?}",
                                        node_indices
                                    );
                                    blackhole_error_node_indices = Some(node_indices.clone());
                                }
                                NetworkError::ForwardingLoops(path) => {
                                    println!("Extracted Path from ForwardingLoop: {:?}", path);
                                    forwarding_loop_error_node_indices = Some(path.clone());
                                    // 如果逻辑需要也存储路径
                                }
                                // 可以处理其他错误类型
                                _ => {
                                    println!("Unhandled Error Type: {:?}", e);
                                }
                            }
                        }
                    }
                    //如果不满足性质
                    if !hard_policy.check() {
                        mod_ok = false;
                        break 'apply_group;
                    }
                    //如果不能做这些更新
                } else {
                    mod_ok = false;
                    break 'apply_group;
                }
            }
            // println!("mod_ok:{}", mod_ok);
            // check if the modifier is ok
            if mod_ok {
                // everything fine, return the index
                return Ok(group_pos);
            } else {
                // undo the hard policy and the network
                (0..num_undo_policy).for_each(|_| hard_policy.undo());
                (0..num_undo).for_each(|_| {
                    net.undo_action().expect("Cannot perform undo!");
                });
                if let Some(node_indices) = blackhole_error_node_indices {
                    return Err(NetworkError::ForwardingBlackHole(node_indices));
                    // 直接解构 Some 并返回内部的 Vec
                }
                if let Some(path) = forwarding_loop_error_node_indices {
                    return Err(NetworkError::ForwardingLoops(path)); // 直接解构 Some 并返回内部的 Vec
                }
            }
        }
        Err(NetworkError::NoConvergence)
        // if we reach this position, we know that every possible option is bad!
        //Err(self.rng.gen_range(frame.idx, frame.rem_groups.len()))
    }

    /// This function tries to find a dependency based on the current position. The arguments
    /// are as follows:
    ///
    /// - `net`: Network at state of the good ordering. After returning, the net will have the exact
    ///   same state as before.
    /// - `hard_policy`: Hard Policy at state of the good ordering. After returning, the hard policy
    ///   will have the exact same state as before.
    /// - `good_ordering`: Ordering of groups, which work up to the point of the bad group
    /// - `bad_group`: Index of the bad group which causes the problme. This function will search a
    ///    dependency to solve this bad group.
    ///
    /// If a dependency was found successfully, then this function will return the new dependency
    /// (first argument), along with the set of groups that are part of this new dependency (second
    /// argument). If no dependency group could be found, then `None` is returned.
    fn find_dependency(
        &mut self,
        net: &mut Network,            // 引用网络对象，表示当前的网络状态
        hard_policy: &mut HardPolicy, // 引用硬性策略对象，表示当前的策略状态
        good_ordering: &[usize],      // 一个引用，代表“良好”的组序列（即没有引发错误的顺序）
        bad_group: usize,             // 一个引发问题的组索引，表示需要修复的“坏组”
        abort: Stopper,               // 停止标志，用于中止操作的信号
    ) -> Option<(Vec<ConfigModifier>, Vec<usize>)> {
        // apply the modifier to the network to get the errors
        let mut num_undo = 0;
        let mut num_undo_policy = 0;
        let mut errors = None;
        'apply_group: for modifier in self.groups[bad_group].iter() {
            num_undo += 1;
            if net.apply_modifier(modifier).is_ok() {
                //应用更新
                num_undo_policy += 1;
                let mut fw_state = net.get_forwarding_state(); //获取网络的转发状态
                hard_policy.step(net, &mut fw_state).expect("cannot check policies!");
                //发现执行以下四行无log，应该是没有进入这个函数
                // println!("Policies checked successfully.");
                // match hard_policy.step(net, &mut fw_state) {
                //     Ok(_) => println!("Policies checked successfully."),
                //     Err(e) => println!("Failed to check policies: {:?}", e),
                // }

                if !hard_policy.check() {
                    errors = Some(hard_policy.get_watch_errors());
                    break 'apply_group;
                }
            } else {
                errors = Some((Vec::new(), vec![Some(PolicyError::NoConvergence)]));
                break 'apply_group;
            }
        }

        // undo the hard policy and the network
        (0..num_undo_policy).for_each(|_| hard_policy.undo());
        (0..num_undo).for_each(|_| {
            net.undo_action().expect("Cannot perform undo!");
        });

        match errors {
            // 如果有错误发生，尝试寻找依赖组来解决这个“坏组”
            Some(errors) => {
                // 将良好排序加上坏组，形成新的组顺序

                let ordering = good_ordering
                    .iter()
                    .cloned() // 克隆良好排序的每个元素
                    .chain(std::iter::once(bad_group)) // 将坏组加到顺序末尾
                    .collect::<Vec<usize>>();
                utils::find_dependency::<PushBackTreeStrategy<RandomOrdering>>(
                    &self.net,
                    &self.groups,
                    &self.hard_policy,
                    &ordering,
                    errors,
                    self.stop_time,
                    self.max_group_solve_time,
                    abort,
                    #[cfg(feature = "count-states")]
                    &mut self.num_states,
                )
            }
            // 如果没有错误发生，抛出异常，表明传入的坏组实际上是“好的”（没有问题）
            None => panic!("The bad group, passed into this function seems to be fine!"),
        }
    }

    /// Returns true if, during exploration, we encountered a dependency without immediate effect.
    ///
    /// *This method is only available if the `"count-states"` feature is enabled!*
    #[cfg(feature = "count-states")]
    pub fn seen_dependency_without_immediage_effect(&self) -> bool {
        self.seen_difficult_dependency
    }
}

#[derive(Debug, Clone)]
enum StackAction {
    Pop,
    Push(StackFrame),
    Reset,
}

/// Single stack frame for the iteration
#[derive(Debug, Clone)]
struct StackFrame {
    /// Number of calls to undo, in order to undo this step
    num_undo: usize,
    /// Remaining groups to try at this position
    rem_groups: Vec<usize>,
    /// index into rem_groups to check next, after all previous branches have been explroed.
    idx: usize,
}

impl StackFrame {
    fn new(options: impl Iterator<Item = usize>, num_undo: usize, rng: &mut ThreadRng) -> Self {
        let mut rem_groups: Vec<usize> = options.collect();
        rem_groups.shuffle(rng);
        Self { num_undo, rem_groups, idx: 0 }
    }
}
