#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import argparse
from pace.graph.graph_utils import (
    children_,
)

QLINEAR_TO_ZENQLINEAR_MAPPING = {
    "quantized::linear": "pace::qlinear2d",
}


def get_zenqlinear_node(orig_op: str) -> str:
    return QLINEAR_TO_ZENQLINEAR_MAPPING[orig_op]


def get_qlinear_mapping():
    return QLINEAR_TO_ZENQLINEAR_MAPPING


def optimize_embedding_bags_q_per_tensor_nodes(graph):
    """
    This method will check if the following pattern is found
    and if found, aggregate all the aten::quantize_per_tensor
    nodes into a single aten::quantize_per_tensor node
    Snippet of original graph :
    %1 = aten::quantize_per_tensor(%emb_output_1, %scale, %zpoint, %dtype)
    %2 = aten::quantize_per_tensor(%emb_output_2, %scale, %zpoint, %dtype)
    %3 = aten::quantize_per_tensor(%emb_output_3, %scale, %zpoint, %dtype)
    ...
    %452 : prim::ListConstruct(%dense_arch_model__mlp_4.1, %1, %2, %3.....)
    %cat.1 : aten::cat(%452, %454)
    %459 : int[] = prim::ListConstruct(%86, %458)
    %460 = aten::reshape(%cat.1, %459)
    Snippet of converted graph :
    %452 : prim::ListConstruct(%616, %emb_output_1, %emb_output_2, ........)
    %cat.1 : aten::cat(%452, %454)
    %459 : prim::ListConstruct(%86, %458)
    %460 : aten::reshape(%cat.1, %459)
    %615 : aten::quantize_per_tensor(%460, %97, %98, %61)
    Args:
        graph : scripted model graph
    """
    list_nodes = graph.findAllNodes("prim::ListConstruct")
    quantize_nodes = graph.findAllNodes("aten::quantize_per_tensor")
    emb_nodes = graph.findAllNodes("quantized::embedding_bag_4bit")
    num_embeddings = 26
    for i in range(num_embeddings):
        list_nodes[0].replaceInput(i + 1, emb_nodes[i].output())

    new_node = graph.create("aten::quantize_per_tensor")
    reshape_nodes = graph.findAllNodes("aten::reshape")
    new_node.addInput(reshape_nodes[0].output())

    new_node.addInput(quantize_nodes[1].inputsAt(1))
    new_node.addInput(quantize_nodes[1].inputsAt(2))
    new_node.addInput(quantize_nodes[1].inputsAt(3))

    graph.setInsertPoint(reshape_nodes[0])
    new_node.insertAfter(reshape_nodes[0])

    qlinear2d_nodes = graph.findAllNodes("pace::qlinear2d")

    qlinear2d_nodes[0].replaceInput(0, new_node.output())

    qlinear2d_relu_nodes = graph.findAllNodes("quantized::linear_relu")
    dequant_node = graph.create("aten::dequantize")
    dequant_node.addInput(qlinear2d_relu_nodes[2].output())

    graph.setInsertPoint(qlinear2d_relu_nodes[2])
    list_nodes[0].replaceInput(0, dequant_node.output())

    dequant_node.insertAfter(qlinear2d_relu_nodes[2])
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)


def replace_quantized_linear_with_pace_qlinear2d(graph, node, new_op):
    """
    This method checks for quantized::linear nodes in the graph
    and converts them to pace::qlinear2d
    Args:
        graph: scripted model graph
        node: node which need to be replaced
        new_op : name of the new operator to which it needs to replace with
    """
    graph.setInsertPoint(node)
    packed_w_b = torch.ops.quantized.linear_unpack(
        node.inputsAt(1).node().output().toIValue()
    )
    w_node = graph.insertConstant(packed_w_b[0])

    if packed_w_b[1] is not None:
        b_node = graph.insertConstant(packed_w_b[1].detach())
    else:
        b_node = graph.insertConstant(packed_w_b[1])

    # Create new node, set op and inputs
    new_node = graph.create(new_op)
    new_node.addInput(node.inputsAt(0))
    new_node.addInput(w_node)
    new_node.addInput(b_node)
    new_node.addInput(node.inputsAt(2))
    new_node.addInput(node.inputsAt(3))

    c = graph.insertConstant(13)
    new_node.addInput(c)
    output_ = new_node.output()

    # Find the next node and set it's input as new node
    node.output().replaceAllUsesWith(output_)

    new_node.insertBefore(node)

    # Remove the old node
    node.destroy()


def merge_mul_add(graph):
    """
    This method merges mul and add post ops in the interaction layer
    Args:
        graph: scripted model graph
    """
    mul_nodes = graph.findAllNodes("quantized::mul")
    add_nodes = graph.findAllNodes("quantized::add")
    reshape_nodes = graph.findAllNodes("aten::reshape")  # new
    reshape_node = reshape_nodes[0]  # new
    for i in range(2):
        mul = mul_nodes[i]
        add = add_nodes[i]
        graph.setInsertPoint(mul)
        new_node = graph.create("pace::qmul_add")
        new_node.addInput(mul.inputsAt(0))
        new_node.addInput(mul.inputsAt(1))
        new_node.addInput(add.inputsAt(1))
        new_node.addInput(add.inputsAt(2))
        new_node.addInput(add.inputsAt(3))
        c = graph.insertConstant(13)
        new_node.addInput(c)

        new_node.replaceInput(0, reshape_node.output())

        add.output().replaceAllUsesWith(new_node.output())
        graph.insertNode(new_node)
        add.destroy()
        mul.destroy()


def update_mul_node(graph):
    """
    This method updates inputs to mul and add nodes after quantize_per_tensor optimizations
    Args:
        graph: scripted model graph
    """
    mul_nodes = graph.findAllNodes("quantized::mul")
    quantize_per_tensor_nodes = graph.findAllNodes("aten::quantize_per_tensor")
    add_nodes = graph.findAllNodes("quantized::add")
    for i in range(3):
        mul_nodes[i].replaceInput(0, quantize_per_tensor_nodes[1].output())

    add_nodes[0].replaceInput(1, quantize_per_tensor_nodes[1].output())


def fuse_qlinear2d_mul_add(graph):
    """
    This method checks for the first following pattern in graph
    %1 : pace::qlinear2d(...)
    %2 : quantized::mul(...)
    %3 : quantized::add(...)
    and merges them to single pace node to
    pace::qlinear2d_mul_add()
    Args:
        graph: scripted model graph
    """
    qlinear2dnodes = graph.findAllNodes("pace::qlinear2d")
    nodes_to_replace = [qlinear2dnodes[1]]
    reshape_nodes = graph.findAllNodes("aten::reshape")
    reshape_node = reshape_nodes[0]

    for exact_node in nodes_to_replace:
        graph.setInsertPoint(exact_node)

        new_node = graph.create("pace::qlinear2d_mul_add")

        dtype = exact_node.inputsAt(5)
        mul_node = children_(exact_node)[0]
        add_node = children_(mul_node)[0]

        scale = add_node.inputsAt(2)
        zero_point = add_node.inputsAt(3)

        alpha = graph.insertConstant(1)
        new_node.addInput(exact_node.inputsAt(0))
        new_node.addInput(exact_node.inputsAt(1))
        new_node.addInput(exact_node.inputsAt(2))
        new_node.addInput(reshape_node.output())
        new_node.addInput(reshape_node.output())

        new_node.addInput(alpha)

        output_ = new_node.output()
        next_node = children_(add_node)
        next_node[0].replaceInput(0, output_)
        next_node[1].replaceInput(1, output_)
        new_node.insertAfter(exact_node)

        add_node.destroy()
        mul_node.destroy()
        exact_node.destroy()

        # add quantize_per_tensor node
        graph.setInsertPoint(new_node)
        q_per_tensor = graph.create("aten::quantize_per_tensor")
        q_per_tensor.addInput(new_node.output())
        q_per_tensor.addInput(scale)
        q_per_tensor.addInput(zero_point)
        q_per_tensor.addInput(dtype)
        output_ = q_per_tensor.output()
        next_node[0].replaceInput(0, output_)
        q_per_tensor.insertAfter(new_node)

        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)


def optimize_qlinear(graph):
    """
    This method converts aten quantized linear ops to pace quantized linear ops
    Args:
        graph: scripted model graph
    """
    for orig_op in get_qlinear_mapping():
        new_op = get_zenqlinear_node(orig_op)

        # Find all the nodes with the original op and replace them with new op
        all_nodes = []

        for node in graph.findAllNodes(orig_op):
            all_nodes.append(node)
            replace_quantized_linear_with_pace_qlinear2d(graph, node, new_op)

    # Eliminate any dead code/node and lint the current graph
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command line arguments for model paths."
    )
    parser.add_argument(
        "--input-model-path",
        type=str,
        required=True,
        help="Path to the original model file with file-name",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        required=True,
        help="Path to save optimized model file",
    )
    args = parser.parse_args()
    print("loading model from ", args.input_model_path)
    model = torch.jit.load(args.input_model_path)
    print("model = ", model.graph)
    print("model load done")
    optimize_qlinear(model.graph)

    optimize_embedding_bags_q_per_tensor_nodes(model.graph)
    update_mul_node(model.graph)

    fuse_qlinear2d_mul_add(model.graph)

    merge_mul_add(model.graph)

    print("saving model as ", args.output_model_path)

    model = torch.jit.save(model, args.output_model_path)

    print("model saved ", args.output_model_path)
