from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker
from .tools import VariantSupport
from .base_node import NODE_NAME, NODE_POSTFIX, FlowNode, BaseNode

NUM_FLOW_SOCKETS = 5

# Internal helper node for for loop counter management
@VariantSupport()
class _ForLoopCounter(BaseNode):
    """Internal node for for loop counter management - not exposed to users"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_value": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("INT", "BOOLEAN")
    RETURN_NAMES = ("decremented", "should_continue")
    FUNCTION = "process_counter"
    CATEGORY = f""
    
    def process_counter(self, current_value):
        decremented = current_value - 1
        should_continue = decremented > 0
        return (decremented, should_continue)


@VariantSupport()
class WhileLoopOpen(FlowNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["FLOW_CONTROL"] + ["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_open"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)


@VariantSupport()
class WhileLoopClose(FlowNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_close"
    
    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        for i in range(NUM_FLOW_SOCKETS):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse" )
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


@VariantSupport()
class ExecutionBlockerNode(FlowNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "input": ("*",),
                "block": ("BOOLEAN",),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }
        return inputs

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execution_blocker"
    
    def execution_blocker(self, input, block, verbose):
        if block:
            return (ExecutionBlocker("Blocked Execution" if verbose else None),)
        return (input,)
    
    
@VariantSupport()
class ForLoopOpen(FlowNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: ("*",) for i in range(1, NUM_FLOW_SOCKETS)
            },
            "hidden": {
                "initial_value0": ("*",)
            }
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT",] + ["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["flow_control", "remaining"] + ["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_open"

    def for_loop_open(self, remaining, **kwargs):
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        while_open = graph.node(f"WhileLoopOpen{NODE_POSTFIX}", condition=remaining, initial_value0=remaining, **{("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }


@VariantSupport()
class ForLoopClose(FlowNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: ("*",{"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_close"

    def for_loop_close(self, flow_control, **kwargs):
        graph = GraphBuilder()
        while_open = flow_control[0]
        
        # Use our internal counter node instead of external dependencies
        counter = graph.node(f"_ForLoopCounter{NODE_POSTFIX}", current_value=[while_open, 1])
        
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node(f"WhileLoopClose{NODE_POSTFIX}",
                flow_control=flow_control,
                condition=counter.out(1),  # should_continue output
                initial_value0=counter.out(0),  # decremented output
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }


# Configuration for node display names
FLOW_CONTROL_NODE_CLASS_MAPPINGS = {
    f"WhileLoopOpen{NODE_POSTFIX}": WhileLoopOpen,
    f"WhileLoopClose{NODE_POSTFIX}": WhileLoopClose,
    f"ExecutionBlocker{NODE_POSTFIX}": ExecutionBlockerNode,
    f"ForLoopOpen{NODE_POSTFIX}": ForLoopOpen,
    f"ForLoopClose{NODE_POSTFIX}": ForLoopClose,
    f"_ForLoopCounter{NODE_POSTFIX}": _ForLoopCounter,  # Internal node, prefixed with underscore
}

# Generate display names with configurable prefix
# Note: _ForLoopCounter is not included in display names as it's internal
FLOW_CONTROL_NODE_DISPLAY_NAME_MAPPINGS = {
    f"WhileLoopOpen{NODE_POSTFIX}": f"While Loop Open {NODE_POSTFIX}",
    f"WhileLoopClose{NODE_POSTFIX}": f"While Loop Close {NODE_POSTFIX}",
    f"ExecutionBlocker{NODE_POSTFIX}": f"Execution Blocker {NODE_POSTFIX}",
    f"ForLoopOpen{NODE_POSTFIX}": f"For Loop Open {NODE_POSTFIX}",
    f"ForLoopClose{NODE_POSTFIX}": f"For Loop Close {NODE_POSTFIX}",
    # Intentionally not including _ForLoopCounter in display mappings
}