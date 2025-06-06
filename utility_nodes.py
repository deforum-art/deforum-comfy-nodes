from comfy_execution.graph_utils import GraphBuilder
import comfy.samplers
import torch
from .tools import VariantSupport
from .base_node import NODE_NAME, ListNode, LogicNode, FlowNode, DebugNode, UtilityNode

VALID_SAMPLERS = comfy.samplers.KSampler.SAMPLERS
VALID_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

@VariantSupport()
class AccumulateNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"
    
    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

@VariantSupport()
class AccumulationHeadNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

@VariantSupport()
class AccumulationTailNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

@VariantSupport()
class AccumulationToListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "accumulation_to_list"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

@VariantSupport()
class ListToAccumulationNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = True
    FUNCTION = "list_to_accumulation"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

@VariantSupport()
class AccumulationGetLengthNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "accumlength"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)
        
@VariantSupport()
class AccumulationGetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "get_item"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)
        
@VariantSupport()
class AccumulationSetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "set_item"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

@VariantSupport()
class IntMathOperation(LogicNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "int_math_operation"

    def int_math_operation(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)


from .flow_control import NUM_FLOW_SOCKETS
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
        while_open = graph.node("WhileLoopOpen", condition=remaining, initial_value0=remaining, **{("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)})
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
        # TODO - Requires WAS-ns. Will definitely want to solve before merging
        sub = graph.node("IntMathOperation", operation="subtract", a=[while_open,1], b=1)
        cond = graph.node("IntConditions", a=sub.out(0), b=0, operation=">")
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("WhileLoopClose",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }

@VariantSupport()
class DebugPrint(DebugNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "label": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "debug_print"

    def debugtype(self, value):
        if isinstance(value, list):
            result = "["
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += "]"
        elif isinstance(value, tuple):
            result = "("
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += ")"
        elif isinstance(value, dict):
            result = "{"
            for k, v in value.items():
                result += ("%s: %s," % (self.debugtype(k), self.debugtype(v)))
            result += "}"
        elif isinstance(value, str):
            result = "'%s'" % value
        elif isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            result = str(value)
        elif isinstance(value, torch.Tensor):
            result = "Tensor[%s]" % str(value.shape)
        else:
            result = type(value).__name__
        return result

    def debug_print(self, value, label):
        print("[%s]: %s" % (label, self.debugtype(value)))
        return (value,)

NUM_LIST_SOCKETS = 10
@VariantSupport()
class MakeListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("*",),
            },
            "optional": {
                "value%d" % i: ("*",) for i in range(1, NUM_LIST_SOCKETS)
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "make_list"
    OUTPUT_IS_LIST = (True,)

    def make_list(self, **kwargs):
        result = []
        for i in range(NUM_LIST_SOCKETS):
            if "value%d" % i in kwargs:
                result.append(kwargs["value%d" % i])
        return (result,)
    
    
class SamplerSelector(UtilityNode):
    """
    Select a sampler from the list of valid samplers.
    If use_override is True, the override_string is used instead of the sampler_name.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (VALID_SAMPLERS,),
            },
            "optional": {
                "override_string": ("STRING", {"multiline": False}),
                "use_override": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (list(VALID_SAMPLERS),)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "select"

    def select(self, sampler_name: str, override_string: str = None, use_override: bool = False):
        if use_override:
            return (override_string,)
        return (sampler_name,)
    
class SchedulerSelector(UtilityNode):
    """
    Select a scheduler from the list of valid schedulers.
    If use_override is True, the override_string is used instead of the scheduler_name.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler_name": (VALID_SCHEDULERS,),
            },
            "optional": {
                "override_string": ("STRING", {"multiline": False}),
                "use_override": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (list(VALID_SCHEDULERS),)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "select"

    def select(self, scheduler_name: str, override_string: str = None, use_override: bool = False):
        if use_override:
            return (override_string,)
        return (scheduler_name,)
    
class StringToCombo(UtilityNode):
    """
    Convert a string to a combo.
    The input string is **trimmed** and **lower‑cased**.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": False}),
            },
        }
    
    RETURN_TYPES = ("COMBO","STRING", "*")
    FUNCTION = "string_to_combo"
    
    def string_to_combo(self, string: str):
        return (string.strip().lower(), string, string)


# Configuration for node display names

UTILITY_NODE_CLASS_MAPPINGS = {
    "AccumulateNode": AccumulateNode,
    "AccumulationHeadNode": AccumulationHeadNode,
    "AccumulationTailNode": AccumulationTailNode,
    "AccumulationToListNode": AccumulationToListNode,
    "ListToAccumulationNode": ListToAccumulationNode,
    "AccumulationGetLengthNode": AccumulationGetLengthNode,
    "AccumulationGetItemNode": AccumulationGetItemNode,
    "AccumulationSetItemNode": AccumulationSetItemNode,
    "ForLoopOpen": ForLoopOpen,
    "ForLoopClose": ForLoopClose,
    "IntMathOperation": IntMathOperation,
    "DebugPrint": DebugPrint,
    "MakeListNode": MakeListNode,
    "SamplerSelector": SamplerSelector,
    "SchedulerSelector": SchedulerSelector,
    "StringToCombo": StringToCombo,
}

# Generate display names with configurable prefix
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "AccumulateNode": f"Accumulate | {NODE_NAME}",
    "AccumulationHeadNode": f"Accumulation Head | {NODE_NAME}",
    "AccumulationTailNode": f"Accumulation Tail | {NODE_NAME}",
    "AccumulationToListNode": f"Accumulation to List | {NODE_NAME}",
    "ListToAccumulationNode": f"List to Accumulation | {NODE_NAME}",
    "AccumulationGetLengthNode": f"Accumulation Get Length | {NODE_NAME}",
    "AccumulationGetItemNode": f"Accumulation Get Item | {NODE_NAME}",
    "AccumulationSetItemNode": f"Accumulation Set Item | {NODE_NAME}",
    "ForLoopOpen": f"For Loop Open | {NODE_NAME}",
    "ForLoopClose": f"For Loop Close | {NODE_NAME}",
    "IntMathOperation": f"Int Math Operation | {NODE_NAME}",
    "DebugPrint": f"Debug Print | {NODE_NAME}",
    "MakeListNode": f"Make List | {NODE_NAME}",
    "SamplerSelector": f"Sampler Selector | {NODE_NAME}",
    "SchedulerSelector": f"Scheduler Selector | {NODE_NAME}",
    "StringToCombo": f"String to Combo | {NODE_NAME}",
}
