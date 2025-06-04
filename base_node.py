
NODE_PREFIX = "Deforum"  # Can be changed to any desired prefix

# Base class for all nodes
class BaseNode:
    CATEGORY = NODE_PREFIX
    
    @classmethod
    def get_category(cls):
        return cls.CATEGORY
    
class LogicNode(BaseNode):
    CATEGORY = f"{NODE_PREFIX}/Logic"
    
class FlowNode(BaseNode):
    CATEGORY = f"{NODE_PREFIX}/Flow"
    
class UtilityNode(BaseNode):
    CATEGORY = f"{NODE_PREFIX}/Utility"
    
class DebugNode(BaseNode):
    CATEGORY = f"{NODE_PREFIX}/Debug"
    
class ListNode(BaseNode):
    CATEGORY = f"{NODE_PREFIX}/Lists"
    