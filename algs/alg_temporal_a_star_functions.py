
from globals import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class AStarNode:
    def __init__(self, n: Node, t: int, h: int, parent: Self | None = None):
        self.n: Node = n
        self.x: int = n.x
        self.y: int = n.y
        self.t: int = t
        self.h: int = h
        self.f: int = t + h
        self.parent: Self | None = parent
        self.neighbours: List[str] = n.neighbours[:]
        self.neighbours_nodes: List[Node] = n.neighbours_nodes[:]
        self.xy_name: str = f'{self.x}_{self.y}'
        self.xyt_name: str = f'{self.x}_{self.y}_{self.t}'
        #----------------- new ---------------
        self._state_id: int | None = None
        self._max_time: int = 10000

    def __eq__(self, other: Self) -> bool:
        return self.xyt_name == other.xyt_name

    def __lt__(self, other: Self) -> bool:
        # for sort
        if self.f < other.f:
            return True
        if self.f > other.f:
            return False
        if self.h < other.h:
            return True
        # critic to put >=
        if self.h >= other.h:
            return False
        raise RuntimeError('uuuups')

    # def __hash__(self):
    #     return hash(self.xyt_name)

    def __str__(self) -> str:
        return f'{self.xyt_name} [{self.f}]'

    def __repr__(self) -> str:
        return f'{self.xyt_name} [{self.f}]'
    #---------------- new --------------------
    def state_id(self) -> int:
        """Fast state ID using bit-packing"""
        if self._state_id is None:
            self._state_id = (self.n.id << 16) | self.t
        return self._state_id

    @property
    def state_id(self) -> int:
        """Fast integer ID for this state (x, y, t combination)"""
        if self._state_id is None:
            self._state_id = self.n.id * 100000 + self.t | self.t  # Bit-pack node_id and time
        return self._state_id

    def fast_equals(self, other: 'AStarNode') -> bool:
        """Fast equality check using integer IDs"""
        return self.state_id == other.state_id
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def reconstruct_path(i_node: AStarNode) -> List[Node]:
    # we have reached the end
    path: List[Node] = []
    node_current = i_node
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()
    return path


def get_latest_vc_on_node(i_node: AStarNode, vc_np: np.ndarray | None) -> int:
    """
    :param i_node:
    :param vc_np: vertex constraints [x, y, t] = bool
    :return: int
    """
    if vc_np is None:
        return 0
    vc_times = vc_np[i_node.x, i_node.y, :]
    indices = np.argwhere(vc_times == 1)
    if len(indices) == 0:
        return 0
    return int(indices[-1][0])






