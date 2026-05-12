from globals import *
from functions_general import *
from functions_plotting import *




class SIPPSNode:
    def __init__(self, n: Node, si: Tuple[int, int, str], given_id: int, is_goal: bool, parent: Self | None = None):
        self.x: int = n.x
        self.y: int = n.y
        self.n = n
        self.neighbours = n.neighbours
        self.xy_name: str = self.n.xy_name
        self.si: List[int] = [si[0], si[1]]
        self.si_type = si[2]
        self.given_id: int = given_id
        self.is_goal: bool = is_goal
        self.parent: Self = parent

        self.g: int = 0
        self.h: int = 0
        self.f: int = 0
        self.c: int = 0


    @property
    def low(self):
        return self.si[0]

    @property
    def high(self):
        return self.si[1]

    @property
    def id(self):
        return self.given_id

    @property
    def ident_str(self):
        return f'{self.xy_name}_{self.given_id}_{self.is_goal}'

    def to_print(self):
        return f'SNode: {self.xy_name}, id={self.given_id}, (l={self.low}, h={self.high}), c={self.c}, g={self.g}, h={self.h}, f={self.f}'

    def __str__(self):
        return self.to_print()

    def __repr__(self):
        return self.to_print()

    def set_low(self, new_v: int):
        self.si[0] = new_v

    def set_high(self, new_v: int):
        self.si[1] = new_v

    def __lt__(self, other: Self):
        if self.c < other.c:
            return True
        if self.c > other.c:
            return False
        if self.f < other.f:
            return True
        if self.f > other.f:
            return False
        if self.h < other.h:
            return True
        if self.h >= other.h:
            return False




def init_si_table(
        nodes: List[Node],
        inf_num: int = int(1e10),
) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    f - free
    s - soft
    """
    si_table: Dict[str, List[Tuple[int, int, str]]] = {}
    for node in nodes:
        si_table[node.xy_name] = [(0, inf_num, 'f')]
    return si_table


def update_si_table_hard(
        new_path: List[Node],
        si_table: Dict[str, List[Tuple[int, int, str]]],
        consider_pc: bool = True
):
    iter_path = new_path[:-1] if consider_pc else new_path[:]
    for i, n in enumerate(iter_path):
        si_list = si_table[n.xy_name]
        new_si_list = []
        for si_from, si_to, si_type in si_list:
            if si_from <= i < si_to:
                if si_from < i:
                    new_si_list.append([si_from, i, si_type])
                if i + 1 < si_to:
                    new_si_list.append([i + 1, si_to, si_type])
                continue
            new_si_list.append([si_from, si_to, si_type])
        si_table[n.xy_name] = [(i[0], i[1], i[2]) for i in new_si_list]
    if consider_pc:
        last_n = new_path[-1]
        si_list = si_table[last_n.xy_name]
        i = len(new_path) - 1
        new_si_list = []
        for si_from, si_to, si_type in si_list:
            if si_from <= i < si_to:
                if si_from < i:
                    new_si_list.append((si_from, i, si_type))
                break
            new_si_list.append((si_from, si_to, si_type))
        si_table[last_n.xy_name] = new_si_list
    return si_table


def update_si_table_soft(
        new_path: List[Node],
        si_table: Dict[str, List[Tuple[int, int, str]]],
        inf_num: int = int(1e10),
        consider_pc: bool = True
) -> Dict[str, List[Tuple[int, int, str]]]:
    iter_path = new_path[:-1] if consider_pc else new_path[:]
    for i, n in enumerate(iter_path):
        si_list = si_table[n.xy_name]
        new_si_list = []
        for si_from, si_to, si_type in si_list:
            if si_from <= i < si_to and si_type == 'f':
                if si_from < i:
                    new_si_list.append([si_from, i, 'f'])
                new_si_list.append([i, i+1, 's'])
                if i+1 < si_to:
                    new_si_list.append([i+1, si_to, 'f'])
                continue
            new_si_list.append([si_from, si_to, si_type])

        polished = False
        while not polished:
            polished = True
            for a, b in itertools.pairwise(new_si_list):
                if a[1] == b[0] and a[2] == b[2]:
                    a[1] = b[1]
                    new_si_list.remove(b)
                    polished = False
                    break
        si_table[n.xy_name] = [(i[0], i[1], i[2]) for i in new_si_list]
    if consider_pc:
        last_n = new_path[-1]
        si_list = si_table[last_n.xy_name]
        i = len(new_path) - 1
        new_si_list = []
        for si_from, si_to, si_type in si_list:
            if si_from <= i < si_to:
                if si_type == 'f':
                    if si_from < i:
                        new_si_list.append((si_from, i, 'f'))
                    new_si_list.append((i, inf_num, 's'))
                elif si_type == 's':
                    new_si_list.append((si_from, inf_num, 's'))
                else:
                    raise RuntimeError('uuuuu')
                break
            new_si_list.append((si_from, si_to, si_type))
        si_table[last_n.xy_name] = new_si_list
    return si_table


def get_T(
        node: Node,
        si_table: Dict[str, List[Tuple[int, int, str]]],
        inf_num: int = int(1e10),
) -> int:
    si_list = si_table[node.xy_name]
    last_si_from, last_si_to, last_si_type = si_list[-1]
    if last_si_to >= inf_num:
        return last_si_from
    if last_si_to < inf_num:
        return inf_num
    raise RuntimeError('iiihaaa')


def get_T_tag(
        node: Node,
        si_table: Dict[str, List[Tuple[int, int, str]]],
        inf_num: int = int(1e10),
) -> int:
    si_list = si_table[node.xy_name]
    last_si_from, last_si_to, last_si_type = si_list[-1]
    if last_si_type == 'f':
        return last_si_from
    if last_si_type == 's':
        return last_si_from
    raise RuntimeError('iiihaaa')


def get_c_p(
        sipps_node: SIPPSNode,
        si_table: Dict[str, List[Tuple[int, int, str]]],
        inf_num: int = int(1e10)
):
    si_list = si_table[sipps_node.xy_name]
    si_from, si_to, si_type = si_list[-1]
    if si_to < inf_num:
        return 1
    if si_type == 's':
        return 1
    return 0


def get_c_v(
        sipps_node: SIPPSNode,
        si_table: Dict[str, List[Tuple[int, int, str]]]
) -> int:
    si_list = si_table[sipps_node.xy_name]
    for si_from, si_to, si_type in si_list:
        if si_from <= sipps_node.high - 1:
            if sipps_node.low <= si_to - 1:
                if si_type == 's':
                    return 1
    return 0


def get_c_e(
        sipps_node: SIPPSNode,
        ec_soft_np: np.ndarray,
) -> int:
    parent = sipps_node.parent
    if sipps_node.low < ec_soft_np.shape[4] and ec_soft_np[sipps_node.x, sipps_node.y, parent.x, parent.y, sipps_node.low] == 1:
        return 1
    return 0


def compute_c_g_h_f_values(
        sipps_node: SIPPSNode,
        goal_node: Node,
        goal_np: np.ndarray,
        T: int,
        T_tag: int,
        ec_soft_np: np.ndarray,
        si_table: Dict[str, List[Tuple[int, int, str]]],
) -> None:
    """
    Each curr_node n also maintains a c-value, which is
    the (underestimated) number of the soft collisions of the partial path from the root curr_node to curr_node n, i.e.,
    c(n) = c(n`) + cv + ce,
    where n` is the parent curr_node of n,
    cv is 1 if the safe interval of n contains soft vertex/target obstacles and 0 otherwise,
    and ce is 1 if ((n`.v, n.v), n.low) ∈ Os and 0 otherwise.
    If n is the root curr_node (i.e., n` does not exist), c(n) = cv.
    """
    c_v = get_c_v(sipps_node, si_table)
    c_v_p = c_v
    if c_v == 0:
        c_p = get_c_p(sipps_node, si_table)
        c_v_p = max(c_v, c_p)
    if sipps_node.parent is None:
        sipps_node.c = c_v_p
    else:
        c_e = get_c_e(sipps_node, ec_soft_np)
        sipps_node.c = sipps_node.parent.c + c_v_p + c_e

    if sipps_node.parent is None:
        sipps_node.g = 0
    else:
        sipps_node.g = sipps_node.low

    if sipps_node.xy_name != goal_node.xy_name:
        d_n = goal_np[sipps_node.x, sipps_node.y]
        if sipps_node.c == 0:
            sipps_node.h = max(d_n, T_tag - sipps_node.g)
        else:
            sipps_node.h = max(d_n, T - sipps_node.g)

    else:
        sipps_node.h = 0

    sipps_node.f = sipps_node.g + sipps_node.h


def extract_path(next_sipps_node: SIPPSNode, agent=None) -> Tuple[List[Node], Deque[SIPPSNode]]:
    sipps_path: Deque[SIPPSNode] = deque([next_sipps_node])
    sipps_path_save: Deque[SIPPSNode] = deque([next_sipps_node])
    parent = next_sipps_node.parent
    while parent is not None:
        sipps_path.appendleft(parent)
        sipps_path_save.appendleft(parent)
        parent = parent.parent

    sipps_path_names: List[str] = [n.to_print() for n in sipps_path]
    path_with_waiting: List[Node] = []
    while len(sipps_path) > 0:
        next_node = sipps_path.popleft()
        path_with_waiting.append(next_node.n)
        if len(sipps_path) == 0:
            break
        while len(path_with_waiting) < sipps_path[0].low:
            path_with_waiting.append(path_with_waiting[-1])
    return path_with_waiting, sipps_path_save


def get_c_future(
        goal_node: Node,
        t: int,
        si_table: Dict[str, List[Tuple[int, int, str]]]
) -> int:
    out_value = 0
    si_list = si_table[goal_node.xy_name]
    for si_from, si_to, si_type in si_list:
        if si_from > t:
            continue
        if si_type == 's':
            out_value += 1
    return out_value


def duplicate_sipps_node(node: SIPPSNode) -> SIPPSNode:
    """
    def __init__(self, n: Node, si: Tuple[int, int], _id: int, is_goal: bool, parent: Self | None = None):
    self.x: int = n.x
    self.y: int = n.y
    self.n = n
    self.xy_name: str = self.n.xy_name
    self.si: Tuple[int, int] = si
    self._id: int = _id
    self.is_goal: bool = is_goal
    self.parent: Self = parent

    self.g: int = 0
    self.h: int = 0
    self.f: int = 0
    self.c: int = 0
    """
    return_node = SIPPSNode(
        node.n,
        (node.si[0], node.si[1], node.si_type),
        node.id,
        node.is_goal,
        node.parent
    )
    return_node.g = node.g
    return_node.h = node.h
    return_node.f = node.f
    return_node.c = node.c

    return return_node


def get_identical_nodes(
        curr_node: SIPPSNode,
        Q: List[SIPPSNode],
        P: List[SIPPSNode],
        ident_dict: DefaultDict[str, List[SIPPSNode]],
) -> List[SIPPSNode]:
    """
    Two nodes n1 and n2 have the same identity, denoted as n1 ∼ n2, iff:
    (1) n1.v = n2.v
    (2) n1.id = n2.id
    (3) n1.is_goal = n2.is_goal
    """
    identical_nodes: List[SIPPSNode] = []
    curr_id = curr_node.id
    curr_is_goal = curr_node.is_goal
    for n in ident_dict[curr_node.ident_str]:
        if n != curr_node:
            identical_nodes.append(n)
    return identical_nodes


def get_I_group(
        node: SIPPSNode,
        nodes_dict: Dict[str, Node],
        si_table: Dict[str, List[Tuple[int, int, str]]],
        agent=None
) -> List[Tuple[Node, int]]:
    I_group: List[Tuple[Node, int]] = []
    for nei_name in node.neighbours:
        nei_si_list = si_table[nei_name]
        if nei_name == node.xy_name:
            for si_id, si in enumerate(nei_si_list):
                if si[0] == node.high:
                    I_group.append((node.n, si_id))
                    break
            continue
        for si_id, si in enumerate(nei_si_list):
            if ranges_intersect(range1=(si[0], si[1] - 1), range2=(node.low + 1, node.high)):
                I_group.append((nodes_dict[nei_name], si_id))
                continue
    return I_group


def get_low_without_hard_ec(
        prev_sipps_node: SIPPSNode,
        from_node: Node,
        to_node: Node,
        init_low: int,
        init_high: int,
        ec_hard_np: np.ndarray,
        agent=None
) -> int | None:
    for i_t in range(init_low, init_high):
        if i_t < prev_sipps_node.low + 1:
            continue
        if i_t > prev_sipps_node.high:
            return None
        if i_t >= ec_hard_np.shape[4]:
            return max(i_t, prev_sipps_node.g)
        if ec_hard_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0:
            return i_t
    return None


def get_low_without_hard_and_soft_ec(
        prev_sipps_node: SIPPSNode,
        from_node: Node,
        to_node: Node,
        new_low: int,
        init_high: int,
        ec_hard_np: np.ndarray,
        ec_soft_np: np.ndarray,
) -> int | None:
    for i_t in range(new_low, init_high):
        if i_t < prev_sipps_node.low + 1:
            continue
        if i_t > prev_sipps_node.high:
            return None
        if i_t >= ec_hard_np.shape[4]:
            return max(i_t, prev_sipps_node.g)
        no_in_h = ec_hard_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0
        no_in_s = ec_soft_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0
        if no_in_h and no_in_s:
            return i_t
    return None





