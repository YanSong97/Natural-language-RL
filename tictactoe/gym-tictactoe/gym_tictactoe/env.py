import copy
import logging
from typing import List, Optional

import gym
from gym import spaces

CODE_MARK_MAP = {0: " ", 1: "O", 2: "X"}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0

LEFT_PAD = "  "
LOG_FMT = logging.Formatter(
    "%(levelname)s " "[%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S"
)


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == "O" else 2


def next_mark(mark):
    return "X" if mark == "O" else "O"


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def parse_current_mark(state):
    hashsum = sum([1 for x in state if x == 0])
    if hashsum % 2 == 1:
        return "O"
    else:
        return "X"


def after_action_state(state, action):
    """Execute an action and returns resulted state.

    Args:
        state (tuple): Board status + mark
        action (int): Action to run

    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


def check_game_status(board):
    """Return game status by current board status.

    Args:
        board (list): Current board state

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j + 3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j + 3] == t and board[j + 6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    @staticmethod
    def parse_str2state(board_str: str):
        def _parse_cell(x: str):
            x = x.strip()
            if x.isdigit() and 1 <= int(x) <= 9:
                return 0
            elif x == "O":
                return 1
            elif x == "X":
                return 2
            else:
                raise ValueError("Invalid cell: '{}'".format(x))

        state = []
        lines = board_str.split("\n")
        for i, lines in enumerate(lines):
            if i % 2 == 1:
                continue
            cells = lines.split("|")
            for cell in cells:
                state.append(_parse_cell(cell))
        return state

    @staticmethod
    def state2str(state: List[int], show_number=True):

        ret_str = ""
        for j in range(0, 9, 3):

            def mark(i):
                return (
                    tomark(state[i]) if not show_number or state[i] != 0 else str(i + 1)
                )

            ret_str += LEFT_PAD + "|".join([mark(i) for i in range(j, j + 3)]) + "\n"

            if j < 6:
                ret_str += LEFT_PAD + "-----" + "\n"
        return ret_str

    def __init__(
        self,
        *,
        # 0 for empty, 1 for 'O', 2 for 'X'
        from_position: Optional[List[int]] = None,
        alpha=0.02,
        show_number=True
    ):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.alpha = alpha
        self.set_start_mark("O")
        self.show_number = show_number

        self._from_position = from_position
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def reset(self):
        if self._from_position is None:
            self.board = [0] * NUM_LOC
        else:
            self.board = copy.deepcopy(self._from_position)
        self.mark = parse_current_mark(self.board)
        self.done = False
        return self._get_obs()

    def step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        self.board[loc] = tocode(self.mark)
        status = check_game_status(self.board)
        logging.debug(
            "check_game_status board {} mark '{}'"
            " status {}".format(self.board, self.mark, status)
        )
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == "O" else X_REWARD

        # switch turn
        self.mark = next_mark(self.mark)
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        check_mark = parse_current_mark(self.board)
        assert self.mark == check_mark
        return tuple(self.board), self.mark

    def render(self, mode="human", close=False):
        if close:
            return
        if mode == "human":
            self._show_board(print)  # NOQA
            print("")
        else:
            self._show_board(logging.info)
            logging.info("")

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn: lambda: None):
        """Draw tictactoe board."""
        ret_str = ""
        for j in range(0, 9, 3):

            def mark(i):
                return (
                    tomark(self.board[i])
                    if not self.show_number or self.board[i] != 0
                    else str(i + 1)
                )

            ret_str += LEFT_PAD + "|".join([mark(i) for i in range(j, j + 3)]) + "\n"

            if j < 6:
                ret_str += LEFT_PAD + "-----" + "\n"
        showfn(ret_str)
        return ret_str

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn("")

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]


def set_log_level_by(verbosity):
    """Set log level by verbosity level.

    verbosity vs log level:

        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG

    Args:
        verbosity (int): Verbosity level given by CLI option.

    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level
