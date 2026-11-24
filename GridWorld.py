import yaml


class GridWorld:
    def __init__(self, size: tuple[int, int] = (10, 10)) -> None:
        width, height = size

        if width <= 0 or height <= 0:
            raise ValueError("Please enter valid values " \
                             "for grid’s length and width.")

        self.start_point: tuple[int, int]
        self.end_point: tuple[int, int]
        self.size = size
        self.grid: list[list[str]]

        # 0 - up, 1 - down, 2 - left, 3 - right
        self.ACTIONS = (
            (0, -1), (0, 1), (-1, 0), (1, 0)
        )

        self.agent: tuple[int, int] = (0, 0)

        with open("agents/reward.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.rewards = config["rewards"]

    def _create_grid(self) -> list[list[str]]:
        """Creating an environment based on specified parameters."""
        grid = []
        for _ in range(self.size[0]):
            row = []
            for _ in range(self.size[1]):
                row.append('.')
            grid.append(row)

        start_x, start_y = self.start_point
        end_x, end_y = self.end_point
        grid[start_y][start_x] = "A"
        grid[end_y][end_x] = "B"
        return grid


    def print_grid(self) -> None:
        """Display the grid in the console."""
        for row in self.grid:
            print(row, end="\n")

    def init_env(self, start_point: tuple[int, int] = (0, 0), 
                 end_point: tuple[int, int] = (9, 9)) -> tuple[float, float, float, float]:
        """Initialising the environment for tаhe agent."""
        width, height = self.size
        for x, y in (start_point, end_point):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("Please enter valid start and " \
                "finish values.")
        
            if start_point == end_point:
                raise ValueError("Start and finish must not be at " \
                "the same point.")
                
        self.start_point = start_point
        self.end_point = end_point
        self.grid = self._create_grid()
        self.agent = start_point

        return self._observation()
        

    def step(self, action_idx: int) -> tuple[tuple[float, float, float, float], 
                                             float, bool, bool]:
        """Executes agent’s turn, calculates new state, reward, 
        whether agent has gone out of bounds, or whether 
        episode has ended."""
        if action_idx not in range(len(self.ACTIONS)):
            raise ValueError("Incorrect move.")

        dx, dy = self.ACTIONS[action_idx]
        x, y = self.agent
        next_position = (x + dx, y + dy)
        out_of_bounds = False

        if self._inside(next_position):
            self.agent = next_position
            reward = self.rewards["step"]
        else:
            out_of_bounds = True
            reward = self.rewards["out_of_bounds"]

        done = self.agent == self.end_point
        if done:
            reward = self.rewards["finish"]
        return self._observation(), reward, out_of_bounds, done

    def _inside(self, position: tuple[int, int]) -> bool:
        """Checking whether agent is inside the grid."""
        x, y = position
        width, height = self.size
        return 0 <= x < width and 0 <= y < height


    def _observation(self) -> tuple[int, int, int, int]:
        """Calculation current state of environment."""
        x, y = self.agent
        finish_x, finish_y = self.end_point
        width, height = self.size

        return (
            x / max(width - 1, 1),
            y / max(height - 1, 1),
            finish_x / max(width - 1, 1),
            finish_y / max(height - 1, 1)
            )