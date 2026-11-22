import yaml


class GridWorld:
    def __init__(self, size: tuple[int, int] = (10, 10), 
                 start_point: tuple[int, int] = (0, 0),
                 end_point: tuple[int, int] = (9, 9)) -> None:
        width, height = size

        if width <= 0 or height <= 0:
            raise ValueError("Please enter valid values " \
                             "for grid’s length and width.")
        
        self.size = size

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

        # 0 - up, 1 - down, 2 - left, 3 - right
        self.actions = (
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

    def step(self, action: int) -> tuple[tuple[int, int, int, int], 
                                         float, bool, bool]:
        """Executes agent’s turn, calculates new state, reward, 
        whether agent has gone out of bounds, or whether 
        episode has ended."""
        if action not in len(self.actions):
            raise ValueError("Incorrect move.")

        dx, dy = self.actions[action]
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
        return self._observation(), reward, out_of_bounds, done

    def _inside(self, position: tuple[int, int]) -> bool:
        x, y = position
        width, height = self.size
        return 0 <= x < width and 0 <= y < height


    def _observation(self) -> tuple[int, int, int, int]:
        x, y = self.agent
        finish_x, finish_y = self.end_point
        return x, y, finish_x, finish_y