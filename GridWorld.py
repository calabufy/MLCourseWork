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
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError("Please enter valid start and " \
                                 "finish values.")

        if start_point == end_point:
            raise ValueError("Start and finish must not be at " \
                             "the same point.")
        
        self.start_point = start_point
        self.end_point = end_point
        self.grid = self._create_grid()


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