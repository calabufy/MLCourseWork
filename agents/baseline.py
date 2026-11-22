def manhattan_distance(
        start_point: tuple[int, int],
        end_point: tuple[int, int],
) -> int:
    """Calculating Manhattan distance between start and finish."""
    start_x, start_y = start_point
    end_x, end_y = end_point

    return abs(end_x - start_x) + abs(end_y - start_y)