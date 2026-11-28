"""Interactive inference for a trained GridWorld agent."""

from pathlib import Path

import numpy as np
import pygame

from GridWorld import GridWorld
from agents.DQN import DQNAgent


class GridWorldUI:
    def __init__(
        self,
        env: GridWorld,
        agent: DQNAgent,
        step_interval: float = 0.25,
        max_steps: int = 100,
    ) -> None:
        if step_interval <= 0 or max_steps <= 0:
            raise ValueError("step_interval and max_steps must be positive")
        self.env = env
        self.agent = agent
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.start = None
        self.goal = None
        self.state = None
        self.path = []
        self.steps = 0
        self.running_route = False
        self.message = "Выберите стартовую клетку"
        self.next_step = 0

    def reset(self) -> None:
        self.start = None
        self.goal = None
        self.state = None
        self.path = []
        self.steps = 0
        self.running_route = False
        self.next_step = 0
        self.message = "Выберите стартовую клетку"

    def select(self, point: tuple[int, int]) -> None:
        if self.running_route:
            return
        if self.start is None:
            self.start = point
            self.message = "Выберите клетку назначения"
            return
        if point == self.start:
            self.message = "Выберите другую клетку"
            return
        self.goal = point
        self.state = np.asarray(
            self.env.init_env(start_point=self.start, end_point=point),
            dtype=np.float32,
        )
        self.path = [self.start]
        self.steps = 0
        self.running_route = True
        self.next_step = pygame.time.get_ticks() + int(self.step_interval * 1000)
        self.message = "Агент движется к цели"

    def advance(self) -> None:
        action = self.agent.choose_action(self.state, training=False)
        observation, _, out_of_bounds, done = self.env.step(action)
        self.state = np.asarray(observation, dtype=np.float32)
        self.steps += 1
        self.path.append(self.env.agent)
        if done:
            self.start = self.env.agent
            self.running_route = False
            self.message = "Цель достигнута. Выберите новую цель"
        elif out_of_bounds or self.steps >= self.max_steps:
            self.start = self.env.agent
            self.running_route = False
            self.message = "Маршрут не завершён. Выберите новую цель"

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((1000, 760), pygame.RESIZABLE)
        pygame.display.set_caption("GridWorld - DQN")
        font = pygame.font.SysFont(None, 30)
        small_font = pygame.font.SysFont(None, 23)
        clock = pygame.time.Clock()
        running = True
        try:
            while running:
                window_width, window_height = screen.get_size()
                width, height = self.env.size
                cell_size = min((window_width - 80) / width, (window_height - 170) / height)
                cell_size = max(cell_size, 1)
                grid_width, grid_height = cell_size * width, cell_size * height
                left = (window_width - grid_width) / 2
                top = 100 + (window_height - 150 - grid_height) / 2

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.reset()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        x, y = event.pos
                        if left <= x < left + grid_width and top <= y < top + grid_height:
                            self.select((int((x - left) / cell_size), int((y - top) / cell_size)))

                now = pygame.time.get_ticks()
                if self.running_route and now >= self.next_step:
                    self.advance()
                    self.next_step = now + int(self.step_interval * 1000)

                screen.fill((17, 24, 39))
                screen.blit(font.render("GRIDWORLD  /  DQN", True, (234, 242, 255)), (32, 22))
                screen.blit(small_font.render(self.message, True, (173, 191, 215)), (32, 58))
                for y in range(height):
                    for x in range(width):
                        rect = pygame.Rect(round(left + x * cell_size), round(top + y * cell_size),
                                           round(cell_size), round(cell_size))
                        pygame.draw.rect(screen, (31, 43, 61), rect)
                        pygame.draw.rect(screen, (55, 72, 95), rect, 1)
                for x, y in self.path:
                    center = (round(left + (x + 0.5) * cell_size), round(top + (y + 0.5) * cell_size))
                    pygame.draw.circle(screen, (70, 101, 133), center, max(2, round(cell_size * 0.08)))
                if self.goal is not None:
                    x, y = self.goal
                    center = (round(left + (x + 0.5) * cell_size), round(top + (y + 0.5) * cell_size))
                    pygame.draw.circle(screen, (248, 178, 90), center, max(3, round(cell_size * 0.26)), 2)
                if self.start is not None:
                    x, y = self.env.agent if self.state is not None else self.start
                    center = (round(left + (x + 0.5) * cell_size), round(top + (y + 0.5) * cell_size))
                    pygame.draw.circle(screen, (76, 211, 179), center, max(3, round(cell_size * 0.29)))
                footer = f"Шаги: {self.steps} / {self.max_steps}    •    R - сброс    •    Esc - выход"
                screen.blit(small_font.render(footer, True, (173, 191, 215)), (32, window_height - 35))
                pygame.display.flip()
                clock.tick(60)
        finally:
            pygame.quit()


def main(
    weights: str | Path,
    layers: int = 2,
    neurons: int = 64,
    size: tuple[int, int] = (10, 10),
    interval: float = 0.25,
    max_steps: int = 100,
) -> None:
    agent = DQNAgent(NN_layers=layers, NN_neurons=neurons)
    agent.build_model()
    agent.load(str(weights))
    env = GridWorld(size=size)
    GridWorldUI(env, agent, interval, max_steps).run()


if __name__ == "__main__":
    main(weights="agents/history/DQN_episode7500/20261124_223029_546903/model.weights.h5")
