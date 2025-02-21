#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LCLGame:
    def __init__(self):
        pass

    def is_valid_construct(self, pieces):
        """
        A construct is valid if:
         1) No two pieces overlap on the same y level.
         2) All pieces form a single connected component in 2D:
            - Two pieces are 'connected' if |y1 - y2| == 1 AND
              their x-ranges intersect in at least one stud.
        """
        if not pieces:
            return False  # Empty construct is invalid

        # -------------------------
        # 1. Check for Overlapping
        # -------------------------
        occupied_positions = {}
        for piece in pieces:
            if not isinstance(piece, tuple) or len(piece) != 3:
                print(f"Invalid piece format: {piece}")
                return False

            x, y, color = piece
            if not isinstance(x, int) or not isinstance(y, int) or not isinstance(color, str):
                print(f"Invalid piece data types: {piece}")
                return False

            # Each piece occupies x..x+3 (4 studs total)
            piece_positions = set(range(x, x + 4))

            # Check if there's already something at this y level
            if y in occupied_positions:
                if piece_positions & occupied_positions[y]:
                    return False  # Overlap detected
                occupied_positions[y].update(piece_positions)
            else:
                occupied_positions[y] = piece_positions

        # -------------------------
        # 2. Check for Connectivity
        # -------------------------
        def are_connected(p1, p2):
            x1, y1, _ = p1
            x2, y2, _ = p2
            # Must differ by exactly 1 in y
            if abs(y1 - y2) != 1:
                return False
            # Must share at least one x-stud
            xrange1 = set(range(x1, x1 + 4))
            xrange2 = set(range(x2, x2 + 4))
            return not xrange1.isdisjoint(xrange2)

        # Build adjacency
        n = len(pieces)
        adj_list = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if are_connected(pieces[i], pieces[j]):
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # BFS or DFS to check single connected component
        visited = set()

        def bfs(start):
            from collections import deque
            queue = deque([start])
            visited.add(start)
            while queue:
                current = queue.popleft()
                for neighbor in adj_list[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Start BFS from piece 0 (assuming at least one piece)
        bfs(0)

        # If we haven't visited all, it's disconnected
        return len(visited) == n

class LCLVisualizer:
    """
    Visualizes a 2D 'stud-level' view of Lego pieces.
    Each piece is drawn as a rectangle of width 4 and height 1
    at the (x, y) coordinates with the given facecolor.
    """
    def __init__(self, output_path='lcl_construct_axes.png'):
        self.output_path = output_path

    def draw_piece(self, ax, position, color='blue'):
        # position is (x, y). We'll draw a rectangle 4 studs wide, 1 stud high
        ax.add_patch(
            patches.Rectangle(
                position,
                width=4,
                height=1,
                edgecolor='black',
                facecolor=color
            )
        )

    def display_construct(self, pieces):
        fig, ax = plt.subplots()

        # Draw each piece
        for piece in pieces:
            x, y, color = piece
            self.draw_piece(ax, (x, y), color)

        # Determine min/max for x and y to set axis bounds
        if pieces:
            xs = [p[0] for p in pieces]
            ys = [p[1] for p in pieces]
            min_x, max_x = min(xs), max(xs) + 4  # account for piece width
            min_y, max_y = min(ys), max(ys) + 1  # account for piece height
        else:
            # Fallback if somehow no pieces
            min_x, max_x = 0, 4
            min_y, max_y = 0, 1

        ax.set_xlim(left=min_x - 1, right=max_x + 1)
        ax.set_ylim(bottom=min_y - 1, top=max_y + 1)

        # Grid + aspect
        ax.set_xticks(range(min_x - 1, max_x + 2))
        ax.set_yticks(range(min_y - 1, max_y + 2))
        ax.set_aspect('equal', 'box')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Studs (X)')
        ax.set_ylabel('Layers (Y)')
        ax.set_title(f"Construct: {pieces}")

        # Save and close figure
        plt.savefig(self.output_path, bbox_inches='tight')
        plt.close(fig)

def demo_scenarios():
    """
    Demonstrate the original four scenarios plus five extra
    tests focusing on how many studs (0 to 4) two stacked bricks can share.
    """
    game = LCLGame()

    # -------------------------
    # Original Four Scenarios
    # -------------------------
    scenarios = [
        # (scenario_pieces, name_of_scenario)
        ([(0, 0, 'red'), (2, 0, 'blue')], "Scenario 1: Overlapping + Not Connected"),
        ([(0, 0, 'red'), (2, 0, 'blue'), (2, 1, 'green')], "Scenario 2: Overlapping + Connected"),
        ([(0, 0, 'red'), (0, 1, 'blue')], "Scenario 3: Not Overlapping + Connected"),
        ([(0, 0, 'red'), (5, 0, 'blue')], "Scenario 4: Not Overlapping + Not Connected")
    ]

    # We'll assign each scenario an output file scenario1.png, scenario2.png, ...
    for i, (pieces, name) in enumerate(scenarios, start=1):
        valid = game.is_valid_construct(pieces)
        print(f"{name}")
        print(f"Pieces: {pieces}")
        print("is_valid_construct:", valid)
        out_file = f"scenario{i}.png"
        visualizer = LCLVisualizer(out_file)
        visualizer.display_construct(pieces)
        print(f"Saved visualization to {out_file}")
        print("-" * 60)

    # -------------------------
    # Additional Connectivity Tests
    #   Show different # of overlapping studs (0..4)
    # -------------------------
    connectivity_tests = [
        ([(0, 0, 'red'), (4, 1, 'blue')],
         "Scenario 5: 0-stud overlap (y=1) => Not Connected => Invalid"),
        ([(0, 0, 'red'), (3, 1, 'blue')],
         "Scenario 6: 1-stud overlap => Connected => Valid"),
        ([(0, 0, 'red'), (2, 1, 'blue')],
         "Scenario 7: 2-stud overlap => Connected => Valid"),
        ([(0, 0, 'red'), (1, 1, 'blue')],
         "Scenario 8: 3-stud overlap => Connected => Valid"),
        ([(0, 0, 'red'), (0, 1, 'blue')],
         "Scenario 9: 4-stud overlap => Connected => Valid")
    ]

    # Visualize these as scenario5.png, scenario6.png, etc.
    for i, (pieces, name) in enumerate(connectivity_tests, start=5):
        valid = game.is_valid_construct(pieces)
        print(name)
        print(f"Pieces: {pieces}")
        print("is_valid_construct:", valid)
        out_file = f"scenario{i}.png"
        visualizer = LCLVisualizer(out_file)
        visualizer.display_construct(pieces)
        print(f"Saved visualization to {out_file}")
        print("-" * 60)

if __name__ == "__main__":
    demo_scenarios()
