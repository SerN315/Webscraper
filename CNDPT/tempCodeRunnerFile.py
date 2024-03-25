from heapq import heappush, heappop

def heuristic(a, b):
    return sum(abs(val - b[idx]) for idx, val in enumerate(a) if val != 0)

def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0)
    if zero_index % 3 != 0:  # Can move the blank tile left
        swap_index = zero_index - 1
        new_state = list(state)
        new_state[zero_index], new_state[swap_index] = new_state[swap_index], new_state[zero_index]
        neighbors.append(tuple(new_state))
    if zero_index % 3 != 2:  # Can move the blank tile right
        swap_index = zero_index + 1
        new_state = list(state)
        new_state[zero_index], new_state[swap_index] = new_state[swap_index], new_state[zero_index]
        neighbors.append(tuple(new_state))
    if zero_index > 2:  # Can move the blank tile up
        swap_index = zero_index - 3
        new_state = list(state)
        new_state[zero_index], new_state[swap_index] = new_state[swap_index], new_state[zero_index]
        neighbors.append(tuple(new_state))
    if zero_index < 6:  # Can move the blank tile down
        swap_index = zero_index + 3
        new_state = list(state)
        new_state[zero_index], new_state[swap_index] = new_state[swap_index], new_state[zero_index]
        neighbors.append(tuple(new_state))
    return neighbors

def solve_puzzle(start, goal):
    open_set = []
    heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {start: (None, None)}
    cost_so_far = {start: 0}
    move_counter = {'Right': 0, 'Left': 0, 'Up': 0, 'Down': 0}

    while open_set:
        _, current_cost, current = heappop(open_set)

        if current == goal:
            path = [current]
            moves = []
            while came_from[current][0] is not None:
                current, move = came_from[current]
                path.append(current)
                moves.append(move)
            path.reverse()
            moves.reverse()
            return path, moves, move_counter

        for neighbor in get_neighbors(current):
            new_cost = current_cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = (current, get_move(current, neighbor))
                move = get_move(current, neighbor)
                move_counter[move] += 1

    return None, None, None  # No solution found

def get_move(old_state, new_state):
    old_zero_index = old_state.index(0)
    new_zero_index = new_state.index(0)
    if new_zero_index == old_zero_index - 1:
        return 'Right'
    elif new_zero_index == old_zero_index + 1:
        return 'Left'
    elif new_zero_index == old_zero_index + 3:
        return 'Up'
    elif new_zero_index == old_zero_index - 3:
        return 'Down'

def print_board(state):
    board = ""
    for i in range(0, 9, 3):
        board += "| " + " | ".join(str(x) if x != 0 else ' ' for x in state[i:i+3]) + " |\n"
    return board

# Example usage:
start_state = (8,7,6,2,0,5,1,4,3)  # 0 represents the empty slot
goal_state = (1,3,5,8,0,7,6,4,2)

path, moves, move_counter = solve_puzzle(start_state, goal_state)
if path and moves:
    for step, move in zip(path, moves):
        print(f"Move: {move}")
        print(print_board(step))
        move_counter[move] += 1  # Increment the move counter
    print("Number of moves:")
    for move, count in move_counter.items():
        print(f"{move}: {count}")
else:
    print("No solution found or the goal state is not reachable from the start state.")