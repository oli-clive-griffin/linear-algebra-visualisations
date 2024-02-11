import numpy as np
import plotly.graph_objects as go

def row_picture_3d(X: np.ndarray, b: np.ndarray):
    X = X.copy() + np.random.normal(0, 0.001, X.shape)  # Add some noise to the points to avoid degeneracy
    """returns a plot with planes defined by the rows of X"""

    if np.linalg.det(X) == 0:
        raise ValueError("must be a good matrix")

    fig = go.Figure()
    aug = np.hstack((X, b.reshape(-1, 1)))

    # plot the planes
    for color, i in zip(['red', 'blue', 'green',], range(aug.shape[0])):
        normal = aug[i, :-1]
        d = -aug[i, -1] #TODO validate
        x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
        z = (-normal[0] * x - normal[1] * y - d) * 1.0 / normal[2]
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, opacity=0.5, showscale=False, colorscale=[[0, color], [1, color]]
        ))

    # plot the intersections if the planes intersect
    for i1, i2 in [(0, 1), (1, 2), (0, 2)]:
        n1 = aug[i1, :-1]
        n2 = aug[i2, :-1]
        b1 = aug[i1, -1]
        b2 = aug[i2, -1]

        try:
            point1, point2, direction_vector = find_intersection_line_and_coefficients(
                np.stack((n1, n2)), np.array([b1, b2])
            )
        except:
            continue

        t_values = np.linspace(-10, 10, 100)
        x_values = point1[0] + direction_vector[0] * t_values
        y_values = point1[1] + direction_vector[1] * t_values
        z_values = point1[2] + direction_vector[2] * t_values

        fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', name='Intersection Line', opacity=0.5, line=dict(color='black', width=5)))
    
    solution = np.linalg.solve(X, b)

    traice = go.Scatter3d(x=[solution[0]], y=[solution[1]], z=[solution[2]], mode='markers', name='Solution', marker=dict(size=10, color='red'))
    fig.add_trace(traice)

    # crop the plot to the region of interest
    fig.update_layout(
        scene=dict(
            # scale
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-10, 10]),

            # size of the plot
            aspectmode='cube',
        )
    )
    return fig

def find_intersection_line_and_coefficients(A, b):
    """
    Find two points on the intersecting line of two planes represented by the equations in A and b,
    and return the coefficients for the line's parametric equations.

    Parameters:
    - A: A 2x3 numpy array representing the coefficients of the variables in the equations.
    - b: A 2x1 numpy array representing the constant terms of the equations.

    Returns:
    - Two points on the intersecting line.
    - Coefficients (a, b, c) for the line's parametric equations.
    """
    assert A.shape == (2, 3) and b.shape == (2,), "A must be of shape (2, 3) and b must be of shape (2,)"
    
    points = []
    for z in [0, 1]:
        B_adjusted = b - A[:, 2] * z
        xy = np.linalg.solve(A[:,:2], B_adjusted)
        point = np.append(xy, z)
        points.append(point)
    
    if len(points) != 2:
        raise ValueError("The planes do not intersect.")
    point1, point2 = points
    direction_vector = point2 - point1  # This is (a, b, c)

    return point1, point2, direction_vector

def upper_row_echelon(A):
    """Transforms a matrix into Upper Row Echelon Form."""
    r, c = A.shape
    for i in range(min(r, c)):
        # Find the pivot
        max_row = np.argmax(np.abs(A[i:, i])) + i
        # Swap the current row with the row containing the max pivot
        A[[i, max_row]] = A[[max_row, i]]
        # Eliminate the rows below
        for j in range(i+1, r):
            if A[i, i] == 0:  # Check for zero pivot
                continue
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
    return A

# A = np.array([
#     [1, 4, -2],
#     [3, 2, 7],
#     [8, 3, 2],
# ])

# b = np.array([
#     [10],
#     [20],
#     [30],
# ])


# u = upper_row_echelon(np.hstack((A, b)))
# A_r, b_r = u[:, :-1], u[:, -2:-1]

# # print(A_r)
# # print(b_r)

# # row_picture_3d(A_r, b_r).show()

# stages = [
#     [
#         [1, 4, -2, 10],
#         [3, 2, 7, 20],
#         [8, 3, 2, 30],
#     ],
#     [
#         [1, 4, -2, 10],
#         [0, -10, 13, -10],
#         [8, 3, 2, 30],
#     ],
#     [
#         [1, 4, -2, 10],
#         [0, -10, 13, -10],
#         [0, -29, 18, -50],
#     ],
#     [
#         [1, 4, -2, 10],
#         [0, -10, 13, -10],
#         [0, 0, 18 - (13 * 2.9), -50 + (10 * 2.9)],
#     ],
#     [
#         [1, 4, -2, 10],
#         [0, -10, 0, -10 - (13 / (18 - (13 * 2.9))) * -10],
#         [0, 0, 18 - (13 * 2.9), -50 + (10 * 2.9)],
#     ],

# ]

# stage1 = np.array(stages[0])
# solution_1 = np.linalg.solve(stage1[:, :-1], stage1[:, -1])
# for i in range(1, len(stages)):
#     stage = np.array(stages[i])
#     solution = np.linalg.solve(stage[:, :-1], stage[:, -1])
#     assert np.allclose(solution, solution_1), f"Stage {i} solution does not match stage 1 solution."

# for i in range(1, len(stages)):
#     stage = np.array(stages[i])
#     fig = row_picture_3d(stage[:, :-1], stage[:, -2:-1])
#     fig.show()



# # # plot the planes and their intersections



# row_fig = row_picture_3d(A, b)
# row_fig.show()


# A = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
# ])

# b = np.array([
#     [1],
#     [1],
#     [1],
# ])

# row_picture_3d(A, b).show()
