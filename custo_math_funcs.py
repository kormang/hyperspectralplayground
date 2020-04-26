import numpy as np
import math
from math_utils import clamp

#### 2nd degree bsplines

def solve_linear_equation(a, b):
    return -b/a

def solve_quadratic_equation(a, b, c):
    if a == 0:
        return solve_linear_equation(b, c)

    detr = b*b - 4*a*c
    
    assert (detr >= 0), 'Failed to calculate bspline coordinate - determinant < 0!'
    
    result = (-b + math.sqrt(detr)) / (2*a)    

    if result < 0 or result > 1:
        result = (-b - math.sqrt(detr)) / (2*a)
    
    assert (result >= 0 and result <= 1), 'Failed to calculate bspline coordinate - all solutions out of [0, 1]'
    return result

def bspline2t_x(x, p0, p1, p2):
    a = p0[0] - 2*p1[0] + p2[0]
    b = -2*p0[0] + 2*p1[0]
    c = p0[0] + p1[0] - 2*x
    
    return solve_quadratic_equation(a, b, c)

def bspline2y_t(t, p0, p1, p2):
    q0 = (1 - t)*(1 - t)
    q1 = -2*t*t + 2*t + 1
    q2 = t*t
    y = (q0*p0[1] + q1*p1[1] + q2*p2[1])/2
    return y

def bspline2y_x(x, p0, p1, p2):
    t = bspline2t_x(x, p0, p1, p2)
    return bspline2y_t(t, p0, p1, p2)

def bspline2x_t(t, p0, p1, p2):
    q0 = (1 - t)*(1 - t)
    q1 = -2*t*t + 2*t + 1
    q2 = t*t
    x = (q0*p0[0] + q1*p1[0] + q2*p2[0])/2
    return x

def bspline2_get_points_on_segment(s, points):
    plen = len(points)
    p0i = clamp(s - 1, 0, plen - 1)
    p1i = clamp(s, 0, plen - 1)
    p2i = clamp(s + 1, 0, plen - 1)
    return points[p0i], points[p1i], points[p2i]

def bspline2_from_points(points):
    """
        Internally highly dependens on result of pycontinuum_points,
        e.g. there should be results[0][0] == 0.
        Also it gives x coordinates always as integers,
        and assumes there is not two stitching points between integers values.
    """
    segments = len(points)
    
    # Calculate points where segments stitch together.
    # Also includes first point and last point.
    x_stitching_points = [0]
    for s in range(segments):
        p0, p1, p2 = bspline2_get_points_on_segment(s, points)
        x_stitching_points.append(bspline2x_t(1.0, p0, p1, p2))
    
    # Fill results with pairs (i, y(i)),
    # where i is x coordinate and also index in signature array,
    # and y(i) is respective  y coordinate of bspline2.
    results = np.empty(points[-1][0] + 1)
    for s in range(segments):
        p0, p1, p2 = bspline2_get_points_on_segment(s, points)
        # We need special case for the first element,
        # as we need integer x in results, but first element could
        # be rounded down to x point that is outside this segment.
        # That leads to errors in equations, where we get complex solutions.
        x = x_stitching_points[s]
        #results.append(bspline2y_x(x, p0, p1, p2))
        results[int(x)] = bspline2y_x(x, p0, p1, p2)
        for x in range(int(x_stitching_points[s]) + 1, int(x_stitching_points[s + 1])):
            results[x] = bspline2y_x(x, p0, p1, p2)
            #results.append(bspline2y_x(x, p0, p1, p2))
    
    # Last stitching (accually not stitching, since it is the last one of stitching points)
    # of the last segment is not included because of non-inclusive ranges.
    # We still need it.
    p0, p1, p2 = bspline2_get_points_on_segment(segments - 1, points)
    x = x_stitching_points[-1]
    results[int(x)] = bspline2y_x(x, p0, p1, p2)
    #results.append(bspline2y_x(x, p0, p1, p2))

    return results
    #return np.array(results)


#### 3nd degree bsplines

def solve_cardano_cubic(p, q):
    n = q/2
    m = p/3
    tmp0 = math.sqrt(n**2 + m**3)
    tmp1 = -n + tmp0
    tmp1r = tmp1**(1/3) if tmp1 >= 0 else -(-tmp1)**(1/3)
    tmp2 = -n - tmp0
    tmp2r = tmp2**(1/3) if tmp2 >= 0 else -(-tmp2)**(1/3)
    return tmp1r + tmp2r


def solve_trig01_cubic(p, q, conv_term):
    g = 2*math.sqrt(-p/3)
    h = math.acos(3*q*math.sqrt(-3/p)/(2*p))/3
    j = 2*math.pi/3
    
    for k in range(3):
        r = g*math.cos(h - j*k)
        r = round(r - conv_term, 8)
        if r >= 0.0 and r <= 1.0:
            return r
    
    assert (False), 'Trigonometric fomula gave us no valid solution'

def solve_cubic_equation(a, b, c, d):
    if a == 0:
        return solve_quadratic_equation(b, c, d)

    # Solving ax^3 + bx^2 + cx + d = 0.
    # Transform to depressed cubic: t^3 + pt + q = 0.
    p = (3*a*c - b*b) / (3*a*a)
    q = (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27*a*a*a)
    
    detr = -(4*p*p*p + 27*q*q)
    
    conv_term = b / (3*a)
    
    if detr < 0:
        # There is only one real root. Use Cardano's method.
        t = solve_cardano_cubic(p, q)
        result = round(t - conv_term, 8)
        assert (result >= 0.0 and result <= 1.0), 'Cardano gave us invalid value for spline parameter. ' + str(result)
        return result
    elif detr > 0:
        # All three roots are real, and not rational. Use trigonometric formula.
        return solve_trig01_cubic(p, q, conv_term)
    elif p == 0.0:
        return 0.0
    else:
        t = 3*q/p
        result = round(t - conv_term, 8)
        if result >= 0.0 and result <= 1.0:
            return result
        result = round(-t/2 - conv_term, 8)
        assert (result >= 0.0 and result <= 1.0), 'Could not find valid value for spline parameter.'
        return result

    
    
    
    

def bspline3t_x(x, p0, p1, p2, p3):
    a = -p0[0] + 3*p1[0] - 3*p2[0] + p3[0]
    b = 3*p0[0] - 6*p1[0] + 3*p2[0]
    c = -3*p0[0] + 3*p2[0]
    d = p0[0] + 4*p1[0] + p2[0] - 6*x
       
    return solve_cubic_equation(a, b, c, d)

def bspline3y_t(t, p0, p1, p2, p3):
    q0 = (1 - t)*(1 - t)*(1 - t)
    q1 = 3*t*t*t - 6*t*t + 4
    q2 = -3*t*t*t + 3*t*t + 3*t + 1
    q3 = t*t*t
    y = (q0*p0[1] + q1*p1[1] + q2*p2[1] + q3*p3[1])/6
    return y

def bspline3y_x(x, p0, p1, p2, p3):
    t = bspline3t_x(x, p0, p1, p2, p3)
    return bspline3y_t(t, p0, p1, p2, p3)

def bspline3x_t(t, p0, p1, p2, p3):
    q0 = (1 - t)*(1 - t)*(1 - t)
    q1 = 3*t*t*t - 6*t*t + 4
    q2 = -3*t*t*t + 3*t*t + 3*t + 1
    q3 = t*t*t
    x = (q0*p0[0] + q1*p1[0] + q2*p2[0] + q3*p3[0])/6
    return x

def bspline3_get_points_on_segment(s, points):
    plen = len(points)
    p0i = clamp(s - 2, 0, plen - 1)
    p1i = clamp(s - 1, 0, plen - 1)
    p2i = clamp(s, 0, plen - 1)
    p3i = clamp(s + 1, 0, plen - 1)
    return points[p0i], points[p1i], points[p2i], points[p3i]

def bspline3_from_points(points):
    """
        Internally highly dependens on result of pycontinuum_points,
        e.g. there should be results[0][0] == 0.
        Also it gives x coordinates always as integers,
        and assumes there is not two stitching points between integers values.
    """
    segments = len(points) + 1
    
    # Calculate points where segments stitch together.
    # Also includes first point and last point.
    x_stitching_points = [0]
    for s in range(segments):
        p0, p1, p2, p3 = bspline3_get_points_on_segment(s, points)
        x_stitching_points.append(bspline3x_t(1.0, p0, p1, p2, p3))
    
    # Fill results with pairs (i, y(i)),
    # where i is x coordinate and also index in signature array,
    # and y(i) is respective  y coordinate of bspline3.
    results = np.empty(points[-1][0] + 1)
    for s in range(segments):
        p0, p1, p2, p3 = bspline3_get_points_on_segment(s, points)
        # We need special case for the first element,
        # as we need integer x in results, but first element could
        # be rounded down to x point that is outside this segment.
        # That leads to errors in equations, where we get complex solutions.
        x = x_stitching_points[s]
        results[int(x)] = bspline3y_x(x, p0, p1, p2, p3)
        for x in range(int(x_stitching_points[s]) + 1, int(x_stitching_points[s + 1])):
            results[x] = bspline3y_x(x, p0, p1, p2, p3)
    
    # Last stitching (accually not stitching, since it is the last one of stitching points)
    # of the last segment is not included because of non-inclusive ranges.
    # We still need it.
    p0, p1, p2, p3 = bspline3_get_points_on_segment(segments - 1, points)
    x = x_stitching_points[-1]
    results[int(x)] = bspline3y_x(x, p0, p1, p2, p3)

    return np.array(results)