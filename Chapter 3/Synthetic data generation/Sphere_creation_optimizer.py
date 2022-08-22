from cvxpy import *
import numpy as np
import dccp
import random

def generate_rs_ns_ellipsie(border_constraints = [16,16,16], count_max = 50, min_ratio = 0.4, min_r=8):
    '''
    :param border_constraints: height/length/width
    :param count_max: how many spheres to fill in one cube
    :param min_ratio: min threshold ratio between filled spheres/entire space
    :param min_r: min_r to not collide with boundaries
    :return: list of rs
    '''
    loop_val= 0
    border_constraint = np.min(border_constraints)
    max_r_values = list(range(min_r, int(border_constraint / 2)))
    r_values_list = []
    n_values_list = []

    for i in range(len(max_r_values)):
        r_values_list = r_values_list + np.repeat(max_r_values[i], int(border_constraint / max_r_values[i])).tolist()
        n_values_list.append(int(border_constraint / max_r_values[i]))

    count = 0
    count_actual = 0
    temp = []
    max_vol =border_constraints[0] *border_constraints[1]*border_constraints[2]

    while (count < count_max  or loop_val == 1000) and count_actual < count_max:
        loop_val += 1
        random_number = random.choices(n_values_list, k=1)
        temp1 = np.array(random.choices(r_values_list,k=random_number[0]*3)).reshape(random_number[0],3).tolist()
        whole_vol = 0
        flag_temp = []
        for index, r_vals in enumerate(temp1):
            flag_temp.append(np.sum(r_vals) <= int(border_constraint))
            cube_vol = (4 / 3) * np.pi * (r_vals[0] * r_vals[1]* r_vals[2])
            whole_vol += cube_vol
        ratio = whole_vol / max_vol
        if sum(flag_temp) == 3 and ratio >= min_ratio:
            count_actual += len(flag_temp)
            count += 1
            temp.append(temp1)
    return temp



def fill_cube_with_spheres_diff_rs_ns(border_constraints = [16,16,16], count_max = 50, min_ratio = 0.4, min_r=8):
    '''
    Use optimization to fill in the cubes with different spheres
    :param border_constraints: height/length/width
    :param count_max: how many spheres to fill in one cube
    :param min_ratio: min threshold ratio between filled spheres/entire space
    :param min_r: min_r to not collide with boundaries
    :return: temp, center_list, radius_list
    '''
    temp1 = generate_rs_ns_ellipsie(border_constraints, count_max, min_ratio, min_r=min_r)
    print('radius generated', len(temp1))
    temp = []
    center_list = []
    radius_list = []
    counter = 0
    for set_r_values in temp1:
        counter += 1
        r = set_r_values
        n = len(set_r_values)
        c = Variable(shape=(n,3))
        constr = []
        rconst = [np.max(r_list) for r_list in set_r_values]
        for i in range(n-1):
            for j in range(i+1,n):
                a = np.array([r[i][0],0,0])
                a2 = np.array([0,r[i][1],0])
                a3 = np.array([0,0,r[i][2]])

                constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=border_constraints[0])
                constr.append(norm((c[i,:]-a2)-(c[i,:]+a2))<=border_constraints[1])
                constr.append(norm((c[i,:]-a3)-(c[i,:]+a3))<=border_constraints[2])

        prob = Problem(Minimize(max(max(abs(c),axis=1)+rconst)), constr)
        prob.solve(method = 'dccp', ccp_times = 1)
        l = max(max(abs(c),axis=1)+rconst).value*2
        pi = np.pi
        ratio = pi*sum(square(rconst)).value/square(l).value
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]


        for i in range(n):
          x = r[i][0] * np.cos(u) * np.sin(v)
          y = r[i][1] * np.sin(u) * np.sin(v)
          z = r[i][2] * np.cos(v)

          x2 = c[i,0].value+x+border_constraints[0]/2
          y2 = c[i,1].value+y+border_constraints[1]/2
          z2 = c[i,2].value+z+border_constraints[2]/2
          x1 = x2.reshape(x2.shape[0],x2.shape[1],1)
          y1 = y2.reshape(y2.shape[0],y2.shape[1],1)
          z1 = z2.reshape(z2.shape[0],z2.shape[1],1)

          a = np.concatenate((x1,y1,z1),axis=2)
          a = a.reshape(a.shape[0]*a.shape[1],3)
          max_list, min_list,center = print_cent(a)

          temp2.append([max_list, min_list,center, r[i],a])
          temp3.append(center.tolist())
          temp4.append(r[i])
        center_list.append(temp3)
        radius_list.append(temp4)
        temp.append(temp2)
        whole_vol = 0
        max_vol = border_constraints[0] *border_constraints[1]*border_constraints[2]
        for i in r:
          cube_vol = (4/3) *np.pi* (i[0]*i[1]*i[2])
          whole_vol += cube_vol
        print(counter, ' points generated')

    return temp, center_list, radius_list

def get_target_points(center_list, radius_list, target_points=7,space_dim=3):
    '''
    Returns the target points
    :param center_list: centers
    :param radius_list: radius list
    :param target_points: number of desired target points
    :param space_dim: dimension of space
    :return: center_list_new, radius_list_new, additional_points_list
    '''
    center_list_new = []
    radius_list_new = []
    additional_points_list = []
    center_list_new = [b for i in center_list for b in i]
    radius_list_new = [b for i in radius_list for b in i]
    for j in range(len(center_list_new)):
        additional_points_temp = []
        center = center_list_new[j]
        r = radius_list_new[j]
        if target_points == 7:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              additional_points_temp.append((center + int(r[j] / 2) * change).tolist()[0])
              additional_points_temp.append((center - int(r[j] / 2) * change).tolist()[0])

        elif target_points == 10:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              for delta in range(-r[j], r[j] + 1):
                  temp = (center + int(delta) * change).tolist()[0]
                  temp1 = [int(x) for x in temp]
                  additional_points_temp.append(temp1)
        elif target_points == 13:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              for delta in [-r[j], -int(r[j] / 2), int(r[j] / 2), r[j]]:
                  temp = (center + int(delta) * change).tolist()[0]
                  temp1 = [int(x) for x in temp]
                  additional_points_temp.append(temp1)
        else:
          additional_points_temp.append(center)
        additional_points_list.append(additional_points_temp)

    return center_list_new, radius_list_new, additional_points_list


