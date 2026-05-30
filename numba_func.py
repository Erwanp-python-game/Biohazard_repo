
from numba import njit,prange
import numpy as np
from math import *
@njit(parallel=True, fastmath=True, cache=True)
def intersect(c3, a0, a1, counter_, screenV, screenP, cell_start, cell_count, cell_objects, cell_size,
              all_a, all_b, all_X,
              all_aa, all_bb, all_n, all_ab, all_inv_det, all_opening, all_freq, all_phase, all_tile_z, all_trans_im,
              all_format, all_wall_im, all_light, all_light_w, all_wall_len, all_destruc, all_wall_im2, all_side,
              TORCHE, torch_on, torch_shine, fire, explo, explo_pt, random_explo, all_liquid,all_sphere,all_radius):
    X0 = screenP[0, 0]
    liquid = False
    origin_x = 0.5 * (X0[0] + 100.0)
    origin_y = 0.5 * (X0[1] + 100.0)

    I0x = int(origin_x) // cell_size
    I0y = int(origin_y) // cell_size

    w = 160 * 2
    h = 80 * 2

    explo_R = 4 * explo
    if explo != 0:
        c0 = np.cos(a0)
        s0 = np.sin(-a0)
        c1 = np.cos(a1)
        s1 = np.sin(a1)

    Im = np.full((w, h, 3), 0, dtype=np.float32)
    Im2 = np.full((w, h, 4), 0, dtype=np.float32)
    Im_liquid = np.full((w, h, 3), 0, dtype=np.float32)
    S = np.full((w, h, 3), 1e6, dtype=np.float32)
    S_liquid = np.full((w, h, 3), 1e6, dtype=np.float32)
    S_explo = np.full((w, h, 3), 1e6, dtype=np.float32)
    Xl = np.full((w, h, 3), 1e6, dtype=np.float32)
    POS_l = np.full((w, h), 1e6, dtype=np.float32)
    wall_ind = np.zeros((w, h), np.int32)
    wall_ind_liquid = np.zeros((w, h), np.int32)

    n_obj = len(all_a)
    torch_glob = np.full((w), False, dtype=bool)
    for i in prange(w):

        # Thread-local arrays
        visited = np.empty((h, n_obj), np.uint8)
        visited[:] = 0

        t_int = np.empty(h, np.float32)
        for jj in range(h):
            t_int[jj] = 1e9

        # Precompute rays for this row once
        rays = np.empty((h, 3), np.float32)
        i0 = i // 4

        i1 = (i + 1) // 4
        if i1 > w // 4 - 1:
            i1 = w // 4 - 1

        i2 = (i + 2) // 4
        if i2 > w // 4 - 1:
            i2 = w // 4 - 1

        i3 = (i + 3) // 4
        if i3 > w // 4 - 1:
            i3 = w // 4 - 1

        for j in range(h):
            j0 = j // 4
            j1 = (j + 1) // 4
            j2 = (j + 2) // 4
            j3 = (j + 3) // 4
            if j1 > h // 4 - 1:
                j1 = h // 4 - 1
            if j2 > h // 4 - 1:
                j2 = h // 4 - 1
            if j3 > h // 4 - 1:
                j3 = h // 4 - 1

            r0 = screenV[i0, j0]
            r1 = screenV[i1, j1]
            r2 = screenV[i2, j2]
            r3 = screenV[i3, j3]
            rays[j, 0] = 0.25 * (r0[0] + r1[0] + r2[0] + r3[0])
            rays[j, 1] = 0.25 * (r0[1] + r1[1] + r2[1] + r3[1])
            rays[j, 2] = 0.25 * (r0[2] + r1[2] + r2[2] + r3[2])

        # DDA setup
        ix = I0x
        iy = I0y

        ray0 = rays[h // 4]  # representative ray for stepping
        dx = ray0[0]
        dy = ray0[1]

        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx != 0.0:
            next_x = (ix + (step_x > 0)) * cell_size
            t_max_x = (next_x - origin_x) / dx
            t_delta_x = cell_size / abs(dx)
        else:
            t_max_x = 1e9
            t_delta_x = 1e9

        if dy != 0.0:
            next_y = (iy + (step_y > 0)) * cell_size
            t_max_y = (next_y - origin_y) / dy
            t_delta_y = cell_size / abs(dy)
        else:
            t_max_y = 1e9
            t_delta_y = 1e9

        if t_max_x < t_max_y:
            ix -= step_x
            t_max_x -= t_delta_x
        else:
            iy -= step_y
            t_max_y -= t_delta_y

        # Grid traversal
        for g in range(100):

            if t_max_x < t_max_y:
                ix += step_x
                t = t_max_x
                t_max_x += t_delta_x
            else:
                iy += step_y
                t = t_max_y
                t_max_y += t_delta_y

            # Clamp cell indices
            cx = ix
            cy = iy
            if cx < 0:
                cx = 0
            elif cx >= 500 // cell_size:
                cx = 500 // cell_size - 1

            if cy < 0:
                cy = 0
            elif cy >= 500 // cell_size:
                cy = 500 // cell_size - 1

            cell_id = cx + cy * (500 // cell_size)

            start = cell_start[cell_id]
            count = cell_count[cell_id]

            # For each subpixel
            for j in range(h):

                if t_int[j] < t:
                    continue

                ray = rays[j]
                if explo != 0:
                    # ---- EXPLOSION TEST (INLINE) ----
                    explo_R0 = explo_R + random_explo[i, j]
                    dx0 = X0[0] - explo_pt[0]
                    dy0 = X0[1] - explo_pt[1]
                    dz0 = X0[2] - explo_pt[2]

                    a = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]
                    b = dx0 * ray[0] + dy0 * ray[1] + dz0 * ray[2]
                    c = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 - explo_R0 * explo_R0

                    disc = b * b - a * c

                    if disc > 0.0:

                        sqrt_disc = np.sqrt(disc)

                        t0 = (-b - sqrt_disc) / a
                        t1 = (-b + sqrt_disc) / a

                        if t0 > 0.0:
                            t_exp = t0
                        elif t1 > 0.0:
                            t_exp = t1
                        else:
                            t_exp = 1e9

                        if t_exp < t_int[j]:
                            # mark explosion hit
                            # rotate
                            dx1 = X0[0] + t_exp * ray[0] - explo_pt[0]
                            dy1 = X0[1] + t_exp * ray[1] - explo_pt[1]
                            dz1 = X0[2] + t_exp * ray[2] - explo_pt[2]
                            x1 = c0 * dx1 + s0 * dy1
                            y1 = -s0 * dx1 + c0 * dy1
                            z1 = dz1

                            y2 = y1
                            z2 = s1 * x1 + c1 * z1

                            # camera-plane distance
                            dist2 = y2 * y2 + z2 * z2
                            S_explo[i, j, 1] = dist2
                            S_explo[i, j, 2] = t_exp

                for k in range(start, start + count):
                    obj = cell_objects[k]

                    if visited[j, obj] == 1:
                        continue

                    visited[j, obj] = 1

                    a = all_a[obj]
                    b = all_b[obj]
                    n = all_n[obj]
                    X = all_X[obj]

                    aa = all_aa[obj]
                    bb = all_bb[obj]
                    ab = all_ab[obj]
                    inv_det = all_inv_det[obj]

                    # ---- INLINE INTERSECTION ----
                    if all_sphere[obj]==0:
                        denom = ray[0] * n[0] + ray[1] * n[1] + ray[2] * n[2]

                        if denom > 1e-9 or denom < -1e-9:

                            dx0 = X[0] - X0[0]
                            dy0 = X[1] - X0[1]
                            dz0 = X[2] - X0[2]

                            t_ = (dx0 * n[0] + dy0 * n[1] + dz0 * n[2]) / denom

                            if 0.0 < t_ < t_int[j]:

                                px = X0[0] + t_ * ray[0]
                                py = X0[1] + t_ * ray[1]
                                pz = X0[2] + t_ * ray[2]

                                # if n[0] == 0 and n[1] == 0:
                                #
                                #     if pz < cell_array_z[cx][cy][0]  and X[2] > 0:
                                #         visited[j, obj] = 0
                                #         continue
                                #     if pz > cell_array_z[cx][cy][1]  and X[2] < 0:
                                #         visited[j, obj] = 0
                                #         continue

                                ax = px - X[0]
                                ay = py - X[1]
                                az = pz - X[2]

                                if ab == 0:
                                    u = (ax * a[0] + ay * a[1] + az * a[2]) / aa
                                    v = (ax * b[0] + ay * b[1] + az * b[2]) / bb
                                else:
                                    da = ax * a[0] + ay * a[1] + az * a[2]
                                    db = ax * b[0] + ay * b[1] + az * b[2]
                                    u = (da * bb - db * ab) * inv_det
                                    v = (db * aa - da * ab) * inv_det

                                if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:

                                    open = True
                                    if all_opening[obj]:
                                        f = all_format[obj]
                                        iu = int((1 - u) * f[0])
                                        iv = int((1 - v) * f[1])
                                        tile_z = all_tile_z[obj]
                                        gu = iu % 120
                                        gv = iv % 120
                                        freq = all_freq[obj]
                                        if ((-iv // 120 + all_phase[obj] - freq + 1) % freq) == 0:
                                            shift = 120
                                            if iu // 120 > 0 and not (tile_z):
                                                shift = 0

                                        else:
                                            shift = 0
                                        if all_destruc[obj] < 0:
                                            ind = counter_ // (12 // all_wall_len[obj])
                                        else:
                                            ind = all_destruc[obj]
                                        trans = all_trans_im[obj][ind]
                                        open = trans[gu, gv + shift]
                                    if all_liquid[obj]:
                                        open = False
                                        S_liquid[i, j, 0] = u
                                        S_liquid[i, j, 1] = v
                                        S_liquid[i, j, 2] = t_

                                        wall_ind_liquid[i, j] = obj
                                    if open:
                                        t_int[j] = t_
                                        S[i, j, 0] = u
                                        S[i, j, 1] = v
                                        S[i, j, 2] = t_

                                        wall_ind[i, j] = obj
                                        Xl[i, j, 0] = px
                                        Xl[i, j, 1] = py
                                        Xl[i, j, 2] = pz

                    elif all_sphere[obj]==1:
                        # ---- SPHERE ----

                        C = all_X[obj]  # center
                        r = all_radius[obj]  # you must add this array

                        dx0 = X0[0] - C[0]
                        dy0 = X0[1] - C[1]
                        dz0 = X0[2] - C[2]

                        a = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]
                        b = dx0 * ray[0] + dy0 * ray[1] + dz0 * ray[2]
                        c = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 - r * r

                        disc = b * b - a * c


                        if disc > 0.0:
                            t_ = (-b - np.sqrt(disc))/a

                            if 0.0 < t_ < t_int[j]:
                                px = X0[0] + t_ * ray[0]
                                py = X0[1] + t_ * ray[1]
                                pz = X0[2] + t_ * ray[2]

                                # simple spherical UV (optional)
                                nx = (px - C[0]) / r
                                ny = (py - C[1]) / r
                                nz = (pz - C[2]) / r

                                if nz > 1.0:
                                    nz = 1.0
                                elif nz < -1.0:
                                    nz = -1.0

                                v = 0.5 + np.arctan2(ny, nx) / (2 * np.pi)
                                u = 0.5 - np.arcsin(nz) / np.pi

                                t_int[j] = t_

                                S[i, j, 0] = u
                                S[i, j, 1] = v
                                S[i, j, 2] = t_

                                wall_ind[i, j] = obj

                                Xl[i, j, 0] = px
                                Xl[i, j, 1] = py
                                Xl[i, j, 2] = pz
                    else :
                        # ---- SPHERE ----

                        C = all_X[obj]  # center
                        r = all_radius[obj]  # you must add this array

                        dx0 = X0[0] - C[0]
                        dy0 = X0[1] - C[1]
                        dz0 = X0[2] - C[2]

                        a = ray[0] * ray[0] + ray[1] * ray[1]
                        b = dx0 * ray[0] + dy0 * ray[1]
                        c = dx0 * dx0 + dy0 * dy0 - r * r

                        disc = b * b - a * c


                        if disc > 0.0:
                            t_ = (-b - np.sqrt(disc))/a

                            if 0.0 < t_ < t_int[j]:
                                px = X0[0] + t_ * ray[0]
                                py = X0[1] + t_ * ray[1]
                                pz = X0[2] + t_ * ray[2]

                                # simple spherical UV (optional)
                                nx = (px - C[0]) / r
                                ny = (py - C[1]) / r
                                nz = (pz - C[2]) / r

                                # if nz > 1.0:
                                #     nz = 1.0
                                # elif nz < -1.0:
                                #     nz = -1.0

                                v = 0.5 + np.arctan2(ny, nx) / (2 * np.pi)
                                u = nz

                                t_int[j] = t_

                                S[i, j, 0] = u
                                S[i, j, 1] = v
                                S[i, j, 2] = t_

                                wall_ind[i, j] = obj

                                Xl[i, j, 0] = px
                                Xl[i, j, 1] = py
                                Xl[i, j, 2] = pz
            # Early exit check
            done = True
            for jj in range(h):
                if t_int[jj] >= t:
                    done = False
                    break

            if done or g == 99:

                for jj in range(h):

                    obj1 = wall_ind[i, jj]
                    tile_z = all_tile_z[obj1]
                    u = S[i, jj, 0]
                    v = S[i, jj, 1]
                    f = all_format[obj1]

                    iu = int((1 - u) * f[0])
                    iv = int((1 - v) * f[1])

                    gu = iu % 120
                    gv = iv % 120
                    freq = all_freq[obj1]
                    if ((-iv // 120 + all_phase[obj1] - freq + 1) % freq) == 0:
                        shift = 120
                        if iu // 120 > 0 and not (tile_z):
                            shift = 0
                    else:
                        shift = 0
                    if all_destruc[obj1] < 0:
                        ind = counter_ // (12 // all_wall_len[obj1])
                    else:
                        ind = all_destruc[obj1]
                    if all_side[obj1] < 0:
                        im = all_wall_im[obj1][ind]
                    else:
                        im = all_wall_im2[obj1][ind]
                    Cl = all_light_w[obj1]
                    r = im[gu, gv + shift, 0]

                    light_x = all_light[obj1]
                    dm = 1e6
                    for k in (light_x):
                        dx2 = k[0] - Xl[i, jj, 0]
                        dy2 = k[1] - Xl[i, jj, 1]
                        dz2 = k[2] - Xl[i, jj, 2]
                        d = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
                        if dm > d:
                            dm = d
                    dd = np.sqrt(dm)
                    POS_l[i, jj] = dd
                    if r == -1:
                        Im[i, jj, 0] = 0
                        Im[i, jj, 1] = 255
                        Im[i, jj, 2] = 255
                        POS_l[i, jj] = 1e-9
                    else:
                        torch2 = 1.
                        if torch_shine:
                            torch2 = TORCHE[i, jj, 0]
                        if dm == 1e6:
                            if torch_on:
                                torch_glob[i] = True
                                dz = S[i, jj, 2]
                                torch = torch2 * (0.8 * 1 / (0.01 * 16 * dz) + 1 / (0.1 * np.sqrt(dz)) + 0.2)
                            else:
                                torch = 0.
                        else:
                            if torch_shine and torch_on:
                                dz = S[i, jj, 2]
                                torch = torch2 * (0.8 * 1 / (0.01 * 16 * dz) + 1 / (0.1 * np.sqrt(dz)) + 0.2)
                            else:
                                torch = (0.8 * 1 / (0.01 * 16 * dm) + 1 / (0.1 * dd) + 0.2)

                        f = 0.
                        if fire:
                            f = 100 * TORCHE[i, jj, 0]
                        if explo != 0:
                            f = 40 * explo

                        Im[i, jj, 0] = r * Cl[0] * torch + f
                        Im[i, jj, 1] = im[gu, gv + shift, 1] * Cl[1] * torch + f
                        Im[i, jj, 2] = im[gu, gv + shift, 2] * Cl[2] * torch + f

                    if explo != 0 and S_explo[i, jj, 2] < S[i, jj, 2]:
                        de = np.sqrt(S_explo[i, jj, 1]) + random_explo[i, jj]
                        if de < explo_R / 2:
                            Im2[i, jj, 0] = 255 * ((1 - 2 * de / explo_R) + 2 * de / explo_R)
                            Im2[i, jj, 1] = 255 * ((1 - 2 * de / explo_R) + 2 * de / explo_R)
                            Im2[i, jj, 2] = 255 * (1 - 2 * de / explo_R)
                            Im2[i, jj, 3] = 125
                        else:
                            Im2[i, jj, 0] = 255 * (
                                        (1 - 2 * (de - 0.5 * explo_R) / explo_R) + 2 * (de - 0.5 * explo_R) / explo_R)
                            Im2[i, jj, 1] = 255 * ((1 - 2 * (de - 0.5 * explo_R) / explo_R))
                            Im2[i, jj, 3] = 125

                    if wall_ind_liquid[i, jj] != 0 and S_liquid[i, jj, 2] < S[i, jj, 2]:

                        obj1 = wall_ind_liquid[i, jj]
                        tile_z = all_tile_z[obj1]
                        u = S_liquid[i, jj, 0]
                        v = S_liquid[i, jj, 1]
                        f = all_format[obj1]

                        iu = int((1 - u) * f[0] + c3 * 0.5)
                        iv = int((1 - v) * f[1] + c3 * 0.5)

                        gu = iu % 120
                        gv = iv % 120
                        freq = all_freq[obj1]
                        if ((-iv // 120 + all_phase[obj1] - freq + 1) % freq) == 0:
                            shift = 120
                            if iu // 120 > 0 and not (tile_z):
                                shift = 0
                        else:
                            shift = 0
                        if all_destruc[obj1] < 0:
                            ind = counter_ // (12 // all_wall_len[obj1])
                        else:
                            ind = all_destruc[obj1]
                        if all_side[obj1] < 0:
                            im = all_wall_im[obj1][ind]
                        else:
                            im = all_wall_im2[obj1][ind]
                        Cl = all_light_w[obj1]
                        r = im[gu, gv + shift, 0]

                        light_x = all_light[obj1]
                        dm = 1e6
                        for k in (light_x):
                            dx2 = k[0] - Xl[i, jj, 0]
                            dy2 = k[1] - Xl[i, jj, 1]
                            dz2 = k[2] - Xl[i, jj, 2]
                            d = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
                            if dm > d:
                                dm = d
                        dd = np.sqrt(dm)
                        POS_l[i, jj] = dd
                        torch2 = 1.
                        if torch_shine:
                            torch2 = TORCHE[i, jj, 0]
                        if dm == 1e6:
                            if torch_on:
                                torch_glob[i] = True
                                dz = S[i, jj, 2]
                                torch = torch2 * (0.8 * 1 / (0.01 * 16 * dz) + 1 / (0.1 * np.sqrt(dz)) + 0.2)
                            else:
                                torch = 0.
                        else:
                            if torch_shine and torch_on:
                                dz = S[i, jj, 2]
                                torch = torch2 * (0.8 * 1 / (0.01 * 16 * dz) + 1 / (0.1 * np.sqrt(dz)) + 0.2)
                            else:
                                torch = (0.8 * 1 / (0.01 * 16 * dm) + 1 / (0.1 * dd) + 0.2)

                        f = 0.
                        if fire:
                            f = 100 * TORCHE[i, jj, 0]
                        if explo != 0:
                            f = 40 * explo

                        Im_liquid[i, jj, 0] = r * Cl[0] * torch + f
                        Im_liquid[i, jj, 1] = im[gu, gv + shift, 1] * Cl[1] * torch + f
                        Im_liquid[i, jj, 2] = im[gu, gv + shift, 2] * Cl[2] * torch + f

                break

    liquid = (wall_ind_liquid != 0).any()
    return S, wall_ind, Xl, Im, POS_l, np.sum(torch_glob) > 50, Im2, Im_liquid, liquid, S_liquid


@njit(fastmath=True, cache=True)
def thing_render(counter, counter2, a0, a1, x_perso, all_x_e, Im, S, all_RA, all_im_m, all_im_o, all_obj_mon,
                 all_types_e, all_angle, all_ima_m, all_mort, all_attack_range, all_range, all_light_e, all_im_o_d,
                 all_destr, TORCHE3, torch_on, Im_liquid, liquid, S_liquid, boss_im,scrnL,TAN1,TAN2):
    c0 = np.cos(a0)
    s0 = np.sin(-a0)

    c1 = np.cos(a1)
    s1 = np.sin(a1)

    W = scrnL[0] * 2 * 2
    H = scrnL[1] * 2 * 2

    f1 = scrnL[0] * 2 / TAN2
    f2 = scrnL[1] * 2 / TAN1

    index_e = np.full((W, H), -1, dtype=np.int64)
    depth_e = np.full((W, H), 1e6, dtype=np.float64)
    for i in range(len(all_x_e)):
        x_e = all_x_e[i]
        mort = int(all_mort[i])
        if all_obj_mon[i] == 1:

            if mort == 0:
                if all_attack_range[i]:
                    if all_range[i]:
                        f = counter2 // 8
                    else:
                        f = counter // 4
                    im = all_ima_m[all_types_e[i], f, :, :, :]
                else:
                    im = all_im_m[all_types_e[i], counter // 3, int(all_angle[i] // 45), :, :, :]
            else:
                im = all_ima_m[all_types_e[i], 4 + mort, :, :, :]

        if all_obj_mon[i] == 0:
            if mort == 0:
                im = all_im_o[all_types_e[i], int(all_angle[i] // 45), :, :, :]
            else:
                im = all_im_o_d[all_types_e[i], mort, :, :, :]
        if all_obj_mon[i] == 2:
            im = boss_im

        d = x_e - x_perso
        dx, dy, dz = d

        RA = all_RA[i]
        x1 = c0 * dx + s0 * dy
        if x1 > 0:
            y1 = -s0 * dx + c0 * dy
            z1 = -dz + (0.75 * RA - 5)

            x2 = c1 * x1 - s1 * z1
            y2 = y1
            z2 = s1 * x1 + c1 * z1

            sx = int(W * 0.5 + f1 * y2 / x2)
            sy = int(H * 0.5 - f2 * z2 / x2)
            width = int(RA * W / x1)
            if sx + width // 2 > 0 and sx - width // 2 < W and sy + width // 2 > 0 and sy - width // 2 < H:  # and S[sx,sy,2]>x1:

                for gx in range(0, width):
                    ix = sx - width // 2 + gx
                    ix_r = int(160 * gx / width)
                    if ix >= 0 and ix < W and ix_r < 160:
                        for gy in range(0, width):
                            iy = sy - width // 2 + gy
                            iy_r = int(160 * gy / width)
                            if iy >= 0 and iy < H and S[ix, iy, 2] > x1 and depth_e[ix, iy] > x1 and iy_r < 160:
                                r = im[ix_r, iy_r, 0]
                                g = im[ix_r, iy_r, 1]
                                b = im[ix_r, iy_r, 2]
                                if r + g + b > 0:
                                    l = all_light_e[i]
                                    if torch_on:
                                        l = l * TORCHE3[ix, iy, 0] / (0.1 * np.sqrt(x1))
                                    Im[ix, iy, 0] = r * l[0]
                                    Im[ix, iy, 1] = g * l[1]
                                    Im[ix, iy, 2] = b * l[2]
                                    index_e[ix, iy] = i
                                    depth_e[ix, iy] = x1
    d = np.minimum(depth_e, S[:, :, 2])
    if liquid:
        for i in range(W):
            for j in range(H):
                if S_liquid[i, j, 2] < d[i, j] and S_liquid[i, j, 2] != 1e6:
                    Im[i, j, 0] = (Im[i, j, 0] + Im_liquid[i, j, 0]) / 2
                    Im[i, j, 1] = (Im[i, j, 1] + Im_liquid[i, j, 1]) / 2
                    Im[i, j, 2] = (Im[i, j, 2] + Im_liquid[i, j, 2]) / 2
    return Im, index_e, d
