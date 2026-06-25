from rle import *
from commonfunctions import *
from collections import Counter


row_percentage = 0.3


def calculate_thickness_spacing(rle, most_common):
    bw_patterns = [most_common_bw_pattern(col, most_common) for col in rle]
    bw_patterns = [x for x in bw_patterns if x]  # Filter empty patterns

    flattened = []
    for col in bw_patterns:
        flattened += col

    pair, count = Counter(flattened).most_common()[0]

    line_thickness = min(pair)
    line_spacing = max(pair)

    return line_thickness, line_spacing


def whitene(rle, vals, max_height):
    rlv = []
    for length, value in zip(rle, vals):
        if value == 0 and length < 1.1*max_height:
            value = 1
        rlv.append((length, value))

    n_rle, n_vals = [], []
    count = 0
    for length, value in rlv:
        if value == 1:
            count = count + length
        else:
            if count > 0:
                n_rle.append(count)
                n_vals.append(1)

            count = 0
            n_rle.append(length)
            n_vals.append(0)
    if count > 0:
        n_rle.append(count)
        n_vals.append(1)

    return n_rle, n_vals


def remove_staff_lines(rle, vals, thickness, shape):
    n_rle, n_vals = [], []
    for i in range(len(rle)):
        rl, val = whitene(rle[i], vals[i], thickness)
        n_rle.append(rl)
        n_vals.append(val)

    return hv_decode(n_rle, n_vals, shape)


def remove_staff_lines_2(thickness, img_with_staff):
    img = img_with_staff.copy()
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img[i][j] == 1
        projected.append([1]*proj_sum + [0]*(cols-proj_sum))
        if(proj_sum <= row_percentage*cols):
            img[i, :] = 1
    closed = binary_opening(img, np.ones((3*thickness, 1)))
    return closed


def get_rows(start, most_common, thickness, spacing):
    # start = start-most_common
    rows = []
    num = 6
    if start - most_common >= 0:
        start -= most_common
        num = 7
    for k in range(num):
        row = []
        for i in range(thickness):
            row.append(start)
            start += 1
        start += (spacing)
        rows.append(row)
    if len(rows) == 6:
        rows = [0] + rows
    return rows


def horizontal_projection(img):
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img[i][j] == 1
        projected.append([1]*proj_sum + [0]*(cols-proj_sum))
        if(proj_sum <= 0.1*cols):
            return i
    return 0


def get_staff_row_position(img):
    found = 0
    row_position = -1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] == 0):
                row_position = i
                found = 1
                break
        if found == 1:
            break
    return row_position


def coordinator(bin_img, horizontal):
    rle, vals = hv_rle(bin_img)
    most_common = get_most_common(rle)
    thickness, spacing = calculate_thickness_spacing(rle, most_common)
    
    # Ajustamos un porcentaje más alto y seguro para no borrar barras de corcheas
    global row_percentage
    row_percentage = 0.65  # Sube del 0.3 al 0.65 para proteger figuras musicales
    
    # Usamos el método de remoción geométrica por filas
    no_staff_img = remove_staff_lines_2(thickness, bin_img)
    
    # Aislamos las líneas restando la imagen original de la limpia
    staff_lines = otsu(bin_img - no_staff_img)
    
    # Proyección horizontal para encontrar la VERDADERA primera línea densa
    # Esto evita que los números de digitación o las llaves rompan el 'start'
    projected_rows = np.sum(staff_lines == 0, axis=1) # Contamos píxeles negros por fila
    
    # Buscamos la primera fila donde la densidad supere una parte significativa del ancho
    start = 0
    for idx, val in enumerate(projected_rows):
        if val > (bin_img.shape[1] * 0.4): # Umbral de densidad de línea real
            start = idx
            break
            
    # Si por alguna razón da 0, usamos el fallback geométrico
    if start == 0:
        start = horizontal_projection(bin_img)
        
    # Reconstruimos las posiciones virtuales de referencia para las octavas
    staff_row_positions = get_rows(start, most_common, thickness, spacing)
    staff_row_positions = [np.average(x) for x in staff_row_positions]
    
    return spacing, staff_row_positions, no_staff_img
