from commonfunctions import *
from pre_processing import *
from connected_componentes import *
from staff import calculate_thickness_spacing, remove_staff_lines, coordinator
from segmenter import Segmenter
from fit import predict
from glob import glob
import os
import cv2
import pickle
from scipy.ndimage import binary_fill_holes
from skimage.morphology import thin
import argparse
from skimage.transform import resize
import multiprocessing
from functools import partial

# ==============================================================================
# 1. MAPAS DE TRADUCCIÓN INDEPENDIENTES (CLAVE DE SOL Y FA)
# ==============================================================================
label_map_sol = {
    0: {0: 'g5', 1: 'f5'}, 1: {0: 'f5', 1: 'e5'}, 2: {0: 'd5', 1: 'c5'},
    3: {0: 'b4', 1: 'a4'}, 4: {0: 'g4', 1: 'f4'}, 5: {0: 'e4', 1: 'd4'},
    6: {0: 'c4', 1: 'b3'}, 7: {0: 'a3', 1: 'g3'}
}

label_map_fa = {
    0: {0: 'b3', 1: 'a3'}, 1: {0: 'a3', 1: 'g3'}, 2: {0: 'f3', 1: 'e3'},
    3: {0: 'd3', 1: 'c3'}, 4: {0: 'b2', 1: 'a2'}, 5: {0: 'g2', 1: 'f2'},
    6: {0: 'e2', 1: 'd2'}, 7: {0: 'c2', 1: 'b1'}
}

note_map = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D',
    'e': 'E', 'f': 'F', 'g': 'G'
}

alteration_map = {
    '##': '##', '#': '#',
    '&&': 'bb', '&': 'b', '': ''
}

duration_map = {
    '1': 'R', 'a_1': 'R',
    '2': 'B', 'a_2': 'B',
    '4': 'N', 'a_4': 'N',
    '8': 'C', 'a_8': 'C', '8_b_n': 'C', '8_b_r': 'C',
    '16': 'S', 'a_16': 'S', '16_b_n': 'S', '16_b_r': 'S',
    '32': 'F', 'a_32': 'F', '32_b_n': 'F', '32_b_r': 'F',
}


def guido_to_readable(note_guido):
    dotted = note_guido.endswith('.')
    note_clean = note_guido.rstrip('.')
    if '/' not in note_clean:
        return note_guido
    note_code, dur_code = note_clean.split('/')
    name = note_code[0]
    rest = note_code[1:]
    if rest.startswith('##'):
        alt, octave = '##', rest[2:]
    elif rest.startswith('&&'):
        alt, octave = '&&', rest[2:]
    elif rest.startswith('#'):
        alt, octave = '#', rest[1:]
    elif rest.startswith('&'):
        alt, octave = '&', rest[1:]
    else:
        alt, octave = '', rest
    note_name = note_map.get(name, name.upper())
    alt_name = alteration_map.get(alt, '')
    duration_name = duration_map.get(dur_code, dur_code)
    dot = '.' if dotted else ''
    # Formato: <nota><alteracion> <octava> <duracion><puntillo>
    return f"{note_name}{alt_name} {octave} {duration_name}{dot}"


def estim_note(c, idx, imgs_spacing, imgs_rows, region_idx):
    """
    Determina el nombre de la nota musical y su octava basándose en la posición del píxel (c)
    y en el índice del pentagrama (region_idx) para alternar entre Sol y Fa.
    """
    spacing = imgs_spacing[idx]
    rows = imgs_rows[idx]
    margin = 1 + (spacing / 4)
    
    # Alternancia: Índices pares (0, 2, 4...) = Sol || Índices impares (1, 3, 5...) = Fa
    current_map = label_map_sol if (region_idx % 2 == 0) else label_map_fa
    
    for index, line in enumerate(rows):
        if line - margin <= c <= line + margin:
            return current_map[index + 1][0] if (index + 1) in current_map else 'g4'
        elif line + margin < c <= line + 3 * margin:
            return current_map[index + 1][1] if (index + 1) in current_map else 'f4'
            
    return 'g4' if (region_idx % 2 == 0) else 'd3'

def get_note_name(prev, octave, duration):
    if duration in ['4', 'a_4']:
        return f'{octave[0]}{prev}{octave[1]}/4'
    elif duration in ['8', '8_b_n', '8_b_r', 'a_8']:
        return f'{octave[0]}{prev}{octave[1]}/8'
    elif duration in ['16', '16_b_n', '16_b_r', 'a_16']:
        return f'{octave[0]}{prev}{octave[1]}/16'
    elif duration in ['32', '32_b_n', '32_b_r', 'a_32']:
        return f'{octave[0]}{prev}{octave[1]}/32'
    elif duration in ['2', 'a_2']:
        return f'{octave[0]}{prev}{octave[1]}/2'
    elif duration in ['1', 'a_1']:
        return f'{octave[0]}{prev}{octave[1]}/1'
    else:
        return "c1/4"


def filter_beams(prims, prim_with_staff, bounds):
    n_bounds = []
    n_prims = []
    n_prim_with_staff = []
    for i, prim in enumerate(prims):
        if prim.shape[1] >= 2*prim.shape[0]:
            continue
        else:
            n_bounds.append(bounds[i])
            n_prims.append(prims[i])
            n_prim_with_staff.append(prim_with_staff[i])
    return n_prims, n_prim_with_staff, n_bounds


def get_chord_notation(chord_list):
    chord_res = "{"
    for chord_note in chord_list:
        chord_res += (str(chord_note) + ",")
    chord_res = chord_res[:-1]
    chord_res += "}"
    return chord_res


def procesar_un_pentagrama(args_empaquetados):
    """
    Esta función procesa UN SOLO pentagrama de forma aislada en su propio núcleo.
    """
    (i, img, img_with_staff, spacing, rows, most_common, black_names, 
     ring_names, whole_names, simple_black, disk_size) = args_empaquetados
    
    res = []
    prev = ''
    time_name = ''
    
    # Extraemos las primitivas de este pentagrama específico
    primitives, prim_with_staff, boundary = get_connected_components(img, img_with_staff)
    
    for j, prim in enumerate(primitives):
        # Filtro de velocidad por tamaño (mantenemos la optimización anterior)
        if prim.shape[0] < (most_common * 0.5) and prim.shape[1] < (most_common * 0.5):
            continue
            
        prim = binary_opening(prim, square(np.abs(most_common - spacing)))
        saved_img = (255 * (1 - prim)).astype(np.uint8)
        labels = predict(saved_img)
        label = labels[0]
        
        if label in simple_black:
            c = boundary[j][2]
            note_str = estim_note(int(c), 0, [spacing], [rows], i) # i determina si es Sol o Fa
            res.append(get_note_name(prev, note_str, label))
            
        elif label in black_names:
            test_img = np.copy(prim_with_staff[j])
            test_img = binary_dilation(test_img, disk(disk_size))
            comps, comp_w_staff, bounds = get_connected_components(test_img, prim_with_staff[j])
            comps, comp_w_staff, bounds = filter_beams(comps, comp_w_staff, bounds)
            bounds = [np.array(bound) + disk_size - 2 for bound in bounds]
            
            if len(bounds) > 1 and label not in ['8_b_n', '8_b_r', '16_b_n', '16_b_r', '32_b_n', '32_b_r']:
                l_res = []
                bounds = sorted(bounds, key=lambda b: -b[2])
                for k in range(len(bounds)):
                    note_str = estim_note(boundary[j][0] + bounds[k][2], 0, [spacing], [rows], i)
                    l_res.append(f'{note_str}/4')
                    if k + 1 < len(bounds) and (bounds[k][2] - bounds[k + 1][2]) > 1.5 * spacing:
                        note_str_alt = estim_note(boundary[j][0] + bounds[k][2] - spacing/2, 0, [spacing], [rows], i)
                        l_res.append(f'{note_str_alt}/4')
                res.append(sorted(l_res))
            else:
                for bbox in bounds:
                    c = bbox[2] + boundary[j][0]
                    note_str = estim_note(int(c), 0, [spacing], [rows], i)
                    res.append(get_note_name(prev, note_str, label))
                    
        elif label in ring_names:
            head_img = 1 - binary_fill_holes(1 - prim)
            head_img = binary_closing(head_img, disk(disk_size))
            comps, comp_w_staff, bounds = get_connected_components(head_img, prim_with_staff[j])
            for bbox in bounds:
                c = bbox[2] + boundary[j][0]
                note_str = estim_note(int(c), 0, [spacing], [rows], i)
                res.append(get_note_name(prev, note_str, label))
                
        elif label in whole_names:
            c = boundary[j][2]
            note_str = estim_note(int(c), 0, [spacing], [rows], i)
            res.append(get_note_name(prev, note_str, label))
            
        elif label in ['bar', 'bar_b', 'clef', 'clef_b', 'natural', 'natural_b', 't24', 't24_b', 't44', 't44_b']:
            continue
        elif label in ['#', '#_b']:
            prev = '##' if prim.shape[0] == prim.shape[1] else '#'
        elif label in ['cross']:
            prev = '##'
        elif label in ['flat', 'flat_b']:
            prev = '&&' if prim.shape[1] >= 0.5 * prim.shape[0] else '&'
        elif label in ['dot', 'dot_b', 'p']:
            if len(res) == 0 or (len(res) > 0 and res[-1] in ['flat', 'flat_b', 'cross', '#', '#_b', 't24', 't24_b', 't44', 't44_b']):
                continue
            res[-1] += '.'
        elif label in ['t2', 't4']:
            time_name += label[1]
        if label not in ['flat', 'flat_b', 'cross', '#', '#_b']:
            prev = ''

    # Convertir las notas de este pentagrama a string legible
    notas_pentagrama = []
    for note in res:
        if type(note) == list:
            chord_str = ", ".join([guido_to_readable(n) for n in note])
            notas_pentagrama.append(chord_str)
        else:
            notas_pentagrama.append(guido_to_readable(note))
            
    return notas_pentagrama


# ==============================================================================
# 2. FUNCIÓN RECOGNIZE ADMINISTRADORA DEL POOL MULTITASK
# ==============================================================================
def recognize(out_file, most_common, coord_imgs, imgs_with_staff, imgs_spacing, imgs_rows):
    black_names = ['4', '8', '8_b_n', '8_b_r', '16', '16_b_n', '16_b_r',
                   '32', '32_b_n', '32_b_r', 'a_4', 'a_8', 'a_16', 'a_32', 'chord']
    ring_names = ['2', 'a_2']
    whole_names = ['1', 'a_1']
    simple_black = ['a_4', 'a_8', 'a_16', 'a_32', '4', '8', '16', '32',
                    '8_b_r', '8_b_n', '16_b_r', '16_b_n', '32_b_r', '32_b_n']
    disk_size = most_common / 2

    # Preparamos el empaquetado de argumentos para cada uno de los pentagramas independientes
    tareas = []
    for i in range(len(coord_imgs)):
        argumentos = (
            i, coord_imgs[i], imgs_with_staff[i], imgs_spacing[i], imgs_rows[i],
            most_common, black_names, ring_names, whole_names, simple_black, disk_size
        )
        tareas.append(argumentos)

    # Lanzamos el Pool de Multiprocesamiento usando la cantidad de núcleos disponibles
    # cpu_count() - 1 para dejar un núcleo libre para el sistema operativo
    num_nucleos = max(1, multiprocessing.cpu_count() - 1)
    print(f"-> Launching multitask engine using {num_nucleos} CPU cores in parallel...")
    
    with multiprocessing.Pool(processes=num_nucleos) as pool:
        # pool.map ejecuta 'procesar_un_pentagrama' en paralelo para cada elemento en 'tareas'
        resultados_paralelos = pool.map(procesar_un_pentagrama, tareas)

    # Unificamos los resultados en orden secuencial
    todas_las_notas = []
    for notas_de_bloque in resultados_paralelos:
        todas_las_notas.extend(notas_de_bloque)

    # Formateamos con comas para las pausas en Flutter
    notas_con_pausa = [f"{nota}." for nota in todas_las_notas]
    out_file.write(" ".join(notas_con_pausa))


# ==============================================================================
# 3. FUNCIÓN MAIN
# ==============================================================================
def main(input_path, output_path):
    imgs_path = sorted(glob(os.path.join(input_path, '*')))
    for img_path in imgs_path:
        try:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            out_file = open(os.path.join(output_path, f'{img_name}.txt'), "w")
            print(f"Processing new image {img_name}...")
            img = io.imread(img_path)
            img = gray_img(img)

            # ==============================================================================
            # OPTIMIZACIÓN DE VELOCIDAD: REDIMENSIONAMIENTO DINÁMICO
            # ==============================================================================
            # Si la imagen viene en alta resolución (ej. de la cámara), la bajamos a un ancho 
            # estándar de 1600 píxeles. Esto acelera el procesamiento hasta un 400%.
            if img.shape[1] > 1600:
                print(f"-> Resizing image from {img.shape[1]}px width to 1600px for high performance...")
                scale_factor = 1600 / img.shape[1]
                new_shape = (int(img.shape[0] * scale_factor), 1600)
                # resize devuelve floats entre 0 y 1, lo devolvemos a formato uint8 (0-255)
                img = (resize(img, new_shape, anti_aliasing=True) * 255).astype(np.uint8)
            # ==============================================================================

            horizontal = True
            original = img.copy()
            gray = get_gray(img)
            bin_img = get_thresholded(gray, threshold_otsu(gray))
            
            segmenter = Segmenter(bin_img)
            imgs_with_staff = segmenter.regions_with_staff
            most_common = segmenter.most_common
            imgs_spacing = []
            imgs_rows = []
            coord_imgs = []
            
            for i, img_region in enumerate(imgs_with_staff):
                spacing, rows, no_staff_img = coordinator(img_region, horizontal)
                rows = rows[:5]
                imgs_rows.append(rows)
                imgs_spacing.append(spacing)
                coord_imgs.append(no_staff_img)
                
            print("Recognize...")
            recognize(out_file, most_common, coord_imgs,
                      imgs_with_staff, imgs_spacing, imgs_rows)
            
            out_file.flush()
            out_file.close()
            print("Done...")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help="Input File")
    parser.add_argument("outputfolder", help="Output File")
    args = parser.parse_args()
    main(args.inputfolder, args.outputfolder)
