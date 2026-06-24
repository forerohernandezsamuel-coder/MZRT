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

label_map = {
    0: {0: 'N0'},
    1: {0: 'b2', 1: 'a2'},
    2: {0: 'g2', 1: 'f2'},
    3: {0: 'e2', 1: 'd2'},
    4: {0: 'c2', 1: 'b1'},
    5: {0: 'a1', 1: 'g1'},
    6: {0: 'f1', 1: 'e1'},
    7: {0: 'd1', 1: 'c1'}
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


def estim(c, idx, imgs_spacing, imgs_rows):
    spacing = imgs_spacing[idx]
    rows = imgs_rows[idx]
    margin = 1+(spacing/4)
    for index, line in enumerate(rows):
        if c >= line - margin and c <= line + margin:
            return index+1, 0
        elif c >= line + margin and c <= line + 3*margin:
            return index+1, 1
    return 7, 1


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


def recognize(out_file, most_common, coord_imgs, imgs_with_staff, imgs_spacing, imgs_rows):
    black_names = ['4', '8', '8_b_n', '8_b_r', '16', '16_b_n', '16_b_r',
                   '32', '32_b_n', '32_b_r', 'a_4', 'a_8', 'a_16', 'a_32', 'chord']
    ring_names = ['2', 'a_2']
    whole_names = ['1', 'a_1']
    simple_black = ['a_4', 'a_8', 'a_16', 'a_32', '4', '8', '16', '32',
                    '8_b_r', '8_b_n', '16_b_r', '16_b_n', '32_b_r', '32_b_n']
    disk_size = most_common / 2
    if len(coord_imgs) > 1:
        out_file.write("{\n")
    for i, img in enumerate(coord_imgs):
        res = []
        prev = ''
        time_name = ''
        primitives, prim_with_staff, boundary = get_connected_components(
            img, imgs_with_staff[i])
        for j, prim in enumerate(primitives):
            prim = binary_opening(prim, square(
                np.abs(most_common-imgs_spacing[i])))
            saved_img = (255*(1 - prim)).astype(np.uint8)
            labels = predict(saved_img)
            octave = None
            label = labels[0]
            if label in simple_black:
                c = boundary[j][2]
                line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                l = label_map[line_idx][p]
                res.append(get_note_name(prev, l, label))
            elif label in black_names:
                test_img = np.copy(prim_with_staff[j])
                test_img = binary_dilation(test_img, disk(disk_size))
                comps, comp_w_staff, bounds = get_connected_components(
                    test_img, prim_with_staff[j])
                comps, comp_w_staff, bounds = filter_beams(
                    comps, comp_w_staff, bounds)
                bounds = [np.array(bound)+disk_size-2 for bound in bounds]
                if len(bounds) > 1 and label not in ['8_b_n', '8_b_r', '16_b_n', '16_b_r', '32_b_n', '32_b_r']:
                    l_res = []
                    bounds = sorted(bounds, key=lambda b: -b[2])
                    for k in range(len(bounds)):
                        idx, p = estim(
                            boundary[j][0]+bounds[k][2], i, imgs_spacing, imgs_rows)
                        l_res.append(f'{label_map[idx][p]}/4')
                        if k+1 < len(bounds) and (bounds[k][2]-bounds[k+1][2]) > 1.5*imgs_spacing[i]:
                            idx, p = estim(
                                boundary[j][0]+bounds[k][2]-imgs_spacing[i]/2, i, imgs_spacing, imgs_rows)
                            l_res.append(f'{label_map[idx][p]}/4')
                    res.append(sorted(l_res))
                else:
                    for bbox in bounds:
                        c = bbox[2]+boundary[j][0]
                        line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                        l = label_map[line_idx][p]
                        res.append(get_note_name(prev, l, label))
            elif label in ring_names:
                head_img = 1-binary_fill_holes(1-prim)
                head_img = binary_closing(head_img, disk(disk_size))
                comps, comp_w_staff, bounds = get_connected_components(
                    head_img, prim_with_staff[j])
                for bbox in bounds:
                    c = bbox[2]+boundary[j][0]
                    line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                    l = label_map[line_idx][p]
                    res.append(get_note_name(prev, l, label))
            elif label in whole_names:
                c = boundary[j][2]
                line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                l = label_map[line_idx][p]
                res.append(get_note_name(prev, l, label))
            elif label in ['bar', 'bar_b', 'clef', 'clef_b', 'natural', 'natural_b', 't24', 't24_b', 't44', 't44_b'] or label in []:
                continue
            elif label in ['#', '#_b']:
                if prim.shape[0] == prim.shape[1]:
                    prev = '##'
                else:
                    prev = '#'
            elif label in ['cross']:
                prev = '##'
            elif label in ['flat', 'flat_b']:
                if prim.shape[1] >= 0.5*prim.shape[0]:
                    prev = '&&'
                else:
                    prev = '&'
            elif label in ['dot', 'dot_b', 'p']:
                if len(res) == 0 or (len(res) > 0 and res[-1] in ['flat', 'flat_b', 'cross', '#', '#_b', 't24', 't24_b', 't44', 't44_b']):
                    continue
                res[-1] += '.'
            elif label in ['t2', 't4']:
                time_name += label[1]
            elif label == 'chord':
                img = thin(1-prim.copy(), max_iter=20)
                head_img = binary_closing(1-img, disk(disk_size))
            if label not in ['flat', 'flat_b', 'cross', '#', '#_b']:
                prev = ''

        out_file.write("=== Pentagrama {} ===\n".format(i+1))
        for note in res:
            if type(note) == list:
                chord_str = ", ".join([guido_to_readable(n) for n in note])
                out_file.write("[ {} ]\n".format(chord_str))
            else:
                out_file.write(guido_to_readable(note) + "\n")
        out_file.write("\n")

    if len(coord_imgs) > 1:
        out_file.write("}")
    print("###########################", res, "##########################")


def main(input_path, output_path):
    imgs_path = sorted(glob(os.path.join(input_path, '*')))
    for img_path in imgs_path:
        try:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            out_file = open(os.path.join(output_path, f'{img_name}.txt'), "w")
            print(f"Processing new image {img_name}...")
            img = io.imread(img_path)
            img = gray_img(img)
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
            for i, img in enumerate(imgs_with_staff):
                spacing, rows, no_staff_img = coordinator(img, horizontal)
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

#ultimate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help="Input File")
    parser.add_argument("outputfolder", help="Output File")
    args = parser.parse_args()
    main(args.inputfolder, args.outputfolder)
