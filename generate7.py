import numpy as np
import os
import svgwrite
from svgpathtools import real, imag, svg2paths
from rdp import rdp

i=0

def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1

def lines_to_strokes(lines, omit_first_point=True):
    """
    Convert polyline format to stroke-3 format.
    lines: list of strokes, each stroke has format Nx2
    """
    strokes = []
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :] if omit_first_point else strokes

def read_svg(svg_path, scale=100.0, draw_mode=False):
    """
    read svg, centralised and convert to stroke-3 format
    scale: stroke-3 output having max dimension [-scale, +scale]
    """
    try:
        paths, path_attrs = svg2paths(svg_path, return_svg_attributes=False)  # svg to paths
        lines = []
        lens = []
        for path_id, path in enumerate(paths):  # get poly lines from path
            erase = False  # path could be erased by setting stroke attribute to #fff (sketchy)
            path_attr = path_attrs[path_id]
            if 'stroke' in path_attr and path_attr['stroke'] == '#fff':
                erase = True
            # try:
            plen = int(path.length())
            # except ZeroDivisionError:
            #     plen = 0
            if plen > 0 and not erase:
                lines.append([path.point(i) for i in np.linspace(0, 1, max(2, plen))])
                lens.append(plen)

        # convert to (x,y) coordinates
        lines = [np.array([[real(x), imag(x)] for x in path]) for path in lines]

        # get dimension of this drawing
        tmp = np.concatenate(lines, axis=0)
        w_max, h_max = np.max(tmp, axis=0)
        w_min, h_min = np.min(tmp, axis=0)
        w = w_max - w_min
        h = h_max - h_min
        max_hw = max(w, h)

        def group(line):
            out = np.array(line, dtype=np.float32)
            out[:, 0] = ((out[:, 0] - w_min) / max_hw * 2.0 - 1.0) * scale
            out[:, 1] = ((out[:, 1] - h_min) / max_hw * 2.0 - 1.0) * scale
            return out

        # normalised
        lines = [group(path) for path in lines]
        lines_simplified = [rdp(path, epsilon=1.5) for path in lines]  # apply RDP algorithm

        strokes_simplified = lines_to_strokes(lines_simplified)  # convert to 3-stroke format (dx,dy,pen_state)
        # scale_bound(strokes_simplified, 10)
        if draw_mode:
            draw_strokes3(strokes_simplified, 1.0)  # no need to concat the origin point
            print('num points: {}'.format(len(strokes_simplified)))
        return np.array(strokes_simplified, dtype=np.float32)
    except Exception as e:
        print('Error encountered: {} - {}'.format(type(e), e))
        print('Location: {}'.format(svg_path))
        raise



def get_bounds(data, factor=1.0):
    """Return bounds of stroke-3 data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return min_x, max_x, min_y, max_y

def draw_strokes3(data, factor=0.2, svg_filename='test.svg', stroke_width=1):
    """
    draw stroke3 to svg
    :param data: stroke3, add origin (0,0) if doesn't have
    :param factor: scale factor
    :param svg_filename: output file
    :return: None
    """
    if np.abs(data[0]).sum() != 0:
        data2 = np.r_[np.zeros((1, 3), dtype=np.float32), data]
    else:
        data2 = data
    parent_dir = os.path.dirname(svg_filename)
    if parent_dir and not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    min_x, max_x, min_y, max_y = get_bounds(data2, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data2)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data2[i, 0]) / factor
        y = float(data2[i, 1]) / factor
        lift_pen = data2[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()

from cairosvg import svg2png

def svg_to_png(in_svg, out_png):

    svg2png(open(in_svg, 'rb').read(), write_to=open(out_png, 'wb'))

if __name__ == '__main__':

    for root, dirs, files in os.walk("/share/kuhu6123/atd12k_svg/test_2k_540p"):
        # for dir in dirs:
        #     print(dir)
        #     os.makedirs('atd12k_svg/train_10k/'+dir)
        #     i+=1
        #     print(len(dirs))
        #     print(os.path.join(root,dir))

        for file in files:
            if file == '.DS_Store':
                continue
            if not file.__contains__('frame'):
                continue
            if file.__contains__('3') or file.__contains__('1'):
                continue
            # if file == 'inter12.jpg' or file == 'inter14.jpg':
            #     continue
            image3 = root + '/' + file
            # image1 = image3.replace('frame3', 'frame1')

            # print(image)

            interpath = image3.replace('frame2', 'frame2vector')
            outimagepath = interpath.replace('.svg', '.png')
            svg3 = read_svg(image3, 100)
            # svg1 = read_svg(image1, 100)
            output = []
            for g3 in svg3:
                out = lerp(g3, g3, 0.5)
                output.append(out)

            draw_strokes3(output, 0.1, svg_filename=interpath)
            # jpg = output.replace('.svg', '.jpg')
            # drawing = svg2rlg(interpath)
            # renderPM.drawToFile(drawing, outimagepath, fmt="JPG")
            svg_to_png(interpath, outimagepath)

            # input_file = 'Japan_v2_3_172427_s2/frame1.jpg'
            # print(image)

            # output_file = 'output.svg'
            # os.system(f'convert {interpath} {outimagepath}')
            # os.system(f'potrace output.ppm -s -o {out}')
            i+=1
            if (i % 100 == 0):
                print(i)

print(i)