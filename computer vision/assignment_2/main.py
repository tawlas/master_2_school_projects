import alassane_watt
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def main():

    data_root = './img1/'
    gt_path = './gt/gt.txt'

    _W = 1280
    _H = 960
    _N = 684  # number of frames

    def format_id(frame):
        assert _N >= frame
        return '{:03d}'.format(frame)

    def read_frame(root, frame):
        """Read frames and create integer frame_id-s"""
        assert _N >= frame
        return cv2.imread(os.path.join(root, format_id(frame)+'.jpg'), cv2.IMREAD_UNCHANGED)

    def read_gt(filename):
        """Read gt and create list of bb-s"""
        assert os.path.exists(filename)
        with open(filename, 'r') as file:
            lines = file.readlines()
        # truncate data (last columns are not needed)
        return [list(map(lambda x: int(x), line.split(',')[:6])) for line in lines]

    def annotations_for_frame(solution, frame):
        assert _N >= frame
        return [bb for bb in solution if int(bb[0]) == int(frame)]

    def evaluate_solution(gt, solution, N):
        """Caclulate evaluation metric"""
        score = []
        # for frame in [300]:
        for frame in tqdm(range(1, N)):
            bbs_sol = annotations_for_frame(solution, frame)
            bbs_gt = annotations_for_frame(gt, frame)

            black_sol = np.zeros((_H, _W))
            black_gt = np.zeros((_H, _W))
            for bb in bbs_sol:
                x, y = bb[2:4]
                dx, dy = bb[4:6]
                cv2.rectangle(black_sol, (x, y), (x+dx, y+dy), (255), -1)
            for bb in bbs_gt:
                x, y = bb[2:4]
                dx, dy = bb[4:6]
                cv2.rectangle(black_gt, (x, y), (x+dx, y+dy), (255), -1)
            # intersection over union
            intersection = black_sol * black_gt
            intersection[intersection > 0.5] = 1
            union = black_sol + black_gt
            union[union > 0.5] = 1
            if not union.any():
                continue
            score.append(intersection.sum()/union.sum())

        return np.asarray(score).mean()

    def show_annotation(solution, frame):
        assert _N >= frame
        im = read_frame(data_root, frame)
        bbs = annotations_for_frame(solution, frame)
        for bb in bbs:
            x, y = bb[2:4]
            dx, dy = bb[4:6]
            cv2.rectangle(im, (x, y), (x+dx, y+dy), (0, 255, 0), 10)
        plt.imshow(im)
        plt.title('Annotations for frame {}.'.format(frame))
        plt.show()
        return

    gt = read_gt(gt_path)

    # show_annotation(gt, 2)
    print("Computing the perfect score that can be achieved...")

    # print('A perfect score... {}'.format(evaluate_solution(gt, gt, _N)))

    print("Computing the solution...")

    # your solution will be tested simply by changing the dataset
    # and changing the module, i.e., the following has to work
    # with simply using your module
    sol = alassane_watt.pedestrians(data_root, _W, _H, _N)
    print("Evaluating the solution...")
    show_annotation(sol, 102)
    print('A great score! {}'.format(evaluate_solution(sol, gt, _N)))


if __name__ == '__main__':
    main()
