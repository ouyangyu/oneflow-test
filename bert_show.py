import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="flags for cnn benchmark")
parser.add_argument(
    "--log_file", type=str, default="./bert_b32_fp32_1.log",
    required=False)
parser.add_argument("--type", type=str, default='tf')
args = parser.parse_args()


def extract_info_from_file(log_file):

    result_y = []
    result_x = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] == 'step:':
                result_x.append(int(ss[1][0:-1])+1)
                result_y.append(float(ss[3][0:-1]))
            elif ss[0] == 'DLL':
                if ss[4] == 'Iteration:':
                    result_x.append(int(ss[5]))
                    result_y.append(float(ss[21]))

    return result_x,result_y

if __name__ == "__main__":
    tf_x, tf_y = extract_info_from_file('./bert_b32_fp32_1.log')
    of_x, of_y = extract_info_from_file('./of_bert_fp32_b32_oneflow.log')
    l1, = plt.plot(tf_x, tf_y, label='linear line')
    #l2, = plt.plot(of_x, of_y, color='red', linewidth=1.0, linestyle='--', label='square line')
    #plt.legend(loc = 'upper right')
    plt.show()

