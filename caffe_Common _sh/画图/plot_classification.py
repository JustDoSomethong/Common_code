from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv
import os
import re


## extract seconds
def extract_datetime_from_line(line, year):
    # Expected format: I0210 13:39:22.381027 25210 solver.cpp:204] Iteration 100, lr = 0.00992565
    line = line.strip().split()
    month = int(line[0][1:3])
    day = int(line[0][3:])
    timestamp = line[1]
    pos = timestamp.rfind('.')
    ts = [int(x) for x in timestamp[:pos].split(':')]
    hour = ts[0]
    minute = ts[1]
    second = ts[2]
    microsecond = int(timestamp[pos + 1:])
    dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)
    return dt


def get_log_created_year(input_file):
    """Get year from log file system timestamp
    """

    log_created_time = os.path.getctime(input_file)
    log_created_year = datetime.datetime.fromtimestamp(log_created_time).year
    return log_created_year


def get_start_time(line_iterable, year):
    """Find start time from group of lines
    """

    start_datetime = None
    for line in line_iterable:
        line = line.strip()
        if line.find('Solving') != -1:
            start_datetime = extract_datetime_from_line(line, year)
            break
    return start_datetime


def extract_seconds(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    log_created_year = get_log_created_year(input_file)
    start_datetime = get_start_time(lines, log_created_year)
    assert start_datetime, 'Start time not found'

    out = open(output_file, 'w')
    for line in lines:
        line = line.strip()
        if line.find('Iteration') != -1:
            dt = extract_datetime_from_line(line, log_created_year)
            elapsed_seconds = (dt - start_datetime).total_seconds()
            out.write('%f\n' % elapsed_seconds)
    out.close()


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_row = None
    test_row = None

    logfile_year = get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = get_start_time(f, logfile_year)

        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = int(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_datetime_from_line(line,
                                                  logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            seconds = (time - start_time).total_seconds()

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))

            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list,
                line, iteration, seconds, learning_rate
            )
            test_dict_list, test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list,
                line, iteration, seconds, learning_rate
            )

    fix_initial_nan_learning_rate(train_dict_list)
    fix_initial_nan_learning_rate(test_dict_list)

    return train_dict_list, test_dict_list


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['iter'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('iter', iteration),
                ('sec', seconds),
                ('lr', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['lr'] = dict_list[1]['lr']


def save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list,
                   delimiter='\t', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    # train_filename = os.path.join(output_dir, log_basename + '.train')
    train_filename = os.path.join('0.train')
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    # test_filename = os.path.join(output_dir, log_basename + '.test')
    test_filename = os.path.join('0.test')
    write_csv(test_filename, test_dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter='\t', verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def extract_inf(parse_file, start_iter=0):
    with open(parse_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    n = len(re.split(r'\s+', lines[0]))
    train_inf = dict()
    for i in range(n):
        train_inf[i] = list()
    for i in range(1, len(lines)):
        inf = re.split(r'\s+', lines[i])
        if len(inf) == n and start_iter <= float(inf[0]):
            for j in range(n):
                train_inf[j].append(float(inf[j]))
    for key in train_inf:
        train_inf[key] = np.array(train_inf[key], dtype=np.float)
    return train_inf


def plot(y_label, x, y, color, linewidth, grid, marker, extreme_point_type, extreme_point_color):
    try:
        imagename = y_label + ' v.s. iter'
        plt.figure(imagename)
        plt.plot(x, y, color=color, marker=marker, linewidth=linewidth)
        plt.grid(grid)

        if extreme_point_type == 'min':
            extreme_point_x = np.argmin(y)
            extreme_point_y = y[extreme_point_x]
            extreme_point_x = int(x[extreme_point_x])
            plt.axvline(extreme_point_x, ls="--", color=extreme_point_color)
            plt.axhline(extreme_point_y, ls="--", color=extreme_point_color)
        elif extreme_point_type == 'max':
            extreme_point_x = np.argmax(y)
            extreme_point_y = y[extreme_point_x]
            extreme_point_x = int(x[extreme_point_x])
            plt.axvline(extreme_point_x, ls="--", color=extreme_point_color)
            plt.axhline(extreme_point_y, ls="--", color=extreme_point_color)
        plt.xlabel('Iters')
        plt.ylabel(y_label)
        if extreme_point_type == 'min' or extreme_point_type == 'max':
            plt.title(imagename + '\n' + extreme_point_type
                      + ' = ' + '(' + str(extreme_point_x) + ',  ' + str(extreme_point_y) + ')')
        else:
            plt.title(imagename)
        plt.savefig(imagename + '.png')
    except:
        print 'plot %s failed' % imagename


def plot2(y_label, grid, linewidth, x1, y1, color1, marker1, extreme_point_type1,
          x2, y2, color2, marker2, extreme_point_type2):
    try:
        imagename = y_label + ' v.s. iter'
        legend = re.split(r'\s?&\s?',y_label)
        plt.figure(imagename)
        plt.plot(x1, y1, color=color1, marker=marker1, linewidth=linewidth)
        plt.plot(x2, y2, color=color2, marker=marker2, linewidth=linewidth)
        plt.legend(legend)
        plt.grid(grid)

        if extreme_point_type1 == 'min':
            extreme_point_x1 = np.argmin(y1)
            extreme_point_y1 = y1[extreme_point_x1]
            extreme_point_x1 = int(x1[extreme_point_x1])
            plt.axvline(extreme_point_x1, ls="--", color=color1)
            plt.axhline(extreme_point_y1, ls="--", color=color1)
        elif extreme_point_type1 == 'max':
            extreme_point_x1 = np.argmax(y1)
            extreme_point_y1 = y1[extreme_point_x1]
            extreme_point_x1 = int(x1[extreme_point_x1])
            plt.axvline(extreme_point_x1, ls="--", color=color1)
            plt.axhline(extreme_point_y1, ls="--", color=color1)

        if extreme_point_type2 == 'min':
            extreme_point_x2 = np.argmin(y2)
            extreme_point_y2 = y2[extreme_point_x2]
            extreme_point_x2 = int(x2[extreme_point_x2])
            plt.axvline(extreme_point_x2, ls="--", color=color2)
            plt.axhline(extreme_point_y2, ls="--", color=color2)
        elif extreme_point_type2 == 'max':
            extreme_point_x2 = np.argmax(y2)
            extreme_point_y2 = y2[extreme_point_x2]
            extreme_point_x2 = int(x2[extreme_point_x2])
            plt.axvline(extreme_point_x2, ls="--", color=color2)
            plt.axhline(extreme_point_y2, ls="--", color=color2)

        plt.xlabel('Iters')
        plt.ylabel(y_label)
        plt.title(imagename)
        plt.savefig(imagename + '.png')
    except:
        print 'plot %s failed' % imagename


if __name__ == '__main__':
    log_path = '/home/kb539/Desktop/kb539/YH/SE-BN-Inception_iter_class_center_loss/output/out.log'
    if os.path.isdir(log_path):
        try:
            log_path_list = os.listdir(log_path)
            log_path_list = [log for log in log_path_list if os.path.splitext(log)[-1]=='.log']
            log_path_list.sort()
            log_path = os.path.join(log_path, log_path_list[-1])
        except:
            print 'invalid log file folder.'
            exit(-1)
    train_dict_list, test_dict_list = parse_log(log_path)
    save_csv_files(log_path, './', train_dict_list, test_dict_list)

    iter_threshold = 0
    train_inf = extract_inf('0.train', iter_threshold)
    plot(y_label='Training Loss', x=train_inf[0], y=train_inf[3],
         color=[68.0 / 255, 116.0 / 255, 187.0 / 255], linewidth=1.5, grid=True,
         marker='o', extreme_point_type='min', extreme_point_color='r')
    plot(y_label='Training Learning Rate', x=train_inf[0], y=train_inf[2],
         color=[68.0 / 255, 116.0 / 255, 187.0 / 255], linewidth=1.5, grid=True,
         marker='', extreme_point_type='', extreme_point_color='')
    if not len(test_dict_list):
        print 'cannot plot test learning curve until next net testing iter.'
    else:
        test_inf = extract_inf('0.test', iter_threshold)
        plot(y_label='Test Accuracy', x=test_inf[0], y=test_inf[3],
             color=[111.0 / 255, 145.0 / 255, 125.0 / 255], linewidth=1.5, grid=True,
             marker='h', extreme_point_type='max', extreme_point_color='r')
        plot(y_label='Test Loss', x=test_inf[0], y=test_inf[4],
             color=[111.0 / 255, 145.0 / 255, 125.0 / 255], linewidth=1.5, grid=True,
             marker='h', extreme_point_type='min', extreme_point_color='r')

        plot2(y_label='Training Loss & Test Loss', grid=True, linewidth=1.5,
              x1=train_inf[0], y1=train_inf[3], color1=[68.0 / 255, 116.0 / 255, 187.0 / 255], marker1='o',extreme_point_type1='min',
              x2=test_inf[0], y2=test_inf[4],color2=[213.0 / 255, 77.0 / 255, 43.0 / 255],marker2='h', extreme_point_type2='min'
              )
