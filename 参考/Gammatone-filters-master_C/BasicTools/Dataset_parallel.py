import numpy as np

from .ProgressBar import ProgressBar


class Dataset(object):
    def __init__(self, file_reader, file_paths, shuffle_size, batch_size,
                 output_shapes, always_full_batch=False, show_process=False):
        self.file_reader = file_reader

        self.file_paths = file_paths
        np.random.shuffle(self.file_paths)

        self.always_full_batch = always_full_batch
        # ensure each batch has batch_size items
        # if not enough items left in data_bare for a batch, items will be
        # discared

        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.output_shapes = output_shapes
        self.n_output = len(output_shapes)
        self.cur_file_i = 0
        self.data_bare = [np.zeros((0, *shape), dtype=np.float32)
                          for shape in output_shapes]
        self.data_bare_left = [np.zeros((0, *shape), dtype=np.float32)
                               for shape in output_shapes]

        self.show_process = show_process
        self.pb = ProgressBar(len(self.file_paths))

    def _load(self):
        # TODO self.data_bare dtype auto change to float64
        # data left from last loading
        for i in range(self.n_output):
            self.data_bare[i] = np.concatenate(
                [self.data_bare[i].astype(np.float32),
                 self.data_bare_left[i].astype(np.float32)],
                axis=0)
        while (self.data_bare[0].shape[0] < self.shuffle_size
               and not self.is_finish_load()):
            data = self.file_reader(self.file_paths[self.cur_file_i])
            self.cur_file_i = self.cur_file_i + 1

            if self.show_process:
                self.pb.update()

            for i in range(self.n_output):
                self.data_bare[i] = np.concatenate(
                    [self.data_bare[i], data[i].astype(np.float32)],
                    axis=0)
        n_sample_bare = min((self.data_bare[0].shape[0], self.shuffle_size))
        rand_index = np.random.permutation(n_sample_bare).astype(np.int)
        for i in range(self.n_output):
            self.data_bare_left[i] = self.data_bare[i][n_sample_bare:]
            self.data_bare[i] = self.data_bare[i][rand_index]

    def is_finish(self):
        if self.always_full_batch:
            finish = (self.is_finish_load()
                      and self.data_bare[0].shape[0] < self.batch_size)
        else:
            finish = self.is_finish_load() and self.data_bare[0].shape[0] <= 0
        return finish

    def is_finish_load(self):
        return self.cur_file_i >= len(self.file_paths)

    def next_batch(self):
        while (self.data_bare[0].shape[0] < self.batch_size
               and not self.is_finish_load()):
            self._load()
        data_batch = [[] for i in range(self.n_output)]
        for i in range(self.n_output):
            data_batch[i] = self.data_bare[i][:self.batch_size]
            self.data_bare[i] = self.data_bare[i][self.batch_size:]
        return data_batch

    def reset(self):
        self.cur_file_i = 0
        np.random.shuffle(self.file_paths)
        self.data_bare = [np.zeros((0, *shape))
                          for shape in self.output_shapes]
        self.data_bare_left = [np.zeros((0, *shape))
                               for shape in self.output_shapes]
        self.pb = ProgressBar(len(self.file_paths))


class Dataset_combined(object):
    def __init__(self, file_reader, file_paths_1, file_paths_2, shuffle_size,
                 batch_size, output_shapes):
        self.file_reader = file_reader
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.file_paths_1 = file_paths_1
        self.dataset_1 = Dataset(file_reader, file_paths_1, shuffle_size,
                                 batch_size, output_shapes)
        self.file_paths_2 = file_paths_2
        self.dataset_2 = Dataset(file_reader, file_paths_2, shuffle_size,
                                 batch_size, output_shapes)

    def next_batch(self):
        dice = np.random.rand()
        if dice > 0.5 and not self.dataset_1.is_finish():
            return [self.dataset_1.next_batch(), 1]
        elif not self.dataset_2.is_finish():
            return [self.dataset_2.next_batch(), 2]
        else:
            return [self.dataset_1.next_batch(), 1]

    def is_finish(self):
        return self.dataset_1.is_finish() and self.dataset_2.is_finish()

    def reset(self):
        self.dataset_1.reset()
        self.dataset_2.reset()


if __name__ == '__main__':

    def file_reader(fpath_record, is_slice=True):

        frame_len = 320*2
        shift_len = 160
        n_azi = 37

        if isinstance(fpath_record, bytes):
            fpath_record = fpath_record.decode('utf-8')

        *_, test_i, set_type, _, room, fname_tar = fpath_record.split('/')
        azi, wav_i = [np.int16(item) for item in fname_tar[:-4].split('_')]

        # record signal
        record, fs = wav_tools.read(fpath_record)
        wav_r = np.expand_dims(
                    wav_tools.frame_data(record, frame_len, shift_len),
                    axis=-1)

        # direct signal
        fpath_direct = fpath_record.replace('reverb', 'direct')
        direct, fs = wav_tools.read(fpath_direct)
        wav_d = np.expand_dims(
                    wav_tools.frame_data(direct, frame_len, shift_len),
                    axis=-1)

        n_sample = wav_d.shape[0]
        # onehot azi label
        loc_label = np.zeros((n_sample, n_azi))
        loc_label[:, azi] = 1

        return [wav_d, wav_r, loc_label, loc_label]

    from BasicTools.get_file_path import get_file_path
    from BasicTools import wav_tools

    train_set_dir = '../Data/v1/train/reverb/Room_A/'
    file_paths_1 = get_file_path(train_set_dir, '.wav', is_absolute=True)

    train_set_dir = '../Data/v1/train/reverb/Anechoic/'
    file_paths_2 = get_file_path(train_set_dir, '.wav', is_absolute=True)
    # dataset = Dataset_combined(file_reader, file_paths_1, file_paths_2,
    #                            1024*5, 1024,
    #                            [[640, 2, 1], [640, 2, 1], [37]])

    # n_batch = 0
    # while not dataset.is_finish():
    #     batch, dataset_index = dataset.next_batch()
    #     n_batch = n_batch + 1
    # print(n_batch)

    dataset = Dataset(file_reader, file_paths_1, 1024*5, 1024,
                      [[640, 2, 1], [640, 2, 1], [37], [37]])

    n_batch = 0
    while not dataset.is_finish():
        batch = dataset.next_batch()
        n_batch = n_batch + 1
        print(np.argmax(batch[-1][:10], axis=1),
              np.argmax(batch[-2][:10], axis=1))
        input('continue')
    print(n_batch)
